from __future__ import annotations

import fcntl
import json
import logging
import os
import selectors
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .core import (MediaPaths, Store, dimensions, duration, legacy_outputs, probe,
                   same_duration, timecode, video)

log = logging.getLogger(__name__)


def command(source: Path, output: Path, meta: dict, preset: dict, backend: str,
            ffmpeg: str = "ffmpeg") -> list[str]:
    w, h = dimensions(meta, preset)
    ten = preset["bit_depth"] == 10
    fmt = "p010le" if ten else "nv12"
    software_fmt = "yuv420p10le" if ten else "yuv420p"
    v = video(meta)
    args = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info"]
    if backend == "nvidia":
        args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if v["codec_name"] in {"hevc", "h264"}:
            args += ["-c:v", v["codec_name"] + "_cuvid", "-resize", f"{w}x{h}"]
        args += ["-i", str(source), "-vf", f"scale_cuda={w}:{h}:format={fmt},setsar=1"]
    else:
        args += ["-threads", "4", "-i", str(source), "-vf",
                 f"scale={w}:{h}:flags=bicubic,format={software_fmt},setsar=1"]
    args += ["-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0"]
    if backend in {"nvidia", "nvidia-encode"}:
        args += ["-c:v", preset["codec"] + "_nvenc", "-preset", preset["nvenc_preset"],
                 "-rc", "vbr", "-cq", str(preset["quality"])]
    else:
        args += ["-c:v", "libx265" if preset["codec"] == "hevc" else "libx264",
                 "-preset", "fast", "-crf", str(preset["quality"]), "-threads", "4"]
        if preset["codec"] == "hevc":
            args += ["-x265-params", "pools=4:frame-threads=2"]
    args += ["-b:v", f'{preset["bitrate_mbps"]}M', "-maxrate", f'{preset["maxrate_mbps"]}M',
             "-bufsize", f'{preset["maxrate_mbps"] * 2}M', "-g", str(preset["gop"]),
             "-bf", "0", "-fps_mode", "passthrough"]
    if preset["codec"] == "hevc":
        args += ["-tag:v", "hvc1"]
    rate = v.get("avg_frame_rate", "0/1")
    numerator = int(rate.split("/")[0])
    if numerator > 0:
        args += ["-video_track_timescale", str(numerator)]
    args += ["-c:a", "copy" if preset["audio"] == "copy" else "aac"]
    if preset["audio"] == "aac":
        args += ["-b:a", "192k"]
    for field, option in (("color_range", "-color_range"), ("color_space", "-colorspace"),
                          ("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc")):
        if v.get(field) not in (None, "unknown", "unspecified"):
            args += [option, v[field]]
    if timecode(meta):
        args += ["-timecode", timecode(meta)]
    return args + ["-movflags", "+faststart", "-progress", "pipe:1", "-n", str(output)]


def validate_output(source: dict, output: dict, preset: dict):
    if not same_duration(source, output):
        raise ValueError("Output duration does not match the source")
    a, b = video(source), video(output)
    if (b.get("width"), b.get("height")) != dimensions(source, preset):
        raise ValueError("Output dimensions do not match the preset")
    if a.get("nb_frames") and b.get("nb_frames") and a["nb_frames"] != b["nb_frames"]:
        raise ValueError("Output video frame count differs from the source")
    if timecode(source) and timecode(source) != timecode(output):
        raise ValueError("Source timecode was not preserved")
    aa = [s for s in source["streams"] if s.get("codec_type") == "audio"]
    bb = [s for s in output["streams"] if s.get("codec_type") == "audio"]
    if len(aa) != len(bb):
        raise ValueError("Audio track count changed")
    if preset["audio"] == "copy":
        for x, y in zip(aa, bb):
            for key in ("codec_name", "sample_rate", "channels", "bits_per_raw_sample"):
                if x.get(key) != y.get(key):
                    raise ValueError("An audio stream property changed: " + key)
            if x.get("duration") and y.get("duration") and abs(float(x["duration"]) - float(y["duration"])) > .001:
                raise ValueError("Audio duration changed")
    for key in ("color_range", "color_space", "color_primaries", "color_transfer"):
        if a.get(key, "unknown") != b.get(key, "unknown"):
            raise ValueError("Color metadata changed: " + key)


class Worker:
    def __init__(self, store: Store, paths: MediaPaths, ffmpeg="ffmpeg", ffprobe="ffprobe",
                 legacy_manifest: Path | None = None):
        self.store, self.paths = store, paths
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe
        self.legacy_manifest = legacy_manifest
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        self.waiting_reason = ""
        self.lock_file = None

    def start(self):
        self.lock_file = (self.store.directory / "worker.lock").open("a")
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET status='queued',progress=0,message='Recovered after restart' WHERE status='running'")
        self.thread = threading.Thread(target=self.loop, name="transcode-worker", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=15)
        if self.lock_file:
            self.lock_file.close()

    def external_busy(self):
        self.waiting_reason = ""
        if self.legacy_manifest and self.legacy_manifest.exists():
            try:
                state = json.loads(self.legacy_manifest.read_text())
                if any(j.get("status") == "encoding" for j in state.get("jobs", [])):
                    self.waiting_reason = "Waiting for the existing proxy batch to finish"
                    return True
            except (OSError, ValueError):
                self.waiting_reason = "Cannot read the existing batch status"
                return True
        return False

    def loop(self):
        while not self.stop_event.is_set():
            if self.store.setting("paused", True) or self.external_busy():
                self.stop_event.wait(1)
                continue
            job = self.store.claim()
            if not job:
                self.stop_event.wait(1)
                continue
            try:
                self.run(job)
            except Exception as exc:
                log.exception("Job %s failed", job["id"])
                current = self.store.get(job["id"])
                if self.stop_event.is_set():
                    self.store.update(job["id"], status="queued", message="Interrupted; will retry after restart", progress=0)
                elif current["cancel_requested"]:
                    self.store.update(job["id"], status="cancelled", finished=time.time(), message="Cancelled")
                else:
                    self.store.update(job["id"], status="failed", finished=time.time(), message=str(exc)[-1600:])

    def run(self, job):
        identifier = job["id"]
        preset = json.loads(job["preset"])
        source = self.paths.resolve(job["source"], must_exist=True)
        output = self.paths.resolve(job["output"])
        before = source.stat()
        meta = probe(source, self.ffprobe)
        seconds = duration(meta)
        self.store.update(identifier, source_bytes=before.st_size, duration=seconds)
        candidates = [job["output"]]
        if preset.get("reuse_legacy_proxies"):
            candidates += legacy_outputs(job["source"])
        for relative in dict.fromkeys(candidates):
            candidate = self.paths.resolve(relative)
            if candidate == source or not candidate.is_file():
                continue
            try:
                if same_duration(meta, probe(candidate, self.ffprobe)):
                    self.store.update(identifier, status="skipped", finished=time.time(), progress=1,
                                      message="Existing proxy has matching duration", actual_output=relative,
                                      output_bytes=candidate.stat().st_size)
                    return
            except (OSError, ValueError):
                continue
        if self.store.get(identifier)["cancel_requested"]:
            raise RuntimeError("Cancelled")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Resolve again after creating the directory; never overwrite the camera source.
        output = self.paths.resolve(job["output"])
        if output == source:
            raise ValueError("Output resolves to its source")
        required = int(seconds * (preset["maxrate_mbps"] + 8) * 125000 * 1.15)
        if shutil.disk_usage(output.parent).free < required:
            raise RuntimeError("Insufficient free space for this job")
        partial = output.with_name(f".{output.stem}.job-{identifier}.partial{output.suffix}")
        backend = preset["backend"]
        modes = ["nvidia", "nvidia-encode", "software"] if backend == "auto" else [backend]
        error = ""
        for mode in modes:
            if partial.exists():
                partial.unlink()  # Only this job's uncommitted temporary output.
            self.store.update(identifier, backend=mode, message="Encoding", progress=0)
            cmd = command(source, partial, meta, preset, mode, self.ffmpeg)
            result, error, last_frame = self.encode(identifier, cmd, seconds, mode)
            if result == 0:
                break
            if self.stop_event.is_set() or self.store.get(identifier)["cancel_requested"]:
                raise RuntimeError("Interrupted")
            # A failure after frames were produced is a media/runtime error, not a capability probe.
            if last_frame > 0 or backend != "auto":
                raise RuntimeError(error)
        else:
            raise RuntimeError(error)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Source changed while encoding; temporary output was not published")
        self.store.update(identifier, message="Validating output", progress=.995)
        validate_output(meta, probe(partial, self.ffprobe), preset)
        if self.stop_event.is_set() or self.store.get(identifier)["cancel_requested"]:
            raise RuntimeError("Interrupted before publication")
        if output.exists():
            previous = output.with_name(output.name + f".previous-{identifier}")
            if previous.exists():
                raise RuntimeError("A previous output backup already exists")
            output.replace(previous)
        partial.replace(output)
        self.store.update(identifier, status="complete", finished=time.time(), progress=1,
                          message="Validated", actual_output=job["output"], output_bytes=output.stat().st_size)

    def encode(self, identifier, cmd, seconds, mode):
        log_dir = self.store.directory / "logs"
        log_dir.mkdir(exist_ok=True)
        logfile = log_dir / f"{identifier}-{mode}.log"
        last_frame = 0
        with logfile.open("w") as err:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=0)
            selector = selectors.DefaultSelector()
            selector.register(self.process.stdout, selectors.EVENT_READ)
            fields = {}
            pending = b""
            eof = False
            try:
                while not eof:
                    if self.stop_event.is_set() or self.store.get(identifier)["cancel_requested"]:
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                        break
                    for key, _ in selector.select(timeout=.5):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            eof = True
                            break
                        pending += chunk
                        while b"\n" in pending:
                            raw, pending = pending.split(b"\n", 1)
                            line = raw.decode(errors="replace").strip()
                            if "=" not in line:
                                continue
                            name, value = line.split("=", 1)
                            fields[name] = value
                            if name == "frame":
                                last_frame = int(value)
                            if name == "progress":
                                self.store.update(identifier,
                                                  progress=min(.99, max(0, float(fields.get("out_time_us", 0)) / 1e6 / seconds)),
                                                  fps=float(fields.get("fps", 0)), speed=fields.get("speed", ""))
                code = self.process.wait()
            finally:
                selector.close()
                self.process.stdout.close()
                self.process = None
        with logfile.open("rb") as f:
            f.seek(max(0, logfile.stat().st_size - 1600))
            error = f.read().decode(errors="replace")
        return code, error or f"FFmpeg exited with {code}", last_frame
