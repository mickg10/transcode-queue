"""Hardware FFmpeg frames piped into Python, locally or over SSH.

This is deliberately separate from OpenCV VideoCapture: video decoding, resizing,
and frame sampling happen in FFmpeg on the GPU host. Python receives only the
sampled BGR frames. Failure to initialize the hardware decoder is an error.
"""
from __future__ import annotations

import dataclasses
import shlex
import subprocess
import tempfile
from collections.abc import Iterator


@dataclasses.dataclass(frozen=True)
class DecodeConfig:
    source: str
    width: int = 1152
    height: int = 768
    fps: float = 2
    start: float = 0
    duration: float | None = None
    codec: str = "hevc"
    ffmpeg: str = "ffmpeg"
    ssh_host: str | None = None
    # Example: ("docker","run","--rm","--network","none","--runtime","nvidia-runtime",
    # "-e","NVIDIA_VISIBLE_DEVICES=all","-e","NVIDIA_DRIVER_CAPABILITIES=compute,video",
    # "-v","/share/media/photos:/media/photos:ro","--entrypoint","ffmpeg","IMAGE")
    command_prefix: tuple[str, ...] = ()


def hardware_command(config: DecodeConfig) -> list[str]:
    if config.codec not in {"hevc", "h264"}:
        raise ValueError("This hardware frame reader supports HEVC and H.264")
    if config.width < 2 or config.height < 2 or config.width % 2 or config.height % 2:
        raise ValueError("Analysis dimensions must be positive and even")
    if config.fps <= 0 or config.fps > 60 or config.start < 0:
        raise ValueError("Invalid sampling interval")
    cmd = list(config.command_prefix) or [config.ffmpeg]
    cmd += ["-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", config.codec + "_cuvid", "-resize", f"{config.width}x{config.height}"]
    if config.start:
        cmd += ["-ss", str(config.start)]
    cmd += ["-i", config.source]
    if config.duration is not None:
        if config.duration <= 0:
            raise ValueError("Duration must be positive")
        cmd += ["-t", str(config.duration)]
    cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
            f"fps={config.fps},scale_cuda={config.width}:{config.height}:format=nv12,"
            "hwdownload,format=nv12,format=bgr24",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    if config.ssh_host:
        return ["ssh", "-T", "-o", "BatchMode=yes", config.ssh_host, shlex.join(cmd)]
    return cmd


def frames(config: DecodeConfig) -> Iterator[tuple[float, "numpy.ndarray"]]:
    import numpy as np
    command = hardware_command(config)
    frame_bytes = config.width * config.height * 3
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
        index = 0
        try:
            while True:
                data = bytearray()
                while len(data) < frame_bytes:
                    chunk = process.stdout.read(frame_bytes - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                if not data:
                    break
                if len(data) != frame_bytes:
                    raise RuntimeError("Hardware FFmpeg returned an incomplete frame")
                array = np.frombuffer(data, dtype=np.uint8).reshape(config.height, config.width, 3)
                yield config.start + index / config.fps, array
                index += 1
            code = process.wait()
            if code or not index:
                stderr.seek(0, 2)
                stderr.seek(max(0, stderr.tell() - 4000))
                raise RuntimeError("Hardware decoding failed; no software fallback was used:\n"
                                   + stderr.read().decode(errors="replace"))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
