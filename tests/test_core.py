import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.core import C50_PRESET, MediaPaths, Store, dimensions, same_duration
from app.worker import command, validate_output
from analysis.core import DecodeConfig, hardware_command


def meta(seconds=10, frames="240"):
    return {"streams": [{"codec_type": "video", "codec_name": "hevc",
                         "width": 6912, "height": 4608, "duration": str(seconds),
                         "avg_frame_rate": "24000/1001", "nb_frames": frames,
                         "color_range": "pc", "color_space": "bt709"}],
            "format": {"duration": str(seconds), "tags": {"timecode": "17:21:27:12"}}}


def test_paths_reject_traversal_and_unmounted_symlinks(tmp_path):
    root = tmp_path / "media"; root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    paths = MediaPaths(root)
    for bad in ("../outside/file.mp4", "/etc/passwd", "escape/file.mp4", r"..\outside"):
        with pytest.raises(ValueError):
            paths.resolve(bad)
    assert paths.resolve("safe/proxy/movie.mp4") == root / "safe/proxy/movie.mp4"


def test_explicit_alias_stays_in_its_own_subtree(tmp_path):
    root = tmp_path / "photos"; root.mkdir()
    stage = tmp_path / "stage"; stage.mkdir()
    (stage / "a.mp4").touch()
    paths = MediaPaths(root, {"staging_m3": str(stage)})
    assert paths.resolve("staging_m3/a.mp4") == stage / "a.mp4"
    assert paths.children()[0]["name"] == "staging_m3"
    with pytest.raises(ValueError):
        paths.resolve("staging_m3/../elsewhere")


def test_scan_omits_outputs_and_orders_by_capture_date(tmp_path):
    for name in ("A_C001H260901_120000Z.MP4", "A_C002H260830_120000Z.MP4", "a_Proxy.mov"):
        (tmp_path / name).touch()
    for directory in ("proxy", "Proxies", ".rsync-partial", "render"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "ignored.mp4").touch()
    result = MediaPaths(tmp_path).scan("")
    assert len(result) == 2
    assert "260901" in result[0]["name"]


def test_duration_uses_one_frame_tolerance():
    assert same_duration(meta(10), meta(10.02))
    assert not same_duration(meta(10), meta(10.1))
    assert not same_duration(meta(10), meta(0))
    assert not same_duration(meta(10), {"streams": []})


def test_queue_is_atomic_and_presets_are_snapshots(tmp_path):
    media = tmp_path / "media"; media.mkdir()
    source = "A_C001H260901_120000Z.MP4"; (media / source).touch()
    store = Store(tmp_path / "data"); paths = MediaPaths(media)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.enqueue(paths, [source], "c50-proxy"), range(8)))
    assert sum(not r[0]["duplicate"] for r in results) == 1
    assert len(store.jobs()) == 1
    with store.connect() as db:
        changed = {**C50_PRESET, "quality": 30}
        db.execute("UPDATE presets SET body=?", (json.dumps(changed),))
    assert json.loads(store.jobs()[0]["preset"])["quality"] == 21
    assert store.claim()["source"] == source
    assert store.claim() is None


def test_source_cannot_be_destination(tmp_path):
    (tmp_path / "a.mp4").touch()
    with pytest.raises(ValueError, match="replace its source"):
        Store(tmp_path / "data").enqueue(MediaPaths(tmp_path), ["a.mp4"], "c50-proxy", "a.mp4")


def test_hardware_transcode_command_keeps_audio_and_timecode():
    cmd = command(Path("/media/a weird; name.mp4"), Path("/media/proxy/a.mp4"), meta(), C50_PRESET, "nvidia")
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert "hevc_cuvid" in cmd and "hevc_nvenc" in cmd
    assert "0:a?" in cmd and "17:21:27:12" in cmd
    assert "/media/a weird; name.mp4" in cmd
    assert dimensions(meta(), C50_PRESET) == (3456, 2304)
    assert "hwdownload" not in " ".join(cmd)


def test_full_validation_rejects_lost_audio_or_frames():
    source = meta()
    output = meta(frames="239")
    output["streams"][0].update(width=3456, height=2304)
    with pytest.raises(ValueError, match="frame count"):
        validate_output(source, output, C50_PRESET)


def test_analysis_uses_hardware_ffmpeg_and_quotes_remote_source():
    cmd = hardware_command(DecodeConfig(source="/media/clip'; echo BAD.mp4", ssh_host="gpu-host"))
    assert cmd[:4] == ["ssh", "-T", "-o", "BatchMode=yes"]
    import shlex
    remote = shlex.split(cmd[-1])
    assert remote[remote.index("-i")+1] == "/media/clip'; echo BAD.mp4"
    assert remote[remote.index("-hwaccel")+1] == "cuda"
    assert "hevc_cuvid" in remote
    assert "pipe:1" in remote
