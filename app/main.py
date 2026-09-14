from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .core import MediaPaths, Store, is_video
from .worker import Worker


class Preset(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    name: str = Field(min_length=1, max_length=100)
    container: Literal["mp4", "mov"] = "mp4"
    codec: Literal["hevc", "h264"] = "hevc"
    backend: Literal["auto", "nvidia", "nvidia-encode", "software"] = "auto"
    max_width: int = Field(default=3456, ge=128, le=8192)
    max_height: int = Field(default=2304, ge=128, le=8192)
    bit_depth: Literal[8, 10] = 10
    quality: int = Field(default=21, ge=0, le=51)
    bitrate_mbps: int = Field(default=20, ge=1, le=500)
    maxrate_mbps: int = Field(default=30, ge=1, le=800)
    gop: int = Field(default=12, ge=1, le=300)
    nvenc_preset: Literal["p1", "p2", "p3", "p4", "p5", "p6", "p7"] = "p4"
    audio: Literal["copy", "aac"] = "copy"
    output_directory: str = Field(default="proxy", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    reuse_legacy_proxies: bool = True

    @model_validator(mode="after")
    def coherent(self):
        if self.maxrate_mbps < self.bitrate_mbps:
            raise ValueError("Maximum bitrate must be at least the target bitrate")
        if self.codec == "h264" and self.bit_depth != 8:
            raise ValueError("Use 8-bit for the H.264 preset")
        return self


class Enqueue(BaseModel):
    sources: list[str] = Field(min_length=1, max_length=20000)
    preset_id: str = "c50-proxy"
    output: str | None = None


class TreeQueue(BaseModel):
    path: str = ""
    recursive: bool = True
    preset_id: str = "c50-proxy"


class QueueControl(BaseModel):
    paused: bool


def create_app(data_dir: Path | None = None, media_root: Path | None = None, run_worker=True):
    paths = MediaPaths(media_root or Path(os.getenv("MEDIA_ROOT", "/media/photos")),
                       json.loads(os.getenv("MEDIA_ALIASES", "{}")))
    store = Store(data_dir or Path(os.getenv("DATA_DIR", "/data")),
                  os.getenv("START_PAUSED", "true").lower() == "true")
    manifest = os.getenv("WAIT_FOR_MANIFEST")
    worker = Worker(store, paths, os.getenv("FFMPEG", "ffmpeg"), os.getenv("FFPROBE", "ffprobe"),
                    Path(manifest) if manifest else None)

    @asynccontextmanager
    async def lifespan(app):
        if run_worker:
            worker.start()
        yield
        if run_worker:
            worker.stop()

    app = FastAPI(title="Transcode Queue", lifespan=lifespan)
    app.state.store, app.state.paths, app.state.worker = store, paths, worker

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin writes are not permitted"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_request(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request, exc):
        return JSONResponse({"detail": "File or directory not found"}, status_code=404)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/browse")
    def browse(path: str = ""):
        return {"path": path, "entries": paths.children(path)}

    @app.post("/api/scan")
    def scan(body: TreeQueue):
        entries = paths.scan(body.path, body.recursive)
        return {"files": entries, "count": len(entries), "bytes": sum(x["bytes"] for x in entries)}

    @app.get("/api/presets")
    def presets():
        return store.presets()

    @app.put("/api/presets/{identifier}")
    def save_preset(identifier: str, preset: Preset):
        if identifier != preset.id:
            raise ValueError("Preset ID differs from the URL")
        with store.connect() as db:
            db.execute("INSERT OR REPLACE INTO presets VALUES(?,?)",
                       (identifier, preset.model_dump_json()))
        return preset

    @app.delete("/api/presets/{identifier}")
    def delete_preset(identifier: str):
        with store.connect() as db:
            if db.execute("SELECT COUNT(*) FROM presets").fetchone()[0] <= 1:
                raise ValueError("Keep at least one preset")
            db.execute("DELETE FROM presets WHERE id=?", (identifier,))
        return {"ok": True}

    @app.post("/api/jobs")
    def enqueue(body: Enqueue):
        return {"jobs": store.enqueue(paths, body.sources, body.preset_id, body.output)}

    @app.post("/api/queue-tree")
    def queue_tree(body: TreeQueue):
        sources = [x["path"] for x in paths.scan(body.path, body.recursive)]
        return {"jobs": store.enqueue(paths, sources, body.preset_id), "count": len(sources)}

    @app.get("/api/jobs")
    def jobs(limit: int = 500):
        return store.jobs(min(max(limit, 1), 20000))

    @app.get("/api/jobs/{identifier}")
    def job(identifier: int):
        result = store.get(identifier)
        if not result:
            raise HTTPException(404, "Job not found")
        return result

    @app.post("/api/jobs/{identifier}/cancel")
    def cancel(identifier: int):
        with store.connect() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (identifier,)).fetchone()
            if not row:
                raise HTTPException(404, "Job not found")
            if row[0] == "queued":
                db.execute("UPDATE jobs SET status='cancelled',finished=?,message='Cancelled' WHERE id=?",
                           (time.time(), identifier))
            elif row[0] == "running":
                db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (identifier,))
        return {"ok": True}

    @app.post("/api/jobs/{identifier}/retry")
    def retry(identifier: int):
        old = store.get(identifier)
        if not old:
            raise HTTPException(404, "Job not found")
        return {"jobs": store.enqueue(paths, [old["source"]], old["preset_id"], old["output"])}

    @app.get("/api/jobs/{identifier}/log")
    def job_log(identifier: int):
        if not store.get(identifier):
            raise HTTPException(404, "Job not found")
        pieces = []
        for p in sorted((store.directory / "logs").glob(f"{identifier}-*.log")):
            with p.open("rb") as f:
                f.seek(max(0, p.stat().st_size - 12000))
                pieces.append(p.name + "\n" + f.read().decode(errors="replace"))
        return {"text": "\n".join(pieces)}

    @app.get("/api/status")
    def status():
        with store.connect() as db:
            counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())
        return {"paused": store.setting("paused"), "counts": counts, "waiting": worker.waiting_reason,
                "free_bytes": shutil.disk_usage(paths.root).free}

    @app.post("/api/control")
    def control(body: QueueControl):
        store.set_setting("paused", body.paused)
        return {"paused": body.paused}

    @app.get("/api/hardware")
    def hardware():
        try:
            r = subprocess.run(["nvidia-smi",
                                "--query-gpu=name,driver_version,utilization.gpu,utilization.encoder,utilization.decoder,memory.used",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            return {"nvidia": r.stdout.strip(), "available": r.returncode == 0}
        except (OSError, subprocess.TimeoutExpired):
            return {"nvidia": "", "available": False}

    @app.post("/api/upload")
    async def upload(request: Request, directory: str = "", filename: str = ""):
        if not filename or PurePosixPath(filename).name != filename or "\\" in filename:
            raise ValueError("Choose a plain filename")
        if not is_video(Path(filename)):
            raise ValueError("Upload a supported video file")
        folder = paths.resolve(directory, must_exist=True)
        if not folder.is_dir():
            raise ValueError("Upload destination must be a directory")
        relative = str(PurePosixPath(directory) / filename)
        target = paths.resolve(relative)
        if target.exists():
            raise HTTPException(409, "A file with that name already exists")
        length = int(request.headers.get("content-length", "0"))
        if length and length + 1024**3 > shutil.disk_usage(folder).free:
            raise HTTPException(507, "Not enough free space")
        temp = folder / ("." + filename + ".upload-" + uuid.uuid4().hex)
        received = 0
        try:
            with temp.open("xb") as stream:
                async for chunk in request.stream():
                    received += len(chunk)
                    await asyncio.to_thread(stream.write, chunk)
                await asyncio.to_thread(stream.flush)
                os.fsync(stream.fileno())
            if length and received != length:
                raise ValueError("Upload was incomplete")
            target = paths.resolve(relative)
            os.link(temp, target)  # Atomic publication without clobbering another upload.
            return {"path": relative, "bytes": received}
        except FileExistsError:
            raise HTTPException(409, "Another upload created that filename")
        finally:
            temp.unlink(missing_ok=True)

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
