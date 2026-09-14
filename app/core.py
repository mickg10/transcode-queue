from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path, PurePosixPath

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".mxf", ".mts", ".m2ts", ".avi", ".webm", ".ts"}
EXCLUDED_DIRS = {"proxy", "proxies", "reference", "render", "renders", "@recycle", "@eadir",
                 "$recycle.bin", "system volume information"}
C50_PRESET = {
    "id": "c50-proxy", "name": "C50 proxy", "container": "mp4",
    "codec": "hevc", "backend": "auto", "max_width": 3456, "max_height": 2304,
    "bit_depth": 10, "quality": 21, "bitrate_mbps": 20, "maxrate_mbps": 30,
    "gop": 12, "nvenc_preset": "p4", "audio": "copy",
    "output_directory": "proxy", "reuse_legacy_proxies": True,
}


class MediaPaths:
    """Expose only the selected photo tree and explicitly mounted aliases."""
    def __init__(self, root: Path, aliases: dict[str, str] | None = None):
        self.root = root.resolve()
        self.aliases = {name: Path(path).resolve() for name, path in (aliases or {}).items()}
        if any("/" in name or name in {"", ".", ".."} for name in self.aliases):
            raise ValueError("Media aliases must be single directory names")

    def resolve(self, relative: str = "", *, must_exist: bool = False) -> Path:
        if "\\" in relative or "\0" in relative:
            raise ValueError("Use relative paths with forward slashes")
        p = PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Path must stay inside the media directory")
        parts = p.parts
        base = self.root
        if parts and parts[0] in self.aliases:
            base, parts = self.aliases[parts[0]], parts[1:]
        result = base.joinpath(*parts).resolve()
        if not result.is_relative_to(base):
            raise ValueError("Path leaves an exposed media directory")
        if must_exist and not result.exists():
            raise FileNotFoundError(relative)
        return result

    def children(self, relative: str = "") -> list[dict]:
        folder = self.resolve(relative, must_exist=True)
        if not folder.is_dir():
            raise ValueError("Choose a directory")
        entries = {}
        for p in folder.iterdir():
            if p.name.startswith(".") or p.name.startswith("@"):
                continue
            child = str(PurePosixPath(relative) / p.name)
            try:
                actual = self.resolve(child, must_exist=True)
                stat = actual.stat()
                if actual.is_dir() or is_video(actual):
                    entries[p.name] = {"name": p.name, "path": child, "directory": actual.is_dir(),
                                       "bytes": stat.st_size if actual.is_file() else None,
                                       "modified": stat.st_mtime}
            except (OSError, ValueError):
                continue
        if not relative:
            for name, actual in self.aliases.items():
                if actual.is_dir():
                    entries[name] = {"name": name, "path": name, "directory": True,
                                     "bytes": None, "modified": actual.stat().st_mtime}
        return sorted(entries.values(), key=lambda x: (not x["directory"], x["name"].casefold()))

    def scan(self, relative: str, recursive: bool = True, errors: list | None = None) -> list[dict]:
        result, visited = [], set()
        def walk(folder):
            actual = self.resolve(folder, must_exist=True)
            if actual in visited:
                return
            visited.add(actual)
            for item in self.children(folder):
                if item["directory"]:
                    if recursive and item["name"].lower() not in EXCLUDED_DIRS:
                        try:
                            walk(item["path"])
                        except (OSError, ValueError) as exc:
                            if errors is not None:
                                errors.append({"path": item["path"], "error": str(exc)})
                elif is_video(Path(item["name"])):
                    item["recorded"] = recording_time(item["name"], item["modified"])
                    result.append(item)
        walk(relative)
        return sorted(result, key=lambda x: (-x["recorded"], x["path"]))


def is_video(path: Path) -> bool:
    return (path.suffix.lower() in VIDEO_EXTENSIONS and
            not path.name.startswith(".") and
            not re.search(r"(?:_proxy|\.partial|\.upload)$", path.stem, re.I))


def recording_time(name: str, modified: float) -> float:
    match = re.search(r"C\d+[A-Z](\d{6})_(\d{6})", name, re.I)
    if match:
        import datetime
        try:
            return datetime.datetime.strptime("".join(match.groups()), "%y%m%d%H%M%S").replace(
                tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            pass
    return modified


def video(meta: dict) -> dict:
    return next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})


def duration(meta: dict) -> float:
    value = video(meta).get("duration") or meta.get("format", {}).get("duration")
    number = float(value or 0)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Media has no valid video duration")
    return number


def frame_tolerance(meta: dict) -> float:
    try:
        rate = float(Fraction(video(meta).get("avg_frame_rate", "0/1")))
        return 1 / rate if rate > 0 else 0.05
    except (ValueError, ZeroDivisionError):
        return 0.05


def same_duration(source: dict, output: dict) -> bool:
    try:
        return bool(video(output)) and abs(duration(source) - duration(output)) <= frame_tolerance(source) + 1e-6
    except (TypeError, ValueError):
        return False


def probe(path: Path, executable: str = "ffprobe") -> dict:
    r = subprocess.run([executable, "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", str(path)], capture_output=True, text=True, timeout=90)
    if r.returncode:
        raise ValueError("Cannot probe media: " + r.stderr[-1000:])
    meta = json.loads(r.stdout)
    if not video(meta):
        raise ValueError("No video stream found")
    duration(meta)
    return meta


def dimensions(meta: dict, preset: dict) -> tuple[int, int]:
    v = video(meta)
    w, h = int(v["width"]), int(v["height"])
    try:
        w *= float(Fraction(v.get("sample_aspect_ratio", "1:1").replace(":", "/")))
    except (ValueError, ZeroDivisionError):
        pass
    scale = min(1.0, preset["max_width"] / w, preset["max_height"] / h)
    return max(2, int(w * scale / 2) * 2), max(2, int(h * scale / 2) * 2)


def timecode(meta: dict) -> str | None:
    return meta.get("format", {}).get("tags", {}).get("timecode") or next(
        (s.get("tags", {}).get("timecode") for s in meta.get("streams", [])
         if s.get("tags", {}).get("timecode")), None)


def default_output(source: str, preset: dict) -> str:
    p = PurePosixPath(source)
    return str(p.parent / preset["output_directory"] / (p.stem + "." + preset["container"]))


def legacy_outputs(source: str) -> list[str]:
    p = PurePosixPath(source)
    return [str(p.parent / folder / (p.stem + suffix + extension))
            for folder, suffix in (("Proxies", "_Proxy"), ("proxies", "_Proxy"), ("proxy", ""))
            for extension in (".mov", ".mp4")]


class Store:
    def __init__(self, directory: Path, start_paused: bool = True):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.path = directory / "queue.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS presets(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, output TEXT NOT NULL,
                    preset_id TEXT NOT NULL, preset TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    created REAL NOT NULL, recorded REAL NOT NULL, started REAL, finished REAL,
                    progress REAL NOT NULL DEFAULT 0, fps REAL, speed TEXT, message TEXT NOT NULL DEFAULT '',
                    actual_output TEXT, source_bytes INTEGER, output_bytes INTEGER, duration REAL,
                    backend TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_output ON jobs(output)
                    WHERE status IN ('queued','running');
                CREATE INDEX IF NOT EXISTS queue_order ON jobs(status, recorded DESC, id);
            """)
            db.execute("INSERT OR IGNORE INTO presets VALUES(?,?)",
                       (C50_PRESET["id"], json.dumps(C50_PRESET)))
            db.execute("INSERT OR IGNORE INTO settings VALUES('paused',?)",
                       (json.dumps(start_paused),))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def setting(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, json.dumps(value)))

    def presets(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT body FROM presets ORDER BY id")]

    def preset(self, identifier):
        result = next((p for p in self.presets() if p["id"] == identifier), None)
        if not result:
            raise ValueError("Preset does not exist")
        return result

    def enqueue(self, paths: MediaPaths, sources: list[str], preset_id: str,
                output: str | None = None):
        preset = self.preset(preset_id)
        if output and len(sources) != 1:
            raise ValueError("A custom destination requires exactly one source")
        prepared = []
        for source in dict.fromkeys(sources):
            path = paths.resolve(source, must_exist=True)
            if not path.is_file() or not is_video(path):
                raise ValueError("Not a supported video: " + source)
            destination = output or default_output(source, preset)
            target = paths.resolve(destination)
            if path == target:
                raise ValueError("The output cannot replace its source")
            if target.suffix.lower() != "." + preset["container"]:
                raise ValueError("Destination extension must match the preset container")
            st = path.stat()
            prepared.append((source, destination, recording_time(path.name, st.st_mtime)))
        result = []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for source, destination, recorded in sorted(prepared, key=lambda x: (-x[2], x[0])):
                existing = db.execute("SELECT id FROM jobs WHERE output=? AND status IN ('queued','running')",
                                      (destination,)).fetchone()
                if existing:
                    result.append({"id": existing[0], "duplicate": True})
                else:
                    cursor = db.execute("""INSERT INTO jobs(source,output,preset_id,preset,created,recorded)
                                           VALUES(?,?,?,?,?,?)""",
                                        (source, destination, preset_id, json.dumps(preset), time.time(), recorded))
                    result.append({"id": cursor.lastrowid, "duplicate": False})
        return result

    def jobs(self, limit=500):
        with self.connect() as db:
            rows = db.execute("""SELECT * FROM jobs ORDER BY
                CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
                CASE WHEN status IN ('queued','running') THEN recorded ELSE finished END DESC, id
                LIMIT ?""", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        return dict(row) if row else None

    def update(self, identifier, **fields):
        allowed = {"status", "started", "finished", "progress", "fps", "speed", "message",
                   "actual_output", "source_bytes", "output_bytes", "duration", "backend", "cancel_requested"}
        if not fields.keys() <= allowed:
            raise ValueError("Invalid job update")
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in fields) + " WHERE id=?",
                       (*fields.values(), identifier))

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY recorded DESC,id LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET status='running',started=?,message='' WHERE id=?", (time.time(), row["id"]))
        return dict(row) if row else None
