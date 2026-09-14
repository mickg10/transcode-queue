from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import C50_PRESET
from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("MEDIA_ALIASES", raising=False)
    media = tmp_path / "media"; media.mkdir()
    (media / "one.mp4").touch()
    app = create_app(tmp_path / "state", media, run_worker=False)
    with TestClient(app) as c:
        yield c


def test_browse_enqueue_and_cancel(client):
    assert client.get("/api/browse").json()["entries"][0]["name"] == "one.mp4"
    body = {"sources": ["one.mp4"], "preset_id": "c50-proxy"}
    a = client.post("/api/jobs", json=body).json()["jobs"][0]
    b = client.post("/api/jobs", json=body).json()["jobs"][0]
    assert a["id"] == b["id"] and b["duplicate"]
    assert client.post(f'/api/jobs/{a["id"]}/cancel').status_code == 200
    assert client.get(f'/api/jobs/{a["id"]}').json()["status"] == "cancelled"


def test_upload_streams_and_does_not_overwrite(client):
    response = client.post("/api/upload?filename=new.mp4", content=b"fixture bytes")
    assert response.status_code == 200
    assert response.json()["bytes"] == len(b"fixture bytes")
    assert client.post("/api/upload?filename=new.mp4", content=b"replacement").status_code == 409
    assert client.post("/api/upload?filename=../bad.mp4", content=b"bad").status_code == 400


def test_control_and_preset_validation(client):
    assert client.get("/api/status").json()["paused"]
    assert client.post("/api/control", json={"paused": False}).json()["paused"] is False
    valid = {**C50_PRESET, "id": "second", "name": "Second"}
    assert client.put("/api/presets/second", json=valid).status_code == 200
    assert client.put("/api/presets/second", json={**valid, "output_directory": "../"}).status_code == 422
    assert client.delete("/api/presets/second").status_code == 200


def test_cross_origin_writes_and_outside_browse_are_rejected(client):
    assert client.post("/api/control", json={"paused": False},
                       headers={"Origin": "https://unrelated.example"}).status_code == 403
    assert client.get("/api/browse?path=../").status_code == 400
    assert client.get("/health").json() == {"ok": True}
