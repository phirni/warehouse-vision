import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.pipeline import VideoPipeline
from tests.fakes import FakeModel
from tests.test_pipeline import make_video


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "data_dir", tmp_path / "jobs")
    pipe = VideoPipeline(model=FakeModel(), classes=["person"])
    monkeypatch.setattr(main, "_pipeline", pipe)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_rejects_bad_extension(client):
    r = client.post("/jobs", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 415


def test_unknown_job_404(client):
    assert client.get("/jobs/doesnotexist").status_code == 404


def test_full_job_flow(client, tmp_path):
    vid = tmp_path / "v.mp4"
    make_video(vid)
    r = client.post("/jobs", files={"file": ("v.mp4", vid.read_bytes(), "video/mp4")})
    assert r.status_code == 202
    jid = r.json()["id"]

    for _ in range(50):
        job = client.get(f"/jobs/{jid}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "done", job

    assert client.get(f"/jobs/{jid}/events").json()["count"] >= 1
    assert client.get(f"/jobs/{jid}/events?type=bogus").status_code == 422
    assert "performance" in client.get(f"/jobs/{jid}/metrics").json()
    assert client.get(f"/jobs/{jid}/video").status_code == 200
