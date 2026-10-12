"""FastAPI service: upload a video, process in the background, fetch events/metrics/video."""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from .config import Settings
from .jobs import JobStore
from .pipeline import VideoPipeline, save_report

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}
settings = Settings()
store: JobStore | None = None
_pipeline: VideoPipeline | None = None
_infer_lock = threading.Lock()  # one video at a time; the model/tracker isn't thread-safe


def get_pipeline() -> VideoPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VideoPipeline(settings.weights, settings.device, settings.conf, settings.iou,
                                  settings.imgsz, settings.classes, settings.zones_path)
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    store = JobStore(settings.data_dir)
    yield


app = FastAPI(title="Warehouse Vision System", version="0.1.0", lifespan=lifespan)


def _run_job(job_id: str):
    job = store.get(job_id)
    d = store.dir(job_id)
    job["status"] = "running"
    store.save(job)
    try:
        def progress(i, n):
            job["progress"] = round(i / n, 3) if n else 0.0
            store.save(job)

        with _infer_lock:
            report = get_pipeline().process(d / "input.mp4", d / "annotated.mp4", progress=progress)
        save_report(report, d / "report.json")
        job.update(status="done", progress=1.0)
    except Exception as e:  # noqa: BLE001
        job.update(status="failed", error=str(e))
    store.save(job)


def _job_or_404(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


def _report_or_409(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return json.loads((store.dir(job_id) / "report.json").read_text())


@app.get("/health")
def health():
    return {"status": "ok", "weights": settings.weights, "device": settings.device}


@app.post("/jobs", status_code=202)
async def create_job(background: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")
    job = store.create(file.filename)
    dest = store.dir(job["id"]) / "input.mp4"
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                dest.unlink(missing_ok=True)
                job.update(status="failed", error="file too large")
                store.save(job)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
            f.write(chunk)
    background.add_task(_run_job, job["id"])
    return job


@app.get("/jobs")
def list_jobs():
    return store.list()


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id)


@app.get("/jobs/{job_id}/events")
def get_events(job_id: str, zone: str | None = None, type: str | None = Query(None, pattern="^(entry|exit)$")):
    events = _report_or_409(job_id)["events"]
    if zone:
        events = [e for e in events if e["zone"] == zone]
    if type:
        events = [e for e in events if e["type"] == type]
    return {"count": len(events), "events": events}


@app.get("/jobs/{job_id}/metrics")
def get_metrics(job_id: str):
    r = _report_or_409(job_id)
    return {k: r[k] for k in ("video", "config", "performance", "tracking", "zones")}


@app.get("/jobs/{job_id}/video")
def get_video(job_id: str):
    _report_or_409(job_id)
    return FileResponse(store.dir(job_id) / "annotated.mp4", media_type="video/mp4")
