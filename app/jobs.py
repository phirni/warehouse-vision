"""Tiny disk-backed job store: one folder per job, state in job.json (survives restarts)."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._recover()

    def _recover(self):
        """Jobs left 'running'/'queued' by a crash can never finish; mark them failed."""
        for p in self.root.glob("*/job.json"):
            job = json.loads(p.read_text())
            if job["status"] in ("queued", "running"):
                job["status"] = "failed"
                job["error"] = "interrupted by server restart"
                p.write_text(json.dumps(job, indent=2))

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create(self, filename: str) -> dict:
        job_id = uuid.uuid4().hex[:12]
        self.dir(job_id).mkdir(parents=True)
        job = {"id": job_id, "filename": filename, "status": "queued", "progress": 0.0,
               "created_at": datetime.now(timezone.utc).isoformat(), "error": None}
        self.save(job)
        return job

    def save(self, job: dict):
        with self._lock:
            (self.dir(job["id"]) / "job.json").write_text(json.dumps(job, indent=2))

    def get(self, job_id: str) -> Optional[dict]:
        p = self.dir(job_id) / "job.json"
        if not job_id.isalnum() or not p.exists():
            return None
        return json.loads(p.read_text())

    def list(self) -> list[dict]:
        jobs = [json.loads(p.read_text()) for p in self.root.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)
