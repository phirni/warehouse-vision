"""Settings from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _classes() -> list[str] | None:
    raw = os.getenv("CLASSES", "").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] or None


@dataclass
class Settings:
    weights: str = field(default_factory=lambda: os.getenv("WEIGHTS", "yolo11n.pt"))
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    conf: float = field(default_factory=lambda: float(os.getenv("CONF", "0.30")))
    iou: float = field(default_factory=lambda: float(os.getenv("IOU", "0.50")))
    imgsz: int = field(default_factory=lambda: int(os.getenv("IMGSZ", "640")))
    classes: list[str] | None = field(default_factory=_classes)
    zones_path: str = field(default_factory=lambda: os.getenv("ZONES_PATH", "config/zones.json"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data/jobs")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "500")))
