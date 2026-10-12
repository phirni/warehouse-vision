"""Video -> YOLO detection -> ByteTrack -> zone events -> annotated video + report."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .zone_engine import Zone, ZoneEngine, load_zone_config


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


class VideoPipeline:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        device: str = "cpu",
        conf: float = 0.30,
        iou: float = 0.50,
        imgsz: int = 640,
        classes: Optional[list[str]] = None,
        zones_path: str = "config/zones.json",
        tracker: str = "bytetrack.yaml",
        model=None,
    ):
        if model is None:
            from ultralytics import YOLO
            model = YOLO(weights)
        self.model = model
        self.weights = weights
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = tracker
        self.zones, self.zone_settings = load_zone_config(zones_path)

        names = self.model.names  # {id: name}
        self.class_ids = None
        if classes:
            name_to_id = {v: k for k, v in names.items()}
            missing = [c for c in classes if c not in name_to_id]
            if missing:
                raise ValueError(f"Model has no classes {missing}. Available: {sorted(name_to_id)}")
            self.class_ids = [name_to_id[c] for c in classes]
        self.class_names = classes

    # ------------------------------------------------------------------
    def process(
        self,
        source: str | Path,
        output: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if max_frames:
            total = min(total, max_frames) if total else max_frames

        writer = None
        raw_out = None
        if output:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            raw_out = output.with_suffix(".raw.mp4")
            writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        engine = ZoneEngine(
            self.zones, (width, height),
            enter_frames=self.zone_settings.get("enter_frames", 3),
            exit_frames=self.zone_settings.get("exit_frames", 5),
            max_missing_frames=self.zone_settings.get("max_missing_frames", 30),
        )

        # Fresh tracker state for every video (otherwise IDs leak across runs)
        self.model.predictor = None

        track_ms: list[float] = []
        frame_ms: list[float] = []
        infer_ms: list[float] = []
        unique_ids: dict[str, set[int]] = defaultdict(set)
        dets_per_frame: list[int] = []
        frame_idx = 0
        wall_start = time.perf_counter()

        while True:
            t_frame = time.perf_counter()
            ok, frame = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames):
                break

            t0 = time.perf_counter()
            res = self.model.track(
                frame, persist=True, tracker=self.tracker, conf=self.conf, iou=self.iou,
                imgsz=self.imgsz, device=self.device, classes=self.class_ids, verbose=False,
            )[0]
            track_ms.append((time.perf_counter() - t0) * 1000)
            if getattr(res, "speed", None):
                infer_ms.append(res.speed.get("inference", 0.0))

            tracks = self._extract_tracks(res)
            dets_per_frame.append(len(tracks))
            for t in tracks:
                unique_ids[t["cls"]].add(t["track_id"])

            time_s = frame_idx / fps
            engine.update(frame_idx, time_s, tracks)

            if writer is not None:
                writer.write(self._annotate(frame, tracks, engine))
            frame_ms.append((time.perf_counter() - t_frame) * 1000)
            frame_idx += 1
            if progress and frame_idx % 10 == 0:
                progress(frame_idx, total)

        engine.finalize(frame_idx, frame_idx / fps)
        cap.release()
        wall = time.perf_counter() - wall_start
        if writer is not None:
            writer.release()
            self._finalize_video(raw_out, output)

        return {
            "video": {"path": str(source), "fps": fps, "width": width, "height": height,
                      "frames_processed": frame_idx, "duration_s": round(frame_idx / fps, 2)},
            "config": {"weights": self.weights, "device": self.device, "conf": self.conf,
                       "iou": self.iou, "imgsz": self.imgsz, "tracker": self.tracker,
                       "classes": self.class_names, "zones": [z.name for z in self.zones]},
            "performance": {
                "wall_time_s": round(wall, 2),
                "end_to_end_fps": round(frame_idx / wall, 2) if wall else 0.0,
                "detect_track_ms": {"mean": round(float(np.mean(track_ms)), 2) if track_ms else 0.0,
                                    "p50": round(_pct(track_ms, 50), 2),
                                    "p95": round(_pct(track_ms, 95), 2),
                                    "p99": round(_pct(track_ms, 99), 2)},
                "model_inference_ms_mean": round(float(np.mean(infer_ms)), 2) if infer_ms else None,
                "frame_total_ms": {"mean": round(float(np.mean(frame_ms)), 2) if frame_ms else 0.0,
                                   "p95": round(_pct(frame_ms, 95), 2)},
            },
            "tracking": {
                "unique_tracks_by_class": {c: len(ids) for c, ids in unique_ids.items()},
                "mean_detections_per_frame": round(float(np.mean(dets_per_frame)), 2) if dets_per_frame else 0.0,
            },
            "zones": {"peak_occupancy": engine.peak_occupancy,
                      "event_counts": self._event_counts(engine)},
            "events": [e.to_dict() for e in engine.events],
        }

    # ------------------------------------------------------------------
    def _extract_tracks(self, res) -> list[dict]:
        boxes = res.boxes
        if boxes is None or boxes.id is None:  # no confirmed tracks this frame
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        cls = boxes.cls.int().cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        names = self.model.names
        return [{"track_id": i, "cls": names[c], "bbox": tuple(b), "conf": s}
                for b, i, c, s in zip(xyxy, ids, cls, conf)]

    @staticmethod
    def _event_counts(engine: ZoneEngine) -> dict:
        out: dict[str, dict[str, int]] = {z.name: {"entry": 0, "exit": 0} for z in engine.zones}
        for e in engine.events:
            out[e.zone][e.type] += 1
        return out

    def _annotate(self, frame, tracks, engine: ZoneEngine):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        occ = engine.occupancy_total()
        for z in self.zones:
            pts = np.array(z.pixels(w, h), dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 180, 255))
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        for z in self.zones:
            pts = np.array(z.pixels(w, h), dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 180, 255), 2)
            cv2.putText(frame, f"{z.name}: {occ[z.name]}", (int(pts[0][0]) + 6, int(pts[0][1]) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
        for t in tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            col = _color(t["track_id"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"{t['cls']} #{t['track_id']} {t['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
            ax, ay = map(int, ZoneEngine.anchor(t["bbox"]))
            cv2.circle(frame, (ax, ay), 3, (0, 0, 255), -1)
        return frame

    @staticmethod
    def _finalize_video(raw: Path, final: Path):
        """Re-encode to H.264 if ffmpeg exists (browser-playable); else keep mp4v."""
        if shutil.which("ffmpeg"):
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(final)], capture_output=True)
            if r.returncode == 0:
                raw.unlink(missing_ok=True)
                return
        raw.replace(final)


def save_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2))
