# Warehouse Vision System

End-to-end video analytics: **video → YOLO detection → ByteTrack → zone/event engine → annotated video + JSON report → FastAPI**, plus a PyTorch vs ONNX Runtime benchmark.

https://github.com/user-attachments/assets/dba05ae0-3f5f-4f17-ae00-1bb789ee5927

## Status (read this first)

| Part | Verified |
| --- | --- |
| Zone engine (polygons, debounce, occupancy, dwell) | Unit tested |
| Pipeline + API logic | Tested with a fake detector (no real inference) |
| Real YOLO + ByteTrack on real footage | **You need to run this** |
| ONNX export + benchmark | **You need to run this** |
| Warehouse-specific classes (forklift, pallet) | **Not done** — COCO weights don't have them |

## Setup

```bash
python -3.11 -m venv .venv          # Windows: py -3.11 -m venv .venv
source .venv/bin/activate           # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                # optional
pytest                              # should pass without weights or GPU
```

## 1. Run on a video (do this first)

```bash
python -m scripts.process_video data/warehouse.mp4 \
  --output outputs/annotated.mp4 --report outputs/report.json --device cpu
```

Weights (`yolo11n.pt`) download on first run. Useful flags: `--classes person truck`, `--conf 0.4`, `--max-frames 300`, `--no-video`.

**Edit `config/zones.json` to fit your camera view before trusting any event output.** Polygons are normalized (0–1), so open one frame, eyeball the coordinates, and adjust. Watch the annotated video to confirm the zones sit where you think.

Output `report.json` contains video info, config, latency (mean/p50/p95/p99), unique tracks per class, peak occupancy, and every entry/exit event.

## 2. API

```bash
uvicorn app.main:app --reload       # docs at http://127.0.0.1:8000/docs
```

| Endpoint | What |
| --- | --- |
| `POST /jobs` | Upload video, returns job id (processing runs in background) |
| `GET /jobs`, `GET /jobs/{id}` | Status + progress |
| `GET /jobs/{id}/events?zone=&type=` | Entry/exit events |
| `GET /jobs/{id}/metrics` | Perf, tracking, zone summary |
| `GET /jobs/{id}/video` | Annotated video |

Jobs are stored on disk (`data/jobs/<id>/`), so they survive restarts. Jobs interrupted by a restart are marked `failed`. One video is processed at a time.




## 3. ONNX

```bash
python -m scripts.export_onnx --model yolo11n.pt
python -m scripts.benchmark_onnx data/test_images --onnx-model yolo11n.onnx
```

Reports mean/p50/p95 latency, FPS, memory, and a box-matching agreement rate between PyTorch and ONNX outputs (a correctness sanity check, **not** mAP). For mAP, run `yolo val` on both against a labeled set and compare.

## 4. Docker

```bash
docker build -t warehouse-vision .
docker run -p 8000:8000 -v "$PWD/data:/srv/data" --env-file .env warehouse-vision
```

## How events work

- Position = bottom-center of the box.
- A track must be inside/outside for `enter_frames` / `exit_frames` consecutive frames before an event fires. This kills boundary-jitter spam.
- Exit reasons: `left_zone`, `track_lost` (missing > `max_missing_frames`), `video_end`.
- Known weakness: if ByteTrack swaps or loses an ID, you get an exit + a new entry. That's a tracking error showing up in event metrics, so measure it.

## What's left to make this resume-ready

1. **Warehouse classes.** Fine-tune on a forklift/pallet/person dataset (e.g. from Roboflow Universe), then pass the weights with `--weights` and set `--classes`.
2. **Real evaluation.** Hand-label events in a few clips, compute event precision/recall, plus detection mAP and a tracking metric (HOTA/IDF1 via TrackEval). Save the numbers; don't quote assumed ones.
3. **Hardening.** Swap the disk job store for a DB/queue (Celery/RQ) for long videos, add a real-weights integration test in CI, then RTSP input.

## Layout

```
app/        config, pipeline, zone_engine, jobs, main (FastAPI)
scripts/    process_video, export_onnx, benchmark_onnx
config/     zones.json
tests/      zone engine, pipeline, API (fake model)
```
