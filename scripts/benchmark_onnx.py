"""PyTorch vs ONNX Runtime: latency, throughput, memory, and output agreement.

python -m scripts.benchmark_onnx data/test_images --onnx-model yolo11n.onnx
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run(model, images, imgsz, conf, device, warmup, repeats):
    for _ in range(warmup):
        model.predict(images[0], imgsz=imgsz, conf=conf, device=device, verbose=False)
    rss0 = psutil.Process().memory_info().rss
    lat, outs = [], []
    for rep in range(repeats):
        for img in images:
            t = time.perf_counter()
            r = model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            lat.append((time.perf_counter() - t) * 1000)
            if rep == 0:
                outs.append([(b.xyxy[0].tolist(), int(b.cls), float(b.conf)) for b in r.boxes])
    rss = psutil.Process().memory_info().rss
    return {"mean_ms": round(float(np.mean(lat)), 2), "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "throughput_fps": round(1000 / float(np.mean(lat)), 2),
            "rss_mb_after": round(rss / 1e6, 1), "rss_mb_delta": round((rss - rss0) / 1e6, 1)}, outs


def agreement(a_outs, b_outs, thr=0.5):
    """Greedy box matching between PyTorch (a) and ONNX (b) outputs."""
    matched = total_a = total_b = 0
    for a, b in zip(a_outs, b_outs):
        total_a += len(a); total_b += len(b)
        used = set()
        for box, c, _ in a:
            best, bj = 0.0, None
            for j, (box2, c2, _) in enumerate(b):
                if j in used or c2 != c:
                    continue
                v = iou(box, box2)
                if v > best:
                    best, bj = v, j
            if bj is not None and best >= thr:
                used.add(bj); matched += 1
    return {"pytorch_boxes": total_a, "onnx_boxes": total_b, "matched_iou>=0.5": matched,
            "match_rate_vs_pytorch": round(matched / total_a, 4) if total_a else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", help="directory of test images")
    p.add_argument("--pt-model", default="yolo11n.pt")
    p.add_argument("--onnx-model", default="yolo11n.onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="outputs/benchmark.json")
    a = p.parse_args()

    paths = sorted(q for q in Path(a.images).iterdir() if q.suffix.lower() in IMG_EXT)
    if not paths:
        raise SystemExit(f"No images found in {a.images}")
    images = [cv2.imread(str(q)) for q in paths]

    # ONNX first so its memory isn't inflated by PyTorch already being loaded
    onnx_stats, onnx_out = run(YOLO(a.onnx_model, task="detect"), images, a.imgsz, a.conf, a.device, a.warmup, a.repeats)
    pt_stats, pt_out = run(YOLO(a.pt_model), images, a.imgsz, a.conf, a.device, a.warmup, a.repeats)

    result = {"images": len(images), "imgsz": a.imgsz, "device": a.device,
              "pytorch": pt_stats, "onnxruntime": onnx_stats,
              "speedup_mean": round(pt_stats["mean_ms"] / onnx_stats["mean_ms"], 2),
              "agreement": agreement(pt_out, onnx_out)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("\nNote: for mAP on a labeled set, also run `yolo val` on both models and compare.")


if __name__ == "__main__":
    main()
