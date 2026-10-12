"""CLI: python -m scripts.process_video data/warehouse.mp4 --output outputs/annotated.mp4 --report outputs/report.json"""
import argparse

from app.config import Settings
from app.pipeline import VideoPipeline, save_report


def main():
    s = Settings()
    p = argparse.ArgumentParser(description="Run detection + tracking + zone events on a video.")
    p.add_argument("source")
    p.add_argument("--output", default="outputs/annotated.mp4")
    p.add_argument("--report", default="outputs/report.json")
    p.add_argument("--weights", default=s.weights)
    p.add_argument("--device", default=s.device, help="cpu, cuda:0, mps")
    p.add_argument("--conf", type=float, default=s.conf)
    p.add_argument("--iou", type=float, default=s.iou)
    p.add_argument("--imgsz", type=int, default=s.imgsz)
    p.add_argument("--classes", nargs="*", default=s.classes, help="class names to keep")
    p.add_argument("--zones", default=s.zones_path)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-video", action="store_true", help="skip writing annotated video")
    a = p.parse_args()

    pipe = VideoPipeline(a.weights, a.device, a.conf, a.iou, a.imgsz, a.classes, a.zones)
    report = pipe.process(
        a.source, None if a.no_video else a.output, a.max_frames,
        progress=lambda i, n: print(f"\rframe {i}/{n or '?'}", end="", flush=True),
    )
    print()
    save_report(report, a.report)

    perf = report["performance"]
    print(f"frames: {report['video']['frames_processed']}  "
          f"fps: {perf['end_to_end_fps']}  "
          f"detect+track p50/p95: {perf['detect_track_ms']['p50']}/{perf['detect_track_ms']['p95']} ms")
    print(f"tracks: {report['tracking']['unique_tracks_by_class']}")
    print(f"zone events: {report['zones']['event_counts']}")
    print(f"report -> {a.report}" + ("" if a.no_video else f"   video -> {a.output}"))


if __name__ == "__main__":
    main()
