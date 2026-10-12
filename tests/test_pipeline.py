import cv2
import numpy as np

from app.pipeline import VideoPipeline
from tests.fakes import FakeModel


def make_video(path, frames=40, w=320, h=240, fps=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(frames):
        vw.write(np.zeros((h, w, 3), np.uint8))
    vw.release()


def test_pipeline_end_to_end_with_fake_model(tmp_path):
    src = tmp_path / "in.mp4"
    make_video(src)
    pipe = VideoPipeline(model=FakeModel(), zones_path="config/zones.json", classes=["person"])
    report = pipe.process(src, tmp_path / "out.mp4")

    assert report["video"]["frames_processed"] == 40
    assert report["tracking"]["unique_tracks_by_class"] == {"person": 1}
    assert (tmp_path / "out.mp4").exists()
    # person starts left of the walkway zone (x>=0.55w) and walks into it
    walkway = [e for e in report["events"] if e["zone"] == "pedestrian_walkway"]
    assert [e["type"] for e in walkway][0] == "entry"
    assert report["performance"]["detect_track_ms"]["p95"] >= 0


def test_unknown_class_rejected():
    import pytest
    with pytest.raises(ValueError):
        VideoPipeline(model=FakeModel(), classes=["forklift"])
