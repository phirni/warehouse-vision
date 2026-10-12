from app.zone_engine import Zone, ZoneEngine, point_in_polygon

SQUARE = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]


def make_engine(**kw):
    return ZoneEngine([Zone("z", SQUARE)], (100, 100), **kw)


def track(tid, x, y, cls="person"):
    # bottom-center anchor = (x, y)
    return {"track_id": tid, "cls": cls, "bbox": (x - 5, y - 10, x + 5, y)}


def test_point_in_polygon():
    poly = [(25, 25), (75, 25), (75, 75), (25, 75)]
    assert point_in_polygon((50, 50), poly)
    assert not point_in_polygon((10, 10), poly)
    assert not point_in_polygon((80, 50), poly)


def test_entry_fires_once_after_debounce():
    eng = make_engine(enter_frames=3)
    events = []
    for f in range(10):
        events += eng.update(f, f / 10, [track(1, 50, 50)])
    assert [e.type for e in events] == ["entry"]
    assert events[0].frame == 2  # third consecutive frame inside


def test_boundary_jitter_does_not_spam_events():
    eng = make_engine(enter_frames=3, exit_frames=5)
    events = []
    for f in range(40):
        x = 50 if f % 2 == 0 else 80  # flips in/out every frame
        events += eng.update(f, f / 10, [track(1, x, 50)])
    assert events == []


def test_exit_has_dwell_time():
    eng = make_engine(enter_frames=1, exit_frames=2)
    events = []
    for f in range(5):
        events += eng.update(f, float(f), [track(1, 50, 50)])
    for f in range(5, 10):
        events += eng.update(f, float(f), [track(1, 90, 50)])
    types = [e.type for e in events]
    assert types == ["entry", "exit"]
    assert events[1].reason == "left_zone"
    assert events[1].dwell_s > 0


def test_track_lost_emits_exit():
    eng = make_engine(enter_frames=1, max_missing_frames=3)
    eng.update(0, 0.0, [track(1, 50, 50)])
    events = []
    for f in range(1, 10):
        events += eng.update(f, float(f), [])
    assert len(events) == 1
    assert events[0].type == "exit" and events[0].reason == "track_lost"


def test_class_filter():
    eng = ZoneEngine([Zone("z", SQUARE, classes=["person"])], (100, 100), enter_frames=1)
    events = eng.update(0, 0.0, [track(1, 50, 50, "truck"), track(2, 50, 50, "person")])
    assert [e.track_id for e in events] == [2]


def test_occupancy_and_peak():
    eng = make_engine(enter_frames=1)
    eng.update(0, 0.0, [track(1, 50, 50), track(2, 40, 40, "truck")])
    assert eng.occupancy()["z"] == {"person": 1, "truck": 1}
    eng.update(1, 0.1, [track(1, 50, 50)])
    assert eng.peak_occupancy["z"] == 2


def test_finalize_closes_open_tracks():
    eng = make_engine(enter_frames=1)
    eng.update(0, 0.0, [track(1, 50, 50)])
    ev = eng.finalize(10, 1.0)
    assert len(ev) == 1 and ev[0].reason == "video_end"
