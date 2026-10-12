"""Polygon zones, occupancy, and debounced entry/exit events.

Design notes
- Zones are defined in normalized coordinates (0-1) and scaled to the frame size.
- A track's position is the bottom-center of its box (where it touches the floor).
- Debouncing: a track must be on the other side of a zone boundary for N
  consecutive frames before an event fires. This stops a box jittering on an
  edge from spamming entry/exit events (duplicate-event prevention).
- If a track disappears for too long while inside a zone, an exit is emitted
  with reason "track_lost". Remaining tracks get "video_end" on finalize().
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

Point = tuple[float, float]


def point_in_polygon(pt: Point, poly: list[Point]) -> bool:
    """Ray casting. Points exactly on an edge may go either way."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


@dataclass
class Zone:
    name: str
    polygon: list[Point]  # normalized 0-1
    classes: Optional[list[str]] = None  # None = all classes

    def applies_to(self, cls_name: str) -> bool:
        return self.classes is None or cls_name in self.classes

    def pixels(self, width: int, height: int) -> list[Point]:
        return [(x * width, y * height) for x, y in self.polygon]


@dataclass
class Event:
    type: str  # "entry" | "exit"
    zone: str
    track_id: int
    cls: str
    frame: int
    time_s: float
    reason: Optional[str] = None      # exit only: "left_zone" | "track_lost" | "video_end"
    dwell_s: Optional[float] = None   # exit only

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time_s"] = round(d["time_s"], 3)
        if d["dwell_s"] is not None:
            d["dwell_s"] = round(d["dwell_s"], 3)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class _State:
    inside: bool = False
    streak: int = 0
    last_seen: int = 0
    cls: str = ""
    entered_at: float = 0.0


def load_zone_config(path: str | Path) -> tuple[list[Zone], dict]:
    data = json.loads(Path(path).read_text())
    zones = []
    for z in data["zones"]:
        poly = [tuple(p) for p in z["polygon"]]
        if len(poly) < 3:
            raise ValueError(f"Zone '{z['name']}' needs at least 3 points")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in poly):
            raise ValueError(f"Zone '{z['name']}' has points outside 0-1 (use normalized coords)")
        zones.append(Zone(z["name"], poly, z.get("classes")))
    names = [z.name for z in zones]
    if len(names) != len(set(names)):
        raise ValueError("Zone names must be unique")
    return zones, data.get("settings", {})


class ZoneEngine:
    def __init__(
        self,
        zones: list[Zone],
        frame_size: tuple[int, int],
        enter_frames: int = 3,
        exit_frames: int = 5,
        max_missing_frames: int = 30,
    ):
        self.zones = zones
        self.width, self.height = frame_size
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.max_missing = max_missing_frames
        self._poly_px = {z.name: z.pixels(self.width, self.height) for z in zones}
        self._state: dict[tuple[str, int], _State] = {}
        self.events: list[Event] = []
        self.peak_occupancy: dict[str, int] = {z.name: 0 for z in zones}

    @staticmethod
    def anchor(bbox: Iterable[float]) -> Point:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    def update(self, frame: int, time_s: float, tracks: list[dict]) -> list[Event]:
        """tracks: [{"track_id": int, "cls": str, "bbox": (x1,y1,x2,y2)}, ...]"""
        new_events: list[Event] = []
        seen: set[tuple[str, int]] = set()

        for t in tracks:
            pt = self.anchor(t["bbox"])
            for zone in self.zones:
                if not zone.applies_to(t["cls"]):
                    continue
                key = (zone.name, t["track_id"])
                seen.add(key)
                st = self._state.setdefault(key, _State(last_seen=frame, cls=t["cls"]))
                st.last_seen = frame
                st.cls = t["cls"]
                raw_in = point_in_polygon(pt, self._poly_px[zone.name])

                if raw_in == st.inside:
                    st.streak = 0
                    continue
                st.streak += 1
                needed = self.enter_frames if raw_in else self.exit_frames
                if st.streak >= needed:
                    st.inside = raw_in
                    st.streak = 0
                    if raw_in:
                        st.entered_at = time_s
                        ev = Event("entry", zone.name, t["track_id"], t["cls"], frame, time_s)
                    else:
                        ev = Event("exit", zone.name, t["track_id"], t["cls"], frame, time_s,
                                   reason="left_zone", dwell_s=time_s - st.entered_at)
                    new_events.append(ev)

        # Tracks that vanished
        for key in [k for k in self._state if k not in seen]:
            st = self._state[key]
            if frame - st.last_seen > self.max_missing:
                if st.inside:
                    new_events.append(Event("exit", key[0], key[1], st.cls, frame, time_s,
                                            reason="track_lost", dwell_s=time_s - st.entered_at))
                del self._state[key]

        self.events.extend(new_events)
        for name, count in self.occupancy_total().items():
            self.peak_occupancy[name] = max(self.peak_occupancy[name], count)
        return new_events

    def finalize(self, frame: int, time_s: float) -> list[Event]:
        out = []
        for (zone, tid), st in list(self._state.items()):
            if st.inside:
                out.append(Event("exit", zone, tid, st.cls, frame, time_s,
                                 reason="video_end", dwell_s=time_s - st.entered_at))
        self._state.clear()
        self.events.extend(out)
        return out

    def occupancy(self) -> dict[str, dict[str, int]]:
        occ: dict[str, dict[str, int]] = {z.name: defaultdict(int) for z in self.zones}
        for (zone, _), st in self._state.items():
            if st.inside:
                occ[zone][st.cls] += 1
        return {z: dict(c) for z, c in occ.items()}

    def occupancy_total(self) -> dict[str, int]:
        return {z: sum(c.values()) for z, c in self.occupancy().items()}
