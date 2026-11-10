"""Speed zones: axis-aligned rectangles of the map with a speed limit.

A zone is given in WORLD coordinates (the frame of the map YAML's obstacles and
of the roadmap nodes in both discrete and continuous mode). A roadmap edge that
touches a zone -- even only at its boundary, which is the conservative side --
is traversed at ``min(agent max speed, lowest limit of every zone it touches)``
for its whole length, so every edge stays one linear segment for the sweeps,
the schedules and the checker. Zones never limit turning.

YAML / dict form (``map.speed_zones``)::

    speed_zones:
      - {x: [28.0, 33.0], y: [1.0, 15.0], speed_limit: 0.5, name: "slow band"}
      - {bounds: [[0, 5], [0, 5]], speed_limit: 1.0}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SpeedZone:
    x0: float
    x1: float
    y0: float
    y1: float
    speed_limit: float
    name: str | None = None

    def __post_init__(self):
        x0, x1 = sorted((float(self.x0), float(self.x1)))
        y0, y1 = sorted((float(self.y0), float(self.y1)))
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "speed_limit", float(self.speed_limit))
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"speed zone needs a positive extent (got x {x0}..{x1}, y {y0}..{y1})")
        if not (math.isfinite(self.speed_limit) and self.speed_limit > 0.0):
            raise ValueError(f"speed zone limit must be finite and > 0 (got {self.speed_limit})")

    @property
    def bounds(self):
        return ((self.x0, self.x1), (self.y0, self.y1))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def contains(self, point: Sequence[float]) -> bool:
        x, y = float(point[0]), float(point[1])
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def intersects_segment(self, p1: Sequence[float], p2: Sequence[float], eps: float = 1e-12) -> bool:
        """Closed segment ``p1 -> p2`` meets the closed rectangle (Liang-Barsky)."""
        xa, ya = float(p1[0]), float(p1[1])
        dx = float(p2[0]) - xa
        dy = float(p2[1]) - ya
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, xa - self.x0), (dx, self.x1 - xa), (-dy, ya - self.y0), (dy, self.y1 - ya)):
            if abs(p) <= eps:
                if q < -eps:
                    return False
                continue
            r = q / p
            if p < 0.0:
                if r > t0:
                    t0 = r
            else:
                if r < t1:
                    t1 = r
        return t0 <= t1 + eps

    def to_spec(self) -> dict:
        spec = {"x": [self.x0, self.x1], "y": [self.y0, self.y1], "speed_limit": self.speed_limit}
        if self.name is not None:
            spec["name"] = self.name
        return spec


def parse_speed_zone(spec) -> SpeedZone:
    """``SpeedZone`` from a dict ``{x: [x0, x1], y: [y0, y1], speed_limit: v}`` (also
    ``bounds: [[x0, x1], [y0, y1]]``, ``x_min/x_max/y_min/y_max``; the limit may be
    called ``speed_limit``, ``limit`` or ``max_speed``)."""
    if isinstance(spec, SpeedZone):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported speed zone spec {spec!r}")
    d = {str(k).lower(): v for k, v in spec.items()}
    if "bounds" in d:
        (x0, x1), (y0, y1) = d["bounds"]
    elif "x" in d and "y" in d:
        x0, x1 = d["x"]
        y0, y1 = d["y"]
    else:
        x0, x1, y0, y1 = d["x_min"], d["x_max"], d["y_min"], d["y_max"]
    limit = d.get("speed_limit", d.get("limit", d.get("max_speed")))
    if limit is None:
        raise ValueError(f"speed zone {spec!r} has no speed_limit")
    return SpeedZone(x0, x1, y0, y1, limit, name=d.get("name"))


def parse_speed_zones(specs: Iterable | None) -> list[SpeedZone]:
    return [parse_speed_zone(s) for s in (specs or [])]


def zones_to_specs(zones: Iterable[SpeedZone]) -> list[dict]:
    return [z.to_spec() for z in zones]


def segment_speed_limit(zones: Iterable[SpeedZone], p1: Sequence[float], p2: Sequence[float]) -> float | None:
    """Lowest limit of the zones the closed segment ``p1 -> p2`` touches (``None``
    when it touches none)."""
    limit = None
    for z in zones:
        if z.intersects_segment(p1, p2) and (limit is None or z.speed_limit < limit):
            limit = z.speed_limit
    return limit


def point_speed_limit(zones: Iterable[SpeedZone], p: Sequence[float]) -> float | None:
    return segment_speed_limit(zones, p, p)
