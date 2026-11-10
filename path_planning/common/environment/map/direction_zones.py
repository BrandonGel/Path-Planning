"""Direction zones: axis-aligned rectangles that restrict how roadmap edges may be driven.

A zone is given in WORLD coordinates (the frame of the map YAML's obstacles and
of the roadmap nodes). It constrains every roadmap edge whose closed segment
touches the rectangle -- even only at its boundary -- so nothing slips through
the border the wrong way. Two kinds:

- one-way (``direction_deg``): the directed edge ``p -> q`` is allowed only when
  its heading is within ``tolerance_deg`` of one of the listed headings. The
  reverse direction of a kept edge is removed (unless it is listed too).
- angle region (``angles_deg``): the edge's LINE angle (mod 180 degrees) must be
  within ``tolerance_deg`` of one of the listed angles; both directions are
  kept. ``angles_deg: [0, 90]`` keeps only horizontal and vertical edges.

An edge touching several zones must satisfy all of them. Zero-length segments
(waiting, turning in place) are always allowed.

YAML / dict form (``map.direction_zones``)::

    direction_zones:
      - {x: [22, 26], y: [1, 15], direction_deg: 90, tolerance_deg: 45, name: "north lane"}
      - {x: [28, 34], y: [4, 12], angles_deg: [0, 90]}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from path_planning.common.environment.map.speed_zones import SpeedZone

DEFAULT_ONE_WAY_TOLERANCE_DEG = 45.0
DEFAULT_ANGLE_TOLERANCE_DEG = 1.0
_EPS_DEG = 1e-7  # slack on the tolerance test (floating point headings of grid edges)


def _wrap180(deg: float) -> float:
    """Signed angle difference folded into (-180, 180]."""
    d = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


@dataclass(frozen=True)
class DirectionZone:
    x0: float
    x1: float
    y0: float
    y1: float
    headings_deg: tuple          # allowed headings (one-way) or line angles (angle region)
    tolerance_deg: float
    bidirectional: bool = False  # True: angle region (line angles mod 180)
    name: str | None = None

    def __post_init__(self):
        x0, x1 = sorted((float(self.x0), float(self.x1)))
        y0, y1 = sorted((float(self.y0), float(self.y1)))
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "y1", y1)
        hs = self.headings_deg
        if isinstance(hs, (int, float)):
            hs = (hs,)
        mod = 180.0 if self.bidirectional else 360.0
        hs = tuple(float(h) % mod for h in hs)
        object.__setattr__(self, "headings_deg", hs)
        object.__setattr__(self, "tolerance_deg", float(self.tolerance_deg))
        object.__setattr__(self, "bidirectional", bool(self.bidirectional))
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"direction zone needs a positive extent (got x {x0}..{x1}, y {y0}..{y1})")
        if not hs:
            raise ValueError("direction zone needs at least one heading / angle")
        if not (math.isfinite(self.tolerance_deg) and 0.0 <= self.tolerance_deg <= 180.0):
            raise ValueError(f"direction zone tolerance must be in [0, 180] degrees (got {self.tolerance_deg})")

    # the rectangle geometry is the speed zones' (closed rectangle, Liang-Barsky)
    @property
    def bounds(self):
        return ((self.x0, self.x1), (self.y0, self.y1))

    def contains(self, point: Sequence[float]) -> bool:
        x, y = float(point[0]), float(point[1])
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def intersects_segment(self, p1: Sequence[float], p2: Sequence[float], eps: float = 1e-12) -> bool:
        return SpeedZone.intersects_segment(self, p1, p2, eps)

    def allows_heading(self, heading_deg: float) -> bool:
        """Heading (degrees) of a directed motion is permitted by this zone's rule
        (regardless of where the motion is)."""
        tol = self.tolerance_deg + _EPS_DEG
        if self.bidirectional:
            a = float(heading_deg) % 180.0
            for h in self.headings_deg:
                d = abs(a - h) % 180.0
                if min(d, 180.0 - d) <= tol:
                    return True
            return False
        return any(abs(_wrap180(float(heading_deg) - h)) <= tol for h in self.headings_deg)

    def allows(self, p1: Sequence[float], p2: Sequence[float]) -> bool:
        """Directed segment ``p1 -> p2`` satisfies this zone (True when it does
        not touch the zone or has zero length)."""
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        if dx == 0.0 and dy == 0.0:
            return True
        if not self.intersects_segment(p1, p2):
            return True
        return self.allows_heading(math.degrees(math.atan2(dy, dx)))

    def to_spec(self) -> dict:
        spec = {"x": [self.x0, self.x1], "y": [self.y0, self.y1]}
        spec["angles_deg" if self.bidirectional else "direction_deg"] = list(self.headings_deg)
        spec["tolerance_deg"] = self.tolerance_deg
        if self.name is not None:
            spec["name"] = self.name
        return spec


def parse_direction_zone(spec) -> DirectionZone:
    """``DirectionZone`` from a dict: rectangle as ``x``/``y``, ``bounds`` or
    ``x_min..y_max`` (like speed zones) plus either ``direction_deg`` (number or
    list; also ``direction`` / ``heading_deg``; one-way) or ``angles_deg`` (list;
    also ``angles``; both ways), and an optional ``tolerance_deg`` (default 45
    for one-way, 1 for angle regions)."""
    if isinstance(spec, DirectionZone):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported direction zone spec {spec!r}")
    d = {str(k).lower(): v for k, v in spec.items()}
    if "bounds" in d:
        (x0, x1), (y0, y1) = d["bounds"]
    elif "x" in d and "y" in d:
        x0, x1 = d["x"]
        y0, y1 = d["y"]
    else:
        x0, x1, y0, y1 = d["x_min"], d["x_max"], d["y_min"], d["y_max"]
    one_way = d.get("direction_deg", d.get("direction", d.get("heading_deg")))
    angles = d.get("angles_deg", d.get("angles"))
    if (one_way is None) == (angles is None):
        raise ValueError(f"direction zone {spec!r} needs exactly one of direction_deg / angles_deg")
    bidirectional = angles is not None
    hs = angles if bidirectional else one_way
    if isinstance(hs, (int, float)):
        hs = [hs]
    tol = d.get("tolerance_deg", d.get("tolerance"))
    if tol is None:
        tol = DEFAULT_ANGLE_TOLERANCE_DEG if bidirectional else DEFAULT_ONE_WAY_TOLERANCE_DEG
    return DirectionZone(x0, x1, y0, y1, tuple(hs), tol, bidirectional=bidirectional, name=d.get("name"))


def parse_direction_zones(specs: Iterable | None) -> list[DirectionZone]:
    return [parse_direction_zone(s) for s in (specs or [])]


def direction_zones_to_specs(zones: Iterable[DirectionZone]) -> list[dict]:
    return [z.to_spec() for z in zones]


def violated_zone(zones: Iterable[DirectionZone], p1: Sequence[float], p2: Sequence[float]):
    """First zone the directed segment ``p1 -> p2`` violates, or ``None``."""
    for z in zones:
        if not z.allows(p1, p2):
            return z
    return None


def segment_allowed(zones: Iterable[DirectionZone], p1: Sequence[float], p2: Sequence[float]) -> bool:
    """Directed segment ``p1 -> p2`` satisfies every zone it touches."""
    return violated_zone(zones, p1, p2) is None
