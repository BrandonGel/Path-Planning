"""Edge constraints by region, applied to a roadmap's adjacency after it is built.

Each zone is a polygon (its boundary points in order, world coordinates) and a list of allowed
angle ranges ``(lo, hi)``:

``direction_zones``: one-way travel. A directed edge ``(s, t)`` that falls in the zone is kept only
when its heading ``atan2(t - s)``, in ``[0, 2*pi)``, lies in one of the ranges. ``(t, s)`` is judged
by its own heading, so the range ``(0, pi/4)`` keeps an edge heading ``pi/4`` and drops its reverse
(heading ``5*pi/4``).

``edge_zones``: allowed line angles. An edge that falls in the zone is kept, in both directions, only
when its line angle, in ``[0, pi)`` and the same for ``s -> t`` and ``t -> s``, lies in one of the
ranges.

In YAML, under ``roadmap_config:``::

    direction_zones:
      - points: [[0, 0], [0, 10], [10, 10], [10, 0]]
        directions: [[0.0, 0.7853981634]]        # radians
    edge_zones:
      - points: [[20, 0], [20, 10], [30, 10], [30, 0]]
        directions: [[0, 10], [80, 100]]         # horizontal or vertical lines only
        degrees: true                            # ranges given in degrees

``rule`` sets what "falls in" means: ``"intersects"`` (default; any part of the edge touches the
zone), ``"within"`` (the whole edge lies in the zone, boundary included) or ``"midpoint"`` (its
midpoint does). An edge in several zones must satisfy each of them. A range with ``lo > hi`` wraps
through 0, e.g. ``(7*pi/4, pi/4)`` for "roughly +x".
"""
import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.sparse import csr_array

RULES = ("intersects", "within", "midpoint")
ANGLE_EPS = 1e-9                 # a heading this close to a range end counts as inside


@dataclass(frozen=True)
class EdgeZone:
    """A parsed zone: ``polygon``, ``ranges`` (``(R, 2)`` radians) over ``period`` (``2*pi`` for a
    direction zone, ``pi`` for an edge zone) and the membership ``rule``."""
    polygon: shapely.Polygon
    ranges: np.ndarray
    period: float
    rule: str

    @classmethod
    def from_dict(cls, data, period: float, kind: str) -> "EdgeZone":
        """Parse ``{"points": [[x, y], ...], "directions": [[lo, hi], ...], "degrees": bool, "rule": str}``."""
        if not isinstance(data, dict):
            raise TypeError(f"{kind}: each zone is a mapping, got {type(data).__name__}")
        unknown = set(data) - {"points", "directions", "degrees", "rule"}
        if unknown:
            raise TypeError(f"{kind}: unknown zone keys {sorted(unknown)}")
        pts = np.asarray(data.get("points"), dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError(f"{kind}: 'points' must be >= 3 finite (x, y) points in order, got shape {pts.shape}")
        poly = shapely.Polygon(pts)
        if not poly.is_valid or poly.area <= 0:
            raise ValueError(f"{kind}: zone polygon is invalid ({shapely.is_valid_reason(poly)})")
        ranges = np.atleast_2d(np.asarray(data.get("directions"), dtype=float))
        if ranges.ndim != 2 or ranges.shape[1] != 2 or len(ranges) == 0 or not np.isfinite(ranges).all():
            raise ValueError(f"{kind}: 'directions' must be a list of (lo, hi) ranges")
        if data.get("degrees", False):
            ranges = np.radians(ranges)
        if ranges.min() < -ANGLE_EPS or ranges.max() > period + ANGLE_EPS:
            top = "2*pi (360 deg)" if period > 4 else "pi (180 deg)"
            raise ValueError(f"{kind}: ranges must lie in [0, {top}], got {ranges.tolist()}")
        rule = data.get("rule", "intersects")
        if rule not in RULES:
            raise ValueError(f"{kind}: rule must be one of {RULES}, got {rule!r}")
        shapely.prepare(poly)
        return cls(poly, ranges, float(period), rule)

    def matches(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """``(E,)`` bool: which segments ``p[i] -> q[i]`` fall in the zone (symmetric in ``p``, ``q``)."""
        x0, y0, x1, y1 = self.polygon.bounds
        lo, hi = np.minimum(p, q), np.maximum(p, q)
        near = (hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= y0) & (lo[:, 1] <= y1)   # bbox prefilter
        out = np.zeros(len(p), dtype=bool)
        idx = np.flatnonzero(near)
        if len(idx) == 0:
            return out
        if self.rule == "midpoint":
            mid = 0.5 * (p[idx] + q[idx])
            out[idx] = shapely.intersects_xy(self.polygon, mid[:, 0], mid[:, 1])
        else:
            lines = shapely.linestrings(np.stack([p[idx], q[idx]], axis=1))
            out[idx] = (shapely.intersects(self.polygon, lines) if self.rule == "intersects"
                        else shapely.covered_by(lines, self.polygon))
        return out

    def allows(self, angle: np.ndarray) -> np.ndarray:
        """``(E,)`` bool: which angles (already reduced to ``[0, period)``) lie in one of the ranges.
        A plain range also tests ``angle + period``, so a range ending at ``period`` admits 0."""
        a = np.asarray(angle, dtype=float)[:, None]
        lo, hi = self.ranges[None, :, 0], self.ranges[None, :, 1]
        inside = lambda b: (b >= lo - ANGLE_EPS) & (b <= hi + ANGLE_EPS)
        plain = inside(a) | inside(a + self.period)
        wrap = (a >= lo - ANGLE_EPS) | (a <= hi + ANGLE_EPS)
        return np.where(lo <= hi, plain, wrap).any(axis=1)


def parse_zones(direction_zones, edge_zones) -> tuple:
    """``(direction_zones, edge_zones)`` as lists of ``EdgeZone``; raises on a malformed zone."""
    d = [EdgeZone.from_dict(z, 2.0 * math.pi, "direction_zones") for z in (direction_zones or [])]
    e = [EdgeZone.from_dict(z, math.pi, "edge_zones") for z in (edge_zones or [])]
    return d, e


def constrain_adjacency(nodes: np.ndarray, adj: csr_array, direction_zones=None, edge_zones=None) -> csr_array:
    """``adj`` with the edges the zones forbid removed (see the module docstring). ``nodes`` is
    ``(N, 2)``; zones are dicts or ``EdgeZone``s. Without zones ``adj`` is returned unchanged. The
    result is directed when a direction zone drops one direction of an edge."""
    dz, ez = parse_zones(
        [z for z in (direction_zones or []) if not isinstance(z, EdgeZone)],
        [z for z in (edge_zones or []) if not isinstance(z, EdgeZone)])
    dz += [z for z in (direction_zones or []) if isinstance(z, EdgeZone)]
    ez += [z for z in (edge_zones or []) if isinstance(z, EdgeZone)]
    if (not dz and not ez) or adj.nnz == 0:
        return adj
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"edge zones need 2D nodes, got shape {nodes.shape}")
    n = adj.shape[0]
    rows = np.repeat(np.arange(n), np.diff(adj.indptr))
    cols = adj.indices
    p, q = nodes[rows], nodes[cols]
    d = q - p
    heading = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * math.pi)
    line = np.mod(heading, math.pi)
    keep = np.ones(len(rows), dtype=bool)
    for zones, angle in ((dz, heading), (ez, line)):
        for z in zones:
            m = z.matches(p, q) & keep
            if m.any():
                keep[m] = z.allows(angle[m])
    if keep.all():
        return adj
    counts = np.bincount(rows[keep], minlength=n)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(adj.indptr.dtype)
    return csr_array((adj.data[keep], cols[keep], indptr), shape=adj.shape)
