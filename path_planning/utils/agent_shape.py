"""Agent footprint shapes for the spatial-sweep backends.

The sweep backends (``CGAL_Sweep`` / ``ShapelySweep``) historically modelled the
agent as a disc: the ``r`` argument of every query is the COMBINED clearance
``r_i + r_j`` of the query agent and the agent that may occupy the roadmap
element. This module generalises that clearance to convex polygons so that
agents with a rectangular or regular-polygon footprint can be swept exactly.

Anywhere a sweep (or ``GraphSampler.get_constraint_sweep`` /
``get_constraint_segment``) accepts ``r`` it now also accepts

- a :class:`Disc` -- unwrapped to its radius, i.e. the existing disc code path;
- a :class:`ConvexPolygon` -- built with :func:`regular_polygon`,
  :func:`rectangle` or :meth:`ConvexPolygon.from_points`.

The polygon passed to a sweep is the *combined* footprint of the two agents,
exactly like ``r_i + r_j`` is for discs: use :func:`pair_clearance` to build it
(Minkowski sum ``P_i (+) (-P_j)``). A bare :class:`ConvexPolygon` is expressed
in the agent's local frame (the agent's reference point is the origin) with a
FIXED world orientation. An :class:`AlignedShape` wraps a body-frame polygon
(``+x`` = forward) that is rotated to the agent's heading -- the direction of
travel along the current edge (see :func:`path_headings` for the heading
carried while an agent waits); the sweep backends take an
:class:`AlignedClearance` (query footprint, element footprint) for such pairs
and resolve the orientations per roadmap element themselves. All shapes are
immutable and hashable so they can be used directly in the sweep cache keys.

Geometry conventions: vertices are stored counter-clockwise starting at the
lexicographically smallest vertex (so geometrically equal polygons compare
equal), and the polygon is also carried as outward half-planes
``normals[k] . x < offsets[k]`` (strict interior).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Sequence

import numpy as np

_EPS = 1e-12


# ----------------------------------------------------------------------------- hull


def convex_hull(points: Iterable[Sequence[float]]) -> np.ndarray:
    """Counter-clockwise convex hull (Andrew's monotone chain), shape ``(k, 2)``.

    Collinear boundary points are dropped. The first hull vertex is the
    lexicographically smallest input point, which makes the output canonical.
    """
    pts = np.asarray(list(points), dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return pts
    pts = np.unique(pts, axis=0)  # lexicographic sort (x, then y) + dedup
    if pts.shape[0] <= 2:
        return pts
    span = float(np.max(np.ptp(pts, axis=0)))
    eps = 1e-12 * max(span * span, 1.0)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= eps:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= eps:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float).reshape(-1, 2)


# ----------------------------------------------------------------------------- shapes


@dataclass(frozen=True)
class Disc:
    """Disc footprint of the given radius (the classic sweep clearance)."""

    radius: float

    def __post_init__(self):
        object.__setattr__(self, "radius", float(self.radius))
        if not np.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError(f"Disc radius must be finite and >= 0 (got {self.radius}).")

    @property
    def bounding_radius(self) -> float:
        return self.radius

    def as_polygon(self, n_sides: int = 16, angle: float = 0.0, circumscribed: bool = True) -> "ConvexPolygon":
        """Regular ``n_sides``-gon approximation of the disc.

        ``circumscribed=True`` (default) returns the polygon that CONTAINS the
        disc (conservative, circumradius ``r / cos(pi / n)``); ``False`` returns
        the polygon inscribed in the disc (vertices on the circle).
        """
        if self.radius <= 0.0:
            raise ValueError("Cannot polygonise a zero-radius disc.")
        if circumscribed:
            return regular_polygon(n_sides, inradius=self.radius, angle=angle)
        return regular_polygon(n_sides, circumradius=self.radius, angle=angle)

    def reflected(self) -> "Disc":
        return self

    def rotated(self, angle: float) -> "Disc":
        _ = angle
        return self


@dataclass(frozen=True)
class ConvexPolygon:
    """Convex polygon footprint in the agent's local frame.

    ``vertices`` are normalised on construction (convex hull, CCW, canonical
    start vertex), so any point sequence describing a convex polygon is
    accepted; degenerate input (fewer than three non-collinear points) raises.
    """

    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self):
        hull = convex_hull(self.vertices)
        if hull.shape[0] < 3:
            raise ValueError("ConvexPolygon needs at least three non-collinear vertices.")
        object.__setattr__(self, "vertices", tuple((float(x), float(y)) for x, y in hull))

    # -------------------------------------------------------------- constructors

    @classmethod
    def from_points(cls, points: Iterable[Sequence[float]]) -> "ConvexPolygon":
        """Convex hull of an arbitrary 2D point set."""
        return cls(tuple((float(p[0]), float(p[1])) for p in points))

    # -------------------------------------------------------------- derived data

    @cached_property
    def points(self) -> np.ndarray:
        """``(m, 2)`` CCW vertex array."""
        arr = np.asarray(self.vertices, dtype=float).reshape(-1, 2)
        arr.setflags(write=False)
        return arr

    @cached_property
    def normals(self) -> np.ndarray:
        """``(m, 2)`` outward unit normals, ``normals[k]`` for edge ``k -> k+1``."""
        p = self.points
        e = np.roll(p, -1, axis=0) - p
        n = np.stack([e[:, 1], -e[:, 0]], axis=1)
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        n.setflags(write=False)
        return n

    @cached_property
    def offsets(self) -> np.ndarray:
        """``(m,)`` half-plane offsets: interior is ``normals . x < offsets``."""
        d = np.einsum("ij,ij->i", self.normals, self.points)
        d.setflags(write=False)
        return d

    @cached_property
    def bounding_radius(self) -> float:
        """Radius of the smallest origin-centred disc containing the polygon."""
        return float(np.max(np.linalg.norm(self.points, axis=1)))

    @cached_property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)``."""
        lo = self.points.min(axis=0)
        hi = self.points.max(axis=0)
        return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    @cached_property
    def centroid(self) -> tuple[float, float]:
        p = self.points
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        area = 0.5 * float(np.sum(cross))
        if abs(area) < _EPS:
            c = p.mean(axis=0)
        else:
            c = np.sum((p + q) * cross[:, None], axis=0) / (6.0 * area)
        return (float(c[0]), float(c[1]))

    @cached_property
    def area(self) -> float:
        p = self.points
        q = np.roll(p, -1, axis=0)
        return 0.5 * float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))

    # -------------------------------------------------------------- transforms

    def translated(self, offset: Sequence[float]) -> "ConvexPolygon":
        off = np.asarray(offset, dtype=float).reshape(2)
        return ConvexPolygon.from_points(self.points + off)

    def rotated(self, angle: float, about: Sequence[float] = (0.0, 0.0)) -> "ConvexPolygon":
        """Rotate counter-clockwise by ``angle`` radians about ``about``."""
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pivot = np.asarray(about, dtype=float).reshape(2)
        return ConvexPolygon.from_points((self.points - pivot) @ rot.T + pivot)

    def reflected(self) -> "ConvexPolygon":
        """Point reflection through the origin (``-P``)."""
        return ConvexPolygon.from_points(-self.points)

    def scaled(self, factor: float) -> "ConvexPolygon":
        return ConvexPolygon.from_points(self.points * float(factor))

    def minkowski(self, other: "ConvexPolygon") -> "ConvexPolygon":
        """Minkowski sum with another convex polygon."""
        pts = (self.points[:, None, :] + other.points[None, :, :]).reshape(-1, 2)
        return ConvexPolygon.from_points(pts)

    def buffered(self, radius: float, n_sides: int = 16) -> "ConvexPolygon":
        """Conservative dilation by ``radius`` (Minkowski sum with the
        circumscribed ``n_sides``-gon of the disc; see :meth:`Disc.as_polygon`)."""
        if radius <= 0.0:
            return self
        return self.minkowski(Disc(radius).as_polygon(n_sides=n_sides, circumscribed=True))

    def swept(self, u: Sequence[float], v: Sequence[float]) -> "ConvexPolygon":
        """Region covered by the polygon translating from ``u`` to ``v``
        (``conv((P + u) U (P + v))``)."""
        u_arr = np.asarray(u, dtype=float).reshape(2)
        v_arr = np.asarray(v, dtype=float).reshape(2)
        return ConvexPolygon.from_points(np.vstack([self.points + u_arr, self.points + v_arr]))

    # -------------------------------------------------------------- queries

    def support(self, directions: np.ndarray) -> np.ndarray:
        """Support function ``h(d) = max_j d . p_j`` for ``directions`` of shape
        ``(k, 2)`` (returns ``(k,)``) or ``(2,)`` (returns a scalar array)."""
        d = np.asarray(directions, dtype=float)
        return np.max(d @ self.points.T, axis=-1)

    def contains(self, points: np.ndarray, strict: bool = True) -> np.ndarray:
        """Boolean mask of ``points`` ``(N, 2)`` inside the polygon."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        g = pts @ self.normals.T - self.offsets[None, :]
        return np.all(g < 0.0, axis=1) if strict else np.all(g <= 0.0, axis=1)


# ----------------------------------------------------------------------------- headings

TWO_PI = 2.0 * math.pi
HEADING_DECIMALS = 12


def heading_key(angle: float) -> float:
    """Canonical, hashable heading in ``[0, 2*pi)`` rounded to 1e-12 rad.

    The sweep backends, the SIPP tables and the planners must agree on the
    heading that labels a roadmap element, so every heading goes through this
    one function (``-0.0`` / ``-pi`` / ``pi`` all collapse onto one key).
    """
    a = float(angle) % TWO_PI
    a = round(a, HEADING_DECIMALS)
    if a >= round(TWO_PI, HEADING_DECIMALS) or a == 0.0:
        return 0.0
    return a


def turn_angle(h_from: float, h_to: float) -> float:
    """Shortest signed rotation (radians, in ``(-pi, pi]``) from heading ``h_from``
    to ``h_to``."""
    d = (float(h_to) - float(h_from) + math.pi) % TWO_PI - math.pi
    if d <= -math.pi + 1e-15:
        d = math.pi
    return d


def turn_time(h_from: float, h_to: float, turning_speed: float | None) -> float:
    """Time to rotate from ``h_from`` to ``h_to`` at ``turning_speed`` rad/s
    (``None`` / non-finite / non-positive = instantaneous)."""
    if turning_speed is None or not math.isfinite(turning_speed) or turning_speed <= 0.0:
        return 0.0
    return abs(turn_angle(h_from, h_to)) / float(turning_speed)


def heading_of(p, q, default: float | None = None) -> float | None:
    """Heading (``heading_key``) of the direction ``p -> q``; ``default`` when
    the two points coincide."""
    dx = float(q[0]) - float(p[0])
    dy = float(q[1]) - float(p[1])
    if dx == 0.0 and dy == 0.0:
        return default
    return heading_key(math.atan2(dy, dx))


def path_headings(points, headings=None, default: float = 0.0) -> list[float]:
    """Per-segment headings of a piecewise-linear path (one entry per point:
    the heading held from that point until the next one; the last entry is
    the heading held after the path ends).

    A moving segment is aligned with its direction of travel. A stationary
    segment (wait / in-place turn) keeps the heading of the previous moving
    segment; before the first move the agent is assumed to already face its
    first edge; a path that never moves faces ``default``. ``headings`` may
    give explicit per-point headings (``None`` entries are filled by this
    rule), e.g. the ``heading`` keys the SIPP planner writes into its
    schedules so that in-place turns are reproduced exactly.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    out: list[float | None] = [None] * n
    if headings is not None:
        for i, h in enumerate(headings):
            if i < n and h is not None:
                out[i] = heading_key(h)
    for i in range(n - 1):
        if out[i] is None:
            out[i] = heading_of(pts[i], pts[i + 1])
    # carry backwards over stationary stretches, then forwards into the start.
    last = None
    for i in range(n):
        if out[i] is None:
            out[i] = last
        else:
            last = out[i]
    nxt = None
    for i in range(n - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return [heading_key(default) if h is None else h for h in out]


def schedule_headings(schedule, default: float = 0.0) -> list[float]:
    """:func:`path_headings` for a ``[{"x", "y", "t", ("heading")}, ...]`` schedule."""
    return path_headings(
        [(p["x"], p["y"]) for p in schedule],
        headings=[p.get("heading") for p in schedule],
        default=default,
    )


@dataclass(frozen=True)
class AlignedShape:
    """Convex footprint given in the BODY frame (``+x`` = forward) that is
    rotated to the agent's heading: along a roadmap edge the agent faces its
    direction of travel; while waiting it keeps the heading it arrived with
    (see :func:`path_headings`). Turns are instantaneous (the footprint snaps
    between headings; the rotation itself is not swept).

    ``at(heading)`` is the world-frame :class:`ConvexPolygon` for one heading;
    ``envelope`` is the disc covering the footprint under every heading
    (bounding radius), used wherever the heading is unknown.
    """

    body: ConvexPolygon

    def __post_init__(self):
        if isinstance(self.body, AlignedShape):
            object.__setattr__(self, "body", self.body.body)
        if not isinstance(self.body, ConvexPolygon):
            object.__setattr__(self, "body", ConvexPolygon.from_points(self.body))

    @property
    def bounding_radius(self) -> float:
        return self.body.bounding_radius

    @property
    def envelope(self) -> Disc:
        return Disc(self.body.bounding_radius)

    @property
    def vertices(self):
        return self.body.vertices

    @property
    def points(self) -> np.ndarray:
        return self.body.points

    def at(self, heading: float) -> ConvexPolygon:
        """World-frame polygon when the agent faces ``heading`` (radians)."""
        return _rotated_body(self.body, heading_key(heading))

    def rotated(self, angle: float) -> "AlignedShape":
        """Rotate the BODY (changes how the shape sits relative to forward)."""
        return AlignedShape(self.body.rotated(angle))

    def reflected(self) -> "AlignedShape":
        return AlignedShape(self.body.reflected())


@functools.lru_cache(maxsize=4096)
def _rotated_body(body: ConvexPolygon, key: float) -> ConvexPolygon:
    if key == 0.0:
        return body
    # Exact quarter turns keep grid-aligned footprints exactly axis-aligned.
    quarter = key / (0.5 * math.pi)
    q = int(round(quarter))
    if abs(quarter - q) < 1e-9:
        pts = body.points
        for _ in range(q % 4):
            pts = np.stack([-pts[:, 1], pts[:, 0]], axis=1)
        return ConvexPolygon.from_points(pts)
    return body.rotated(key)


AgentShape = Disc | ConvexPolygon | AlignedShape


def resolve_shape(shape: AgentShape, heading: float | None) -> Disc | ConvexPolygon:
    """World-frame footprint of ``shape`` for a given heading: discs and fixed
    polygons are returned unchanged; an :class:`AlignedShape` is rotated to
    ``heading``, or replaced by its envelope disc when the heading is ``None``."""
    if isinstance(shape, AlignedShape):
        return shape.at(heading) if heading is not None else shape.envelope
    return shape


def is_aligned(shape) -> bool:
    return isinstance(shape, AlignedShape)


# ----------------------------------------------------------------------------- factories


def regular_polygon(
    n_sides: int,
    circumradius: float | None = None,
    *,
    inradius: float | None = None,
    angle: float = 0.0,
) -> ConvexPolygon:
    """Regular ``n_sides``-gon centred at the origin.

    Size is given either by ``circumradius`` (vertex distance) or ``inradius``
    (apothem, edge distance). ``angle`` is the direction of the first vertex in
    radians, e.g. ``regular_polygon(4, 1.0, angle=pi/4)`` is an axis-aligned
    square of side ``sqrt(2)``.
    """
    n = int(n_sides)
    if n < 3:
        raise ValueError(f"A regular polygon needs at least 3 sides (got {n}).")
    if (circumradius is None) == (inradius is None):
        raise ValueError("Give exactly one of circumradius / inradius.")
    if circumradius is None:
        circumradius = float(inradius) / math.cos(math.pi / n)
    if circumradius <= 0.0:
        raise ValueError("Regular polygon radius must be > 0.")
    theta = float(angle) + 2.0 * math.pi * np.arange(n) / n
    pts = float(circumradius) * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return ConvexPolygon.from_points(pts)


def rectangle(
    width: float,
    height: float,
    angle: float = 0.0,
    center: Sequence[float] = (0.0, 0.0),
) -> ConvexPolygon:
    """Rectangle of extent ``width`` along the local x-axis and ``height`` along
    the local y-axis, rotated counter-clockwise by ``angle`` radians and placed
    at ``center`` (in the agent frame; the default puts the agent's reference
    point at the rectangle's centre)."""
    w, h = float(width), float(height)
    if w <= 0.0 or h <= 0.0:
        raise ValueError("Rectangle width and height must be > 0.")
    hw, hh = 0.5 * w, 0.5 * h
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    poly = ConvexPolygon.from_points(pts)
    if angle != 0.0:
        poly = poly.rotated(angle)
    c = np.asarray(center, dtype=float).reshape(2)
    if np.any(c != 0.0):
        poly = poly.translated(c)
    return poly


def square(side: float, angle: float = 0.0, center: Sequence[float] = (0.0, 0.0)) -> ConvexPolygon:
    return rectangle(side, side, angle=angle, center=center)


# ----------------------------------------------------------------------------- combination


def minkowski_sum(a: AgentShape, b: AgentShape, n_sides: int = 16) -> AgentShape:
    """Minkowski sum of two footprints.

    ``Disc (+) Disc`` is exact (``Disc(r_a + r_b)``), ``polygon (+) polygon`` is
    exact, and ``Disc (+) polygon`` is a conservative polygon: the disc is
    replaced by its circumscribed ``n_sides``-gon (over-approximation factor
    ``1 / cos(pi / n_sides)``, i.e. ~2% for 16 sides, ~0.5% for 32).
    """
    if isinstance(a, Disc) and isinstance(b, Disc):
        return Disc(a.radius + b.radius)
    if isinstance(a, Disc):
        a, b = b, a
    if not isinstance(a, ConvexPolygon):
        raise TypeError(f"Unsupported shape {type(a).__name__}")
    if isinstance(b, Disc):
        return a.buffered(b.radius, n_sides=n_sides)
    if not isinstance(b, ConvexPolygon):
        raise TypeError(f"Unsupported shape {type(b).__name__}")
    return a.minkowski(b)


def minkowski_batch(a: ConvexPolygon, b: ConvexPolygon, angles) -> np.ndarray:
    """Vertices of ``a (+) rot(angles[k]) b`` for every ``k``: ``(K, m + n, 2)``
    CCW vertex arrays (consecutive vertices may be collinear; every edge has
    positive length). Edge-sequence merge, vectorised over the rotation angles,
    so a whole set of headings costs a couple of sorts instead of a hull each.
    """
    A = a.points
    B = b.points
    ang = np.asarray(angles, dtype=float).reshape(-1)
    K = ang.shape[0]
    c, sn = np.cos(ang), np.sin(ang)
    Bk = np.stack([c[:, None] * B[None, :, 0] - sn[:, None] * B[None, :, 1],
                   sn[:, None] * B[None, :, 0] + c[:, None] * B[None, :, 1]], axis=-1)  # (K, n, 2)
    eA = np.roll(A, -1, axis=0) - A  # (m, 2)
    eB = np.roll(B, -1, axis=0) - B  # (n, 2)
    angA = np.arctan2(eA[:, 1], eA[:, 0])
    angB = np.arctan2(eB[:, 1], eB[:, 0])[None, :] + ang[:, None]  # (K, n)
    eBk = np.stack([c[:, None] * eB[None, :, 0] - sn[:, None] * eB[None, :, 1],
                    sn[:, None] * eB[None, :, 0] + c[:, None] * eB[None, :, 1]], axis=-1)
    # Start at the leftmost (then lowest) vertex of the sum = sum of the
    # leftmost-lowest vertices; CCW from there the edge angles increase
    # through (-pi/2, 3pi/2] (an edge pointing straight down can only be the
    # one arriving at that vertex, so it sorts last).
    def _canon(x):
        y = (x + 0.5 * math.pi) % TWO_PI - 0.5 * math.pi
        return np.where(y < -0.5 * math.pi + 1e-12, y + TWO_PI, y)
    ia = int(np.lexsort((A[:, 1], A[:, 0]))[0])
    ib = np.argmin(Bk[:, :, 0] + 1e-9 * Bk[:, :, 1], axis=1)  # (K,)
    start = A[ia][None, :] + Bk[np.arange(K), ib]  # (K, 2)
    edges = np.concatenate([np.broadcast_to(eA[None], (K,) + eA.shape), eBk], axis=1)  # (K, m+n, 2)
    angs = np.concatenate([np.broadcast_to(_canon(angA)[None], (K, eA.shape[0])), _canon(angB)], axis=1)
    order = np.argsort(angs, axis=1, kind="stable")
    sorted_edges = np.take_along_axis(edges, order[:, :, None], axis=1)
    verts = np.cumsum(sorted_edges, axis=1)
    verts = np.concatenate([np.zeros((K, 1, 2)), verts[:, :-1]], axis=1) + start[:, None, :]
    return verts


def halfplanes_of_vertex_rows(verts: np.ndarray):
    """Outward half-planes of CCW vertex rows ``(K, m, 2)`` (e.g. from
    :func:`minkowski_batch`): ``normals (K, m, 2)``, ``offsets (K, m)``;
    interior is ``normals . x < offsets``."""
    e = np.roll(verts, -1, axis=1) - verts
    n = np.stack([e[..., 1], -e[..., 0]], axis=-1)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    d = np.einsum("kmd,kmd->km", n, verts)
    return n, d


def pair_clearance(shape_i: AgentShape, shape_j: AgentShape, n_sides: int = 16):
    """Combined footprint of agents ``i`` and ``j``: ``P_i (+) (-P_j)``.

    Agent ``i`` at ``c_i`` and agent ``j`` at ``c_j`` overlap iff
    ``c_j - c_i`` lies inside the returned shape, which is what the sweep
    backends test. For two discs this is ``Disc(r_i + r_j)``, mirroring the
    scalar ``r_i + r_j`` the planners already pass. When either footprint is
    heading-aligned the result is an :class:`AlignedClearance` (the sweep
    resolves the orientations per element); use :func:`oriented_pair_clearance`
    when both headings are known.
    """
    if isinstance(shape_i, AlignedShape) or isinstance(shape_j, AlignedShape):
        return AlignedClearance(shape_i, shape_j, n_sides=n_sides)
    return minkowski_sum(shape_i, shape_j.reflected(), n_sides=n_sides)


def oriented_pair_clearance(
    shape_i: AgentShape, heading_i: float | None, shape_j: AgentShape, heading_j: float | None, n_sides: int = 16
) -> Disc | ConvexPolygon:
    """:func:`pair_clearance` with both agents resolved to world-frame
    footprints at the given headings (``None`` -> envelope disc of an aligned
    shape; ignored for discs / fixed polygons)."""
    return minkowski_sum(resolve_shape(shape_i, heading_i), resolve_shape(shape_j, heading_j).reflected(), n_sides=n_sides)


@dataclass(frozen=True)
class AlignedClearance:
    """Clearance between a QUERY agent (moves along the swept segment) and an
    ELEMENT agent (occupies the roadmap vertex / edge) when at least one of
    them is an :class:`AlignedShape`. Hashable, so it can be used as a sweep /
    memo key like a float radius or a polygon.

    The sweep resolves the orientations itself: the query faces its direction
    of travel (or ``query_heading`` when the query is stationary; its envelope
    disc when that is ``None`` too); an aligned element faces each directed
    roadmap edge it may traverse and, at a vertex, every heading of the edges
    incident to that vertex (the vertex results are keyed ``(vertex, heading)``).
    """

    query: AgentShape
    element: AgentShape
    query_heading: float | None = None
    n_sides: int = 16

    def __post_init__(self):
        if isinstance(self.query, (int, float)):
            object.__setattr__(self, "query", Disc(float(self.query)))
        if isinstance(self.element, (int, float)):
            object.__setattr__(self, "element", Disc(float(self.element)))
        if self.query_heading is not None:
            object.__setattr__(self, "query_heading", heading_key(self.query_heading))
        object.__setattr__(self, "n_sides", int(self.n_sides))

    @property
    def aligned_query(self) -> bool:
        return isinstance(self.query, AlignedShape)

    @property
    def aligned_element(self) -> bool:
        return isinstance(self.element, AlignedShape)

    @property
    def bounding_radius(self) -> float:
        return float(self.query.bounding_radius + self.element.bounding_radius)

    def with_query_heading(self, heading: float | None) -> "AlignedClearance":
        """Copy carrying the query's heading (for stationary query segments);
        returns ``self`` when the query is not aligned, so cache keys stay put."""
        if not self.aligned_query:
            return self
        key = None if heading is None else heading_key(heading)
        if key == self.query_heading:
            return self
        return AlignedClearance(self.query, self.element, key, self.n_sides)

    def query_footprint(self, motion_heading: float | None) -> Disc | ConvexPolygon:
        """World-frame query footprint: ``motion_heading`` (direction of the
        swept segment) wins, else ``query_heading``, else the envelope disc."""
        h = motion_heading if motion_heading is not None else self.query_heading
        return resolve_shape(self.query, h)

    def element_footprint(self, heading: float | None) -> Disc | ConvexPolygon:
        return resolve_shape(self.element, heading)

    def resolve(self, query_heading: float | None, element_heading: float | None) -> float | ConvexPolygon:
        """Combined world-frame clearance for known headings: the float
        ``r_q + r_e`` for a disc pair, else ``Q (+) (-E)``."""
        q = self.query_footprint(query_heading)
        e = self.element_footprint(element_heading)
        if isinstance(q, Disc) and isinstance(e, Disc):
            return q.radius + e.radius
        return minkowski_sum(q, e.reflected(), n_sides=self.n_sides)


def as_shape(r: float | AgentShape) -> AgentShape:
    """Normalise a sweep clearance argument (float radius or shape) to a shape."""
    if isinstance(r, (Disc, ConvexPolygon, AlignedShape)):
        return r
    return Disc(float(r))


def shape_from_spec(spec, default_radius: float | None = None) -> AgentShape:
    """Build a footprint from an agent-dict / YAML ``shape`` entry.

    Accepted forms (keys are case-insensitive; ``angle`` is in radians unless
    ``angle_deg`` is given instead):

    - a :class:`Disc` / :class:`ConvexPolygon` instance (returned as is);
    - ``{"type": "disc", "radius": r}``;
    - ``{"type": "rectangle", "width": w, "height": h, "angle": a, "center": [cx, cy]}``
      (``square`` with ``side`` is accepted too);
    - ``{"type": "regular_polygon" | "polygon", "n_sides": n, "circumradius": R | "radius": R | "inradius": r, "angle": a}``;
    - ``{"type": "convex", "vertices": [[x, y], ...]}`` or a bare vertex list.

    Polygons are heading-ALIGNED by default (``"align": true``): the polygon is
    the body-frame footprint with ``+x`` forward (so ``angle`` turns the body
    relative to the direction of travel) and the agent rotates it to face each
    edge it travels (:class:`AlignedShape`). ``"align": false`` keeps the
    polygon fixed in the world frame. ``default_radius`` is used by
    ``{"type": "disc"}`` without a radius.
    """
    if isinstance(spec, (Disc, ConvexPolygon, AlignedShape)):
        return spec
    if isinstance(spec, (list, tuple)):
        return AlignedShape(ConvexPolygon.from_points(spec))
    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported shape spec {spec!r}")
    d = {str(k).lower(): v for k, v in spec.items()}
    kind = str(d.get("type", "disc")).lower().replace("-", "_").replace(" ", "_")
    angle = float(d.get("angle", 0.0))
    if "angle_deg" in d:
        angle = math.radians(float(d["angle_deg"]))
    align = d.get("align", d.get("aligned", d.get("align_heading", True)))
    if isinstance(align, str):
        align = align.strip().lower() in ("1", "true", "yes", "y", "on")
    align = bool(align)
    if kind in ("disc", "disk", "circle"):
        r = d.get("radius", default_radius)
        if r is None:
            raise ValueError("disc shape needs a radius")
        return Disc(float(r))
    if kind in ("rectangle", "rect", "box"):
        poly = rectangle(float(d["width"]), float(d["height"]), angle=angle, center=d.get("center", (0.0, 0.0)))
    elif kind == "square":
        poly = square(float(d["side"]), angle=angle, center=d.get("center", (0.0, 0.0)))
    elif kind in ("regular_polygon", "regular", "polygon", "ngon", "n_gon"):
        n = int(d.get("n_sides", d.get("n", 0)))
        if "inradius" in d:
            poly = regular_polygon(n, inradius=float(d["inradius"]), angle=angle)
        else:
            R = d.get("circumradius", d.get("radius"))
            if R is None:
                raise ValueError("regular polygon needs circumradius / radius / inradius")
            poly = regular_polygon(n, float(R), angle=angle)
    elif kind in ("convex", "convex_polygon", "vertices"):
        poly = ConvexPolygon.from_points(d["vertices"])
    else:
        raise ValueError(f"Unknown shape type {kind!r}")
    return AlignedShape(poly) if align else poly


def shape_to_spec(shape: AgentShape) -> dict:
    """Serialisable dict for ``shape`` (inverse of :func:`shape_from_spec`)."""
    if isinstance(shape, Disc):
        return {"type": "disc", "radius": shape.radius}
    if isinstance(shape, AlignedShape):
        return {"type": "convex", "vertices": [list(v) for v in shape.body.vertices], "align": True}
    return {"type": "convex", "vertices": [list(v) for v in shape.vertices], "align": False}


# ----------------------------------------------------------------------------- motion


def moving_point_interval(
    x0: np.ndarray,
    vel: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    tdur: float | np.ndarray,
    exit_slack: float = 1e-9,
    eps: float = _EPS,
):
    """Time window in ``[0, tdur]`` during which ``x(t) = x0 + t * vel`` lies
    strictly inside the convex region ``{x : normals . x < offsets}``
    (Cyrus-Beck clipping, vectorised over ``N`` rows).

    Shapes: ``x0`` ``(N, 2)``; ``vel`` ``(2,)`` or ``(N, 2)``; ``normals``
    ``(M, 2)`` or ``(N, M, 2)``; ``offsets`` ``(M,)`` or ``(N, M)``; ``tdur``
    scalar or ``(N,)``. A zero normal with a positive offset encodes an
    always-satisfied constraint (used for padding).

    Returns ``(t_enter, t_exit, valid)``: ``valid[i]`` is True iff the path
    spends a positive-measure stretch of ``[0, tdur]`` strictly inside the
    region (tangencies and boundary-only contacts are NOT contacts, matching
    the strict ``<`` of the disc sweep). ``t_enter`` / ``t_exit`` are clipped
    to ``[0, tdur]``; ``t_exit`` carries the same ``exit_slack`` as the disc
    backends' ``get_interval_from_quadratic_equation``. Rows with
    ``valid == False`` hold meaningless times.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1, 2)
    n_rows = x0.shape[0]
    vel = np.asarray(vel, dtype=float)
    if vel.ndim == 1:
        vel = np.broadcast_to(vel.reshape(1, 2), (n_rows, 2))
    normals = np.asarray(normals, dtype=float)
    if normals.ndim == 2:
        normals = np.broadcast_to(normals[None, :, :], (n_rows,) + normals.shape)
    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim == 1:
        offsets = np.broadcast_to(offsets[None, :], (n_rows, offsets.shape[0]))
    tdur_arr = np.broadcast_to(np.asarray(tdur, dtype=float), (n_rows,))
    if n_rows == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty.copy(), np.empty(0, dtype=bool)

    # Constraint k on row i: g[i,k] + t * h[i,k] < 0.
    g = np.einsum("nmd,nd->nm", normals, x0) - offsets
    h = np.einsum("nmd,nd->nm", normals, vel)
    moving = np.abs(h) > eps
    ratio = -g / np.where(moving, h, 1.0)
    # h < 0: x moves inward through this half-plane -> lower bound on t.
    # h > 0: x moves outward through this half-plane -> upper bound on t.
    t_entry = np.max(np.where(h < -eps, ratio, -np.inf), axis=1)
    t_exit = np.min(np.where(h > eps, ratio, np.inf), axis=1)
    static_ok = np.all(moving | (g < 0.0), axis=1)
    valid = static_ok & (t_entry < t_exit) & (t_exit > 0.0) & (t_entry < tdur_arr)
    t_enter = np.clip(t_entry, 0.0, tdur_arr)
    t_leave = np.clip(t_exit + exit_slack, 0.0, tdur_arr)
    # Clipping +/-inf with np.clip is fine; make non-valid rows harmless.
    t_enter = np.where(valid, t_enter, 0.0)
    t_leave = np.where(valid, t_leave, 0.0)
    return t_enter, t_leave, valid
