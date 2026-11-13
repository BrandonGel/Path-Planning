import numpy as np
from numba import njit, prange
from shapely import Polygon, Point
from path_planning.common.world.collision_checking.agent_shape import Shape, Circle
from path_planning.common.world.collision_checking.disc_sweep import (   # noqa: F401  (re-exported)
    _merge_sorted_intervals, merge_intervals, intervals_to_dict)
import shapely
from shapely import affinity as _affinity

def undirected_key(src: int, tgt: int) -> tuple[int, int]:
    """Canonical undirected key; self-loops stay ``(i, i)``."""
    if src == tgt:
        return (src, tgt)
    return (src, tgt) if src < tgt else (tgt, src)

def convert_bidirectional_interval(t1, t2, tdur=None, tdur_rev=None):
    """Map a contact interval from one directed traversal to the reverse.

    Times are relative to the traversal start. For reverse motion of duration
    ``tdur_rev`` (defaults to ``tdur``), contact maps to
    ``(tdur_rev - t2', tdur_rev - t1')``, where ``t'`` is ``t`` rescaled from
    ``[0, tdur]`` into ``[0, tdur_rev]`` when the two durations differ.

    If ``tdur`` is None / non-finite / non-positive (e.g. stationary infinite
    horizon), returns ``(t1, t2)`` unchanged — same query-agent frame for
    undirected capsule geometry.

    Note: same-query expansion of reverse *edge keys* (a,b)↔(b,a) keeps the
    original interval (query frame); call this helper when remapping a result
    onto the reverse *query* / reverse traversal.
    """
    t1 = float(t1)
    t2 = float(t2)
    if tdur is None:
        return (t1, t2)
    tdur = float(tdur)
    if not np.isfinite(tdur) or tdur <= 0.0:
        return (t1, t2)
    if tdur_rev is None:
        tdur_rev = tdur
    else:
        tdur_rev = float(tdur_rev)
        if not np.isfinite(tdur_rev) or tdur_rev <= 0.0:
            return (t1, t2)
    # Rescale into the reverse duration when lengths/speeds differ, then flip.
    if abs(tdur_rev - tdur) > 1e-15:
        scale = tdur_rev / tdur
        t1 *= scale
        t2 *= scale
    lo = tdur_rev - t2
    hi = tdur_rev - t1
    return (lo, hi) if lo <= hi else (hi, lo)

def as_footprint(shape) -> Polygon:
    """Normalise ``shape`` to a body-frame ``Polygon`` at orientation 0 (centroid
    at the origin; apply ``shape_orientation`` and the direction of travel with
    ``place_footprint``). Accepts an ``agent_shape.Shape`` (a ``Circle`` becomes
    a disc, the others their ``outline()``), a number (disc of that radius), an
    ``(K, 2)`` array or point list (polygon outline) or a shapely ``Polygon``.
    The result must be a valid, non-empty polygon."""
    if isinstance(shape, Shape):
        if isinstance(shape, Circle):
            poly = Point(0.0, 0.0).buffer(float(shape.radius()), quad_segs=32)
        else:
            poly = Polygon(shape.outline())
    elif isinstance(shape, Polygon):
        poly = shape
    elif isinstance(shape, (int, float, np.integer, np.floating)):
        if shape <= 0:
            raise ValueError(f"as_footprint: a disc needs a positive radius, got {shape}")
        poly = Point(0.0, 0.0).buffer(float(shape), quad_segs=32)
    else:
        pts = np.asarray(shape, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            raise ValueError("as_footprint: expected a Shape, a radius, a (K>=3, 2) outline or a shapely Polygon")
        poly = Polygon(pts)
    if poly.is_empty or not poly.is_valid or poly.area <= 0.0:
        raise ValueError(f"as_footprint: footprint is not a valid non-empty polygon ({shapely.is_valid_reason(poly)})")
    return poly


def shape_orientation(shape) -> float:
    """Fixed rotation of the body relative to the direction of travel, in radians
    (``Shape.orientation()``; 0 for anything that is not a ``Shape``)."""
    return float(shape.orientation()) if isinstance(shape, Shape) else 0.0


def place_footprint(footprint: Polygon, pos, heading: float) -> Polygon:
    """The footprint in the world frame: rotated by ``heading`` (radians, the
    total world rotation: direction of travel + the shape's orientation) about
    the body origin, then translated to ``pos``."""
    poly = _affinity.rotate(footprint, heading, origin=(0.0, 0.0), use_radians=True) if heading != 0.0 else footprint
    return _affinity.translate(poly, float(pos[0]), float(pos[1]))


def convex_pieces(poly: Polygon) -> list:
    """``poly`` itself when convex, else the triangles of its constrained
    Delaunay triangulation: pieces whose Minkowski sum with a segment is a
    plain convex hull."""
    if abs(poly.convex_hull.area - poly.area) <= 1e-9 * max(1.0, poly.area):
        return [poly]
    return list(shapely.constrained_delaunay_triangles(poly).geoms)


def minkowski_segment(poly: Polygon, d) -> Polygon:
    """``poly (+) segment[0, d]``: the area a polygon covers when translated
    from 0 to ``d``. Exact for any simple polygon: hull(T, T + d) per
    triangle of a constrained Delaunay triangulation, unioned."""
    dx, dy = float(d[0]), float(d[1])
    if dx == 0.0 and dy == 0.0:
        return poly
    hulls = [shapely.union(piece, _affinity.translate(piece, dx, dy)).convex_hull
                for piece in convex_pieces(poly)]
    return hulls[0] if len(hulls) == 1 else shapely.union_all(hulls)


# ---------------------------------------------------------------------------
# Swept region of an agent along a polyline path
# ---------------------------------------------------------------------------

SWEEP_METHODS = ("buffer", "exact", "front_face")


def _bounding_radius(shape) -> float:
    if isinstance(shape, Shape):
        r = float(shape.radius())
    elif isinstance(shape, (int, float, np.integer, np.floating)):
        r = float(shape)
    else:
        xy = np.asarray(as_footprint(shape).exterior.coords)
        r = float(np.hypot(xy[:, 0], xy[:, 1]).max())
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"sweep_region: the shape needs a positive, finite radius, got {r}")
    return r


def _travel_extent(shape) -> tuple:
    """``(x_min, x_max, y_min, y_max)`` of the body in the travel frame (+x = direction of travel)."""
    if isinstance(shape, Shape):
        return shape.extent()
    xy = np.asarray(as_footprint(shape).exterior.coords)
    return float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max())


def _travel_extents(footprint: Polygon, rel) -> np.ndarray:
    """``(m, 4)`` ``(x_min, x_max, y_min, y_max)`` of the body-frame ``footprint`` rotated by each ``rel``
    (the body's rotation relative to the direction of travel), in the travel frame (+x along travel)."""
    xy = np.asarray(footprint.exterior.coords)[:-1]
    c, s = np.cos(rel)[:, None], np.sin(rel)[:, None]
    x = c * xy[None, :, 0] - s * xy[None, :, 1]
    y = s * xy[None, :, 0] + c * xy[None, :, 1]
    return np.stack([x.min(axis=1), x.max(axis=1), y.min(axis=1), y.max(axis=1)], axis=1)


def _place_coords(xy, rot, origin):
    """Body-frame ``xy`` ``(K, 2)`` rotated by each ``rot`` ``(m,)`` and moved to each ``origin`` ``(m, 2)``."""
    c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
    x = c * xy[None, :, 0] - s * xy[None, :, 1] + origin[:, 0:1]
    y = s * xy[None, :, 0] + c * xy[None, :, 1] + origin[:, 1:2]
    return np.stack([x, y], axis=-1)                                   # (m, K, 2)


def _exact_segment_sweeps(footprint: Polygon, orientation: float, path: np.ndarray, rot=None) -> list:
    """``footprint (+) segment`` for every segment, vectorised over segments: the footprint's convex
    pieces are found once, then hull(T, T + d) is built for all segments in one call per piece. The body
    is rotated by the direction of travel + ``orientation``, or by ``rot`` (``(m,)`` world rotations)."""
    seg = path[1:] - path[:-1]
    if rot is None:
        rot = np.arctan2(seg[:, 1], seg[:, 0]) + orientation
    m = len(seg)
    out = []
    for piece in convex_pieces(footprint):
        at_u = _place_coords(np.asarray(piece.exterior.coords)[:-1], rot, path[:-1])
        pts = np.concatenate([at_u, at_u + seg[:, None, :]], axis=1)      # (m, 2K, 2)
        idx = np.repeat(np.arange(m), pts.shape[1])
        out += list(shapely.convex_hull(shapely.multipoints(pts.reshape(-1, 2), indices=idx)))
    return out


def _front_face_segment_sweeps(shape, path: np.ndarray) -> list:
    """The body's front face swept along every segment: one rectangle per segment from the rear of the
    body (``x_min``, behind the centroid) to the segment end plus the front reach (``L + x_max``), with
    the lateral extent ``[y_min, y_max]`` (``shape.front_face()`` wide). The body's bounding box along
    the direction of travel: exact for a rectangle aligned with travel, a superset otherwise."""
    x_min, x_max, y_min, y_max = _travel_extent(shape)
    seg = path[1:] - path[:-1]
    length = np.hypot(seg[:, 0], seg[:, 1])
    head = np.arctan2(seg[:, 1], seg[:, 0])
    m = len(seg)
    lx = np.stack([np.full(m, x_min), length + x_max, length + x_max, np.full(m, x_min)], axis=1)
    ly = np.broadcast_to(np.array([y_min, y_min, y_max, y_max]), (m, 4))
    c, s = np.cos(head)[:, None], np.sin(head)[:, None]
    x = c * lx - s * ly + path[:-1, 0:1]
    y = s * lx + c * ly + path[:-1, 1:2]
    return list(shapely.polygons(np.stack([x, y], axis=-1)))


def _buffer_segment_sweeps(path: np.ndarray, radius: float, quad_segs: int) -> list:
    """ GEOS splits round joins/caps into chords up to ~twice the nominal pi / (4 quad_segs) half-angle,
    so circumscribe with the doubled angle (2% larger radius at quad_segs = 8) to never sag inside r. """
    geom = shapely.LineString(path) if len(path) > 1 else Point(path[0])
    
    r = radius / np.cos(np.pi / (2 * quad_segs))
    return [geom.buffer(r, quad_segs=quad_segs)]


def _is_disc(shape) -> bool:
    return isinstance(shape, Circle) or isinstance(shape, (int, float, np.integer, np.floating))


def _capsules_segment_sweeps(path: np.ndarray, radius: float, quad_segs: int) -> list:
    """Exact disc sweep per segment: ``segment.buffer(radius)``, vectorised. Radius circumscribed for the
    GEOS cap chords (see ``"buffer"``) so the polygon never sags inside the true capsule."""
    r = radius / np.cos(np.pi / (2 * quad_segs))
    if len(path) == 1:
        return [Point(path[0]).buffer(r, quad_segs=quad_segs)]
    lines = shapely.linestrings(np.stack([path[:-1], path[1:]], axis=1))
    return list(shapely.buffer(lines, r, quad_segs=quad_segs))

def _turn_discs_sweeps(path: np.ndarray, radius: float, quad_segs: int) -> list:
    """Bounding disc at every interior waypoint: contains the footprint at any rotation, so it covers
    the turn in place. Circumscribed polygon (no sag inside the true circle)."""
    if len(path) <= 2:
        return []
    r = radius / np.cos(np.pi / (4 * quad_segs))
    return list(shapely.buffer(shapely.points(path[1:-1]), r, quad_segs=quad_segs))

def sweep_region(path, shape, method: str = "front_face", quad_segs: int = 8) -> list:
    """Region covered by ``shape`` moving along the polyline ``path`` (``(P, 2)``), heading = direction
    of travel + the shape's orientation, turning in place at the waypoints. Returns a list of pieces
    whose union is the region (query them directly, e.g. with an STRtree; ``shapely.union_all`` merges).

    ``method``:
      ``"buffer"``      ``LineString(path).buffer(radius)``: the bounding disc swept along the path.
                        One piece for a polygon (a loose superset); a disc agent gets capsules (below).
      ``"exact"``       exact footprint sweep on every segment (Minkowski sum with the segment) plus
                        the bounding disc at every turn. Tightest superset of the true region.
      ``"front_face"``  the front face swept along every segment (the body's bounding box along the
                        direction of travel) plus the bounding disc at every turn. Cheapest to build;
                        exact for travel-aligned rectangles, a few percent larger otherwise.

    A disc agent (``Circle`` or a bare radius) is rotation-invariant, so every method (``"buffer"``,
    ``"exact"`` and ``"front_face"``) returns one capsule per segment (``segment.buffer(radius)``, the exact disc sweep; no turn discs
    needed since each capsule already ends in the disc), built in one vectorised call instead of sweeping
    a 128-gon footprint.

    ``shape`` is an ``agent_shape.Shape`` (or anything ``as_footprint`` accepts). ``quad_segs`` sets the
    disc resolution (``4 * quad_segs`` sides) for the turns, the capsules and ``"buffer"``.
    """
    if method not in SWEEP_METHODS:
        raise ValueError(f"sweep_region: method must be one of {SWEEP_METHODS}, got {method!r}")
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) == 0:
        return []
    radius = _bounding_radius(shape)
    if _is_disc(shape):
        return _capsules_segment_sweeps(path, radius, quad_segs)
    if method == "buffer":
        return _buffer_segment_sweeps(path,radius,quad_segs)
    footprint, orientation = as_footprint(shape), shape_orientation(shape)
    if len(path) == 1:
        return [place_footprint(footprint, path[0], orientation)]
    if method == "exact":
        pieces = _exact_segment_sweeps(footprint, orientation, path)
    else:
        pieces = _front_face_segment_sweeps(shape, path)
    return pieces + _turn_discs_sweeps(path, radius, quad_segs)


# ---------------------------------------------------------------------------
# Swept region over time
# ---------------------------------------------------------------------------

def fixed_orientation(turning_speed) -> bool:
    """Whether ``turning_speed`` means the body never turns (``0``)."""
    return turning_speed is not None and turning_speed == 0


def path_schedule(path, velocity: float, t0: float = 0.0, turning_speed=None):
    """Timing of an agent along the polyline ``path``.

    ``velocity``: constant speed on every segment; ``None`` means every segment takes 1 time unit, whatever
    its length. ``turning_speed`` (rad/s) at each interior waypoint: ``None`` or ``inf`` for instant
    turns, ``0`` for no turning at all (the body keeps one orientation along the whole path; see
    ``sweep_rings_timed``), otherwise the turn takes ``angle / turning_speed``. Turns with ``None``,
    ``inf`` or ``0`` take no time.

    Returns ``(path, seg_windows, turn_windows, headings)``: the path with consecutive duplicate waypoints
    dropped, ``(m, 2)`` ``(depart, arrive)`` per segment, ``(P - 2, 2)`` ``(start, end)`` per interior
    turn, and the ``(m,)`` direction of travel of each segment."""
    if velocity is not None and not (np.isfinite(velocity) and velocity > 0):
        raise ValueError(f"path_schedule: velocity must be positive and finite, or None (1 time unit per segment), got {velocity}")
    if turning_speed is not None and not turning_speed >= 0:
        raise ValueError(f"path_schedule: turning_speed must be >= 0, inf or None, got {turning_speed}")
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) > 1:
        path = path[np.r_[True, np.any(np.diff(path, axis=0) != 0.0, axis=1)]]
    seg = path[1:] - path[:-1]
    headings = np.arctan2(seg[:, 1], seg[:, 0])
    move = np.ones(len(seg)) if velocity is None else np.hypot(seg[:, 0], seg[:, 1]) / velocity
    angle = np.abs((np.diff(headings) + np.pi) % (2.0 * np.pi) - np.pi)
    instant = turning_speed is None or np.isinf(turning_speed) or turning_speed == 0
    turn = angle * 0.0 if instant else angle / turning_speed
    durations = np.empty(max(2 * len(move) - 1, 0))
    durations[0::2], durations[1::2] = move, turn                       # move, turn, move, ..., move
    ends = t0 + np.cumsum(durations)
    starts = np.r_[t0, ends[:-1]] if len(ends) else ends
    seg_windows = np.column_stack([starts[0::2], ends[0::2]])
    turn_windows = np.column_stack([starts[1::2], ends[1::2]])
    return path, seg_windows, turn_windows, headings


def _waypoint_pieces(path, shape, seg_windows, turn_windows, headings, t0, goal_hold, quad_segs):
    """Pieces at the waypoints with their windows: the bounding disc during every interior turn, and the
    footprint at the last waypoint for ``[arrival, arrival + goal_hold]`` (always for a one-point path)."""
    radius = _bounding_radius(shape)
    pieces = _turn_discs_sweeps(path, radius, quad_segs)
    windows = [turn_windows.reshape(-1, 2)]
    arrive = float(seg_windows[-1, 1]) if len(seg_windows) else float(t0)
    if goal_hold > 0 or len(path) == 1:
        if _is_disc(shape):
            r = radius / np.cos(np.pi / (4 * quad_segs))
            goal = Point(path[-1]).buffer(r, quad_segs=quad_segs)
        else:
            rot = (float(headings[-1]) if len(headings) else 0.0) + shape_orientation(shape)
            goal = place_footprint(as_footprint(shape), path[-1], rot)
        pieces.append(goal)
        windows.append(np.array([[arrive, arrive + float(goal_hold)]]))
    return pieces, np.concatenate(windows)


def sweep_region_timed(path, shape, velocity: float, method: str = "front_face", t0: float = 0.0,
                       turning_speed=None, goal_hold: float = 0.0, quad_segs: int = 8):
    """``sweep_region`` with a time window per piece: ``(pieces, windows, segment)``, ``windows[i] =
    (t_start, t_end)`` is when the body can be inside ``pieces[i]`` (see ``path_schedule`` for the timing)
    and ``segment[i]`` the path segment a piece sweeps (``-1`` for the turn discs and the goal footprint).

    Segment pieces get their segment's ``(depart, arrive)``, turn discs their turn's ``(start, end)`` and the
    goal footprint ``[arrival, arrival + goal_hold]`` (``goal_hold`` may be ``inf``). A window covers the whole
    piece, so it is conservative: an element at the front of a segment piece is reported from the departure,
    though the body reaches it later. ``"buffer"`` uses one capsule of the bounding radius per segment, so each
    segment keeps its own window (same union as the untimed ``"buffer"``)."""
    if method not in SWEEP_METHODS:
        raise ValueError(f"sweep_region_timed: method must be one of {SWEEP_METHODS}, got {method!r}")
    path, seg_w, turn_w, headings = path_schedule(path, velocity, t0, turning_speed)
    if len(path) == 0:
        return [], np.empty((0, 2)), np.empty(0, dtype=np.int64)
    radius = _bounding_radius(shape)
    pieces, windows, segment = [], [], []
    if len(path) >= 2:
        if _is_disc(shape) or method == "buffer":
            segs = _capsules_segment_sweeps(path, radius, quad_segs)
        elif method == "exact":
            segs = _exact_segment_sweeps(as_footprint(shape), shape_orientation(shape), path)
        else:
            segs = _front_face_segment_sweeps(shape, path)
        reps = len(segs) // len(seg_w)                                   # exact: convex piece-major order
        pieces += segs
        windows.append(np.tile(seg_w, (reps, 1)))
        segment.append(np.tile(np.arange(len(seg_w)), reps))
    wp, ww = _waypoint_pieces(path, shape, seg_w, turn_w, headings, t0, goal_hold, quad_segs)
    segment.append(np.full(len(wp), -1))
    return pieces + wp, np.concatenate(windows + [ww]), np.concatenate(segment).astype(np.int64)


def travel_reach(shape) -> tuple:
    """``(x_min, x_max)``: how far the body reaches behind / ahead of its reference point along the direction
    of travel (orientation applied; ``(-r, r)`` for a disc)."""
    if _is_disc(shape):
        r = _bounding_radius(shape)
        return -r, r
    x_min, x_max, _, _ = _travel_extent(shape)
    return x_min, x_max


def clip_segment_windows(path, velocity, depart, segment, s_lo, s_hi, x_min, x_max):
    """Tighten piece windows on segments: an element spanning ``[s_lo, s_hi]`` along segment ``k`` (distance
    from its start, in the travel direction) can only touch a body reaching ``[x_min, x_max]`` around its
    reference while the reference is in ``[s_lo - x_max, s_hi - x_min]``, clipped to the segment. Returns
    ``(t1, t2)`` arrays; exact for a travel-aligned rectangle, a superset for any other shape."""
    seg = path[1:] - path[:-1]
    length = np.hypot(seg[:, 0], seg[:, 1])[segment]
    lo = np.clip(s_lo - x_max, 0.0, length)
    hi = np.clip(s_hi - x_min, 0.0, length)
    hi = np.maximum(hi, lo)
    return depart + lo / velocity, depart + hi / velocity


def group_intervals(items) -> dict:
    """``{key: [(t1, t2), ...]}`` from ``(key, t1, t2)`` items with any hashable keys: each key's closed
    windows sorted and merged where they overlap or touch (via ``merge_intervals``)."""
    index, keys, ids, t1, t2 = {}, [], [], [], []
    for k, a, b in items:
        i = index.get(k)
        if i is None:
            i = index[k] = len(keys)
            keys.append(k)
        ids.append(i); t1.append(a); t2.append(b)
    if not keys:
        return {}
    m_ids, m_a, m_b = merge_intervals(ids, t1, t2)
    return intervals_to_dict(m_ids, m_a, m_b, [keys[i] for i in m_ids.tolist()])

# ---------------------------------------------------------------------------
# Segment clipping against polygon pieces
# ---------------------------------------------------------------------------

def _is_convex(pieces) -> np.ndarray:
    """Whether each polygon equals its convex hull (to a relative 1e-9 in area)."""
    area = shapely.area(pieces)
    return np.abs(shapely.area(shapely.convex_hull(pieces)) - area) <= 1e-9 * np.maximum(area, 1.0)


def piece_rings(pieces, known_convex=None) -> tuple:
    """Flat ring arrays of polygon ``pieces`` for the numba kernels: ``(ring_start, ring_end, rx, ry,
    sign, convex)``; ``sign`` is +1 for a counter-clockwise ring, ``convex`` flags convex pieces.
    ``known_convex`` (bool array) marks pieces known to be convex; only the others get the hull check."""
    pieces = np.asarray(pieces, dtype=object)
    rings = shapely.get_exterior_ring(pieces)
    convex = np.zeros(len(pieces), dtype=bool) if known_convex is None else np.asarray(known_convex, dtype=bool).copy()
    if not convex.all():
        convex[~convex] = _is_convex(pieces[~convex])
    coords, owner = shapely.get_coordinates(rings, return_index=True)
    counts = np.bincount(owner, minlength=len(pieces))
    ring_end = np.cumsum(counts)
    sign = np.where(shapely.is_ccw(rings), 1.0, -1.0)
    return ring_end - counts, ring_end, np.ascontiguousarray(coords[:, 0]), np.ascontiguousarray(coords[:, 1]), sign, convex


PIECE_RING, PIECE_CAPSULE, PIECE_SWEEP = 0, 1, 2
# piece kinds: polygon ring; capsule (a disc when a == b); a convex piece of the footprint at the start of
# a segment, moved along it (exact contact windows, see ``_sweep_vertex_range`` / ``_sweep_edge_range``)


def shrunk_convex_pieces(footprint: Polygon, eps: float = 0.0) -> list:
    """The convex pieces of ``footprint`` (``convex_pieces``) as closed counter-clockwise ``(K + 1, 2)``
    rings, with every side on the footprint's boundary moved inward by ``eps``. Internal seams between
    pieces stay put, so the pieces still cover each other's shared sides: an element on a seam is inside
    the body, while one within ``eps`` of the body's boundary is not (touching is not contact)."""
    boundary = footprint.boundary
    scale = max(1.0, float(np.sqrt(footprint.area)))
    out = []
    for piece in convex_pieces(footprint):
        xy = np.asarray(shapely.geometry.polygon.orient(piece, 1.0).exterior.coords)[:-1]
        prev, nxt = np.roll(xy, 1, axis=0), np.roll(xy, -1, axis=0)
        turn = (xy[:, 0] - prev[:, 0]) * (nxt[:, 1] - xy[:, 1]) - (xy[:, 1] - prev[:, 1]) * (nxt[:, 0] - xy[:, 0])
        xy = xy[np.abs(turn) > 1e-12 * scale * scale]                   # drop collinear vertices
        if eps > 0.0:
            e = np.roll(xy, -1, axis=0) - xy
            length = np.hypot(e[:, 0], e[:, 1])
            inward = np.column_stack([-e[:, 1], e[:, 0]]) / length[:, None]
            mid = shapely.points(xy + 0.5 * e)
            on_boundary = shapely.distance(boundary, mid) <= 1e-9 * scale
            q = xy + np.where(on_boundary, eps, 0.0)[:, None] * inward  # a point on each (moved) side line
            qp, ep = np.roll(q, 1, axis=0), np.roll(e, 1, axis=0)       # side i-1 meets side i at vertex i
            den = ep[:, 0] * e[:, 1] - ep[:, 1] * e[:, 0]
            a = ((q[:, 0] - qp[:, 0]) * e[:, 1] - (q[:, 1] - qp[:, 1]) * e[:, 0]) / den
            xy = qp + a[:, None] * ep
        out.append(np.vstack([xy, xy[:1]]))
    return out


def sweep_rings_timed(path, shape, velocity: float, method: str = "front_face", t0: float = 0.0,
                      turning_speed=None, goal_hold: float = 0.0, segments: bool = True,
                      exact: bool = False, eps: float = 0.0, heading=None) -> dict:
    """The swept pieces of ``shape`` along ``path`` with their time windows, as flat arrays for the numba
    kernels. Three piece kinds: polygon rings, capsules ``(ax, ay, bx, by, r)`` (a disc when a == b),
    tested exactly by distance with no polygon approximation, and swept convex pieces (``exact``):

    - a disc agent (``Circle`` or a bare radius), any method: one capsule per segment, a disc per turn and
      for the goal hold;
    - ``exact=True``, polygon agent: per segment, each convex piece of the footprint (``shrunk_convex_pieces``
      with ``eps``) placed at the segment start; the kernels move it along the segment and give the exact
      contact window (``method`` is unused for the segments);
    - ``"buffer"``: one capsule of the bounding radius per segment;
    - ``"front_face"`` / ``"exact"``: the rectangle / exact hull rings per segment;
    - every turn of a polygon agent: a disc of its bounding radius; its goal hold: the footprint ring.

    Timing as in ``path_schedule``: ``velocity = None`` gives every segment 1 time unit. ``turning_speed = 0``
    means the body never turns: it keeps the first segment's rotation (its direction + the shape's
    orientation) on every segment and at the goal, with no turn discs; the front-face rectangles and the
    reach along travel use the body's extent at that rotation on each segment. ``heading`` (radians)
    fixes that rotation instead: the body faces ``heading`` (+ its orientation) along the whole path and
    never turns, whatever ``turning_speed`` is (as ``heading`` in ``place_footprint``).

    ``segments=False`` returns only the waypoint pieces (turns and goal). Returns ``{"path", "seg_windows",
    "seg_speed", "reach", "windows", "segment", "kind", "cap", "rings", "bounds", "geoms"}``: the cleaned
    path, per-segment ``(depart, arrive)``, per-segment speed (length / duration), per-piece reach
    ``(x_min, x_max)`` of the body along its segment's travel (0 for turn / goal pieces), per-piece windows, segment tags (``-1`` for turn / goal pieces), kinds, capsule
    parameters (``(n, 5)``), ``rings`` as ``piece_rings`` gives them (empty rings for capsules), bounds
    ``(xmin, ymin, xmax, ymax)`` (a swept piece's cover its whole segment), and ``geoms`` holding the
    polygon only for a concave goal footprint."""
    if method not in SWEEP_METHODS:
        raise ValueError(f"sweep_rings_timed: method must be one of {SWEEP_METHODS}, got {method!r}")
    p, seg_w, turn_w, _ = path_schedule(path, velocity, t0, 0 if heading is not None else turning_speed)
    coords, counts, windows, segment, kind, cap, convex, geoms = [], [], [], [], [], [], [], []

    def add_rings(c, cnt, w, seg, cvx=True, g=None, k=PIECE_RING):
        coords.append(np.asarray(c, dtype=float).reshape(-1, 2)); counts.append(np.asarray(cnt, dtype=np.int64))
        windows.append(np.asarray(w, dtype=float).reshape(-1, 2)); segment.append(np.asarray(seg, dtype=np.int64))
        kind.append(np.full(len(cnt), k, np.int8)); cap.append(np.zeros((len(cnt), 5)))
        convex.append(np.full(len(cnt), cvx, dtype=bool)); geoms.extend([g] * len(cnt))

    def add_caps(a, b, r, w, seg):
        a = np.asarray(a, dtype=float).reshape(-1, 2); b = np.asarray(b, dtype=float).reshape(-1, 2)
        k = len(a)
        coords.append(np.empty((0, 2))); counts.append(np.zeros(k, dtype=np.int64))
        windows.append(np.asarray(w, dtype=float).reshape(-1, 2)); segment.append(np.asarray(seg, dtype=np.int64))
        kind.append(np.full(k, PIECE_CAPSULE, np.int8)); cap.append(np.column_stack([a, b, np.full(k, float(r))]))
        convex.append(np.ones(k, dtype=bool)); geoms.extend([None] * k)

    disc = _is_disc(shape)
    radius = _bounding_radius(shape) if len(p) else 0.0
    fixed = heading is not None or fixed_orientation(turning_speed)
    ori = shape_orientation(shape)
    seg = p[1:] - p[:-1]
    m = len(seg)
    length = np.hypot(seg[:, 0], seg[:, 1])
    head = np.arctan2(seg[:, 1], seg[:, 0])
    # World rotation of the body on each segment: the direction of travel + its orientation, or (never
    # turning) the first segment's rotation along the whole path.
    if heading is not None:
        rot = np.full(m, float(heading) + ori)
    else:
        rot = np.full(m, head[0] + ori) if fixed and m else head + ori
    # The body's extent in each segment's travel frame (x along travel): (m, 4) x_min, x_max, y_min, y_max.
    if disc:
        ext = np.tile([-radius, radius, -radius, radius], (m, 1))
    elif not fixed:
        ext = np.tile(_travel_extent(shape), (m, 1))
    else:
        ext = _travel_extents(as_footprint(shape), rot - head)
    seg_speed = length.copy() if velocity is None else np.full(m, float(velocity))   # None: 1 time unit per segment
    if segments and m:
        if disc or (method == "buffer" and not exact):
            add_caps(p[:-1], p[1:], radius, seg_w, np.arange(m))
        elif exact:                                              # convex pieces at each segment start
            for ring in shrunk_convex_pieces(as_footprint(shape), eps):
                add_rings(_place_coords(ring, rot, p[:-1]), np.full(m, len(ring)), seg_w, np.arange(m),
                          k=PIECE_SWEEP)
        elif method == "exact":
            g = _exact_segment_sweeps(as_footprint(shape), ori, p, rot)
            c, own = shapely.get_coordinates(shapely.get_exterior_ring(np.asarray(g, dtype=object)), return_index=True)
            reps = len(g) // m                                   # convex piece-major order
            add_rings(c, np.bincount(own, minlength=len(g)), np.tile(seg_w, (reps, 1)), np.tile(np.arange(m), reps))
        else:                                                    # front_face rectangles (as _front_face_segment_sweeps)
            x_min, x_max, y_min, y_max = (ext[:, i:i + 1] for i in range(4))
            lx = np.concatenate([x_min, length[:, None] + x_max, length[:, None] + x_max, x_min, x_min], axis=1)
            ly = np.concatenate([y_min, y_min, y_max, y_max, y_min], axis=1)
            cs, sn = np.cos(head)[:, None], np.sin(head)[:, None]
            ring = np.stack([cs * lx - sn * ly + p[:-1, 0:1], sn * lx + cs * ly + p[:-1, 1:2]], axis=-1)
            add_rings(ring, np.full(m, 5), seg_w, np.arange(m))
    if len(p) > 2 and not fixed:                                 # in-place turns: the bounding disc
        add_caps(p[1:-1], p[1:-1], radius, turn_w, np.full(len(p) - 2, -1))
    if len(p) and (goal_hold > 0 or len(p) == 1):                # footprint held at the goal
        arrive = float(seg_w[-1, 1]) if len(seg_w) else float(t0)
        w = [[arrive, arrive + float(goal_hold)]]
        if disc:
            add_caps(p[-1:], p[-1:], radius, w, [-1])
        else:
            g = place_footprint(as_footprint(shape), p[-1], float(rot[-1]) if m else ori + float(heading or 0.0))
            cvx = bool(_is_convex(np.array([g], dtype=object))[0])
            add_rings(np.asarray(g.exterior.coords), [len(g.exterior.coords)], w, [-1], cvx, None if cvx else g)
    if not counts:
        add_rings(np.empty((0, 2)), np.empty(0, dtype=np.int64), np.empty((0, 2)), np.empty(0))

    rc = np.concatenate(coords)
    cnt = np.concatenate(counts)
    ring_end = np.cumsum(cnt)
    ring_start = ring_end - cnt
    rx, ry = np.ascontiguousarray(rc[:, 0]), np.ascontiguousarray(rc[:, 1])
    kd = np.concatenate(kind)
    cp = np.concatenate(cap)
    n = len(cnt)
    sign = np.ones(n)
    bounds = np.zeros((n, 4))
    ring = np.flatnonzero(cnt > 0)
    if len(ring):
        term = np.zeros(len(rx))
        term[:-1] = rx[:-1] * ry[1:] - rx[1:] * ry[:-1]
        term[ring_end[ring] - 1] = 0.0                           # no term across ring boundaries
        st = ring_start[ring]
        sign[ring] = np.where(np.add.reduceat(term, st) > 0, 1.0, -1.0)
        bounds[ring] = np.stack([np.minimum.reduceat(rx, st), np.minimum.reduceat(ry, st),
                                 np.maximum.reduceat(rx, st), np.maximum.reduceat(ry, st)], axis=1)
    sw = np.flatnonzero(kd == PIECE_SWEEP)
    if len(sw):                                                  # bounds of the piece over its whole segment
        sg = np.concatenate(segment)[sw]
        d = p[1:][sg] - p[:-1][sg]
        bounds[sw] = np.column_stack([bounds[sw, 0] + np.minimum(d[:, 0], 0.0), bounds[sw, 1] + np.minimum(d[:, 1], 0.0),
                                      bounds[sw, 2] + np.maximum(d[:, 0], 0.0), bounds[sw, 3] + np.maximum(d[:, 1], 0.0)])
    caps = np.flatnonzero(kd == PIECE_CAPSULE)
    if len(caps):
        c = cp[caps]
        bounds[caps] = np.column_stack([np.minimum(c[:, 0], c[:, 2]) - c[:, 4], np.minimum(c[:, 1], c[:, 3]) - c[:, 4],
                                        np.maximum(c[:, 0], c[:, 2]) + c[:, 4], np.maximum(c[:, 1], c[:, 3]) + c[:, 4]])
    sgm = np.concatenate(segment)
    reach = np.zeros((n, 2))
    on_seg = sgm >= 0
    reach[on_seg] = ext[sgm[on_seg]][:, :2]
    return {"path": p, "seg_windows": seg_w, "seg_speed": seg_speed, "reach": reach,
            "windows": np.concatenate(windows), "segment": sgm,
            "kind": kd, "cap": cp, "rings": (ring_start, ring_end, rx, ry, sign, np.concatenate(convex)),
            "bounds": bounds, "geoms": np.array(geoms + [None], dtype=object)[:-1]}


@njit(cache=True)
def _convex_contains(px, py, piece, ring_start, ring_end, rx, ry, sign, eps):
    """Whether each point lies inside its convex ring by at least ``eps`` from every side."""
    n = len(px)
    out = np.empty(n, dtype=np.bool_)
    for k in range(n):
        p = piece[k]
        ok = True
        for j in range(ring_start[p], ring_end[p] - 1):
            ex = rx[j + 1] - rx[j]
            ey = ry[j + 1] - ry[j]
            if sign[p] * (ex * (py[k] - ry[j]) - ey * (px[k] - rx[j])) < eps * np.sqrt(ex * ex + ey * ey):
                ok = False
                break
        out[k] = ok
    return out


def points_in_pieces(pieces, piece_idx, xy, eps: float = 0.0, rings=None) -> np.ndarray:
    """Whether each point ``xy[i]`` lies inside ``pieces[piece_idx[i]]`` by more than ``eps`` (the piece
    shrunk by ``eps``; boundary excluded). Convex pieces in numba, concave ones through shapely."""
    pieces = np.asarray(pieces, dtype=object)
    piece_idx = np.asarray(piece_idx, dtype=np.int64)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    out = np.zeros(len(xy), dtype=bool)
    if len(xy) == 0:
        return out
    rs, re, rx, ry, sign, convex = piece_rings(pieces) if rings is None else rings
    on = convex[piece_idx]
    if on.any():
        i = np.flatnonzero(on)
        out[i] = _convex_contains(xy[i, 0], xy[i, 1], piece_idx[i], rs, re, rx, ry, sign, eps)
    if not on.all():
        i = np.flatnonzero(~on)
        shrunk = shapely.buffer(pieces[piece_idx[i]], -eps) if eps > 0 else pieces[piece_idx[i]]
        out[i] = shapely.contains_xy(shrunk, xy[i, 0], xy[i, 1])          # boundary excluded
    return out


@njit(cache=True)
def _clip_segments_convex(ax, ay, bx, by, piece, ring_start, ring_end, rx, ry, sign, eps):
    """Cyrus-Beck: parameter range ``[lo, hi]`` of each segment ``a -> b`` inside its convex ring shrunk
    by ``eps`` (``lo > hi`` when it misses). Points on the shrunk boundary count as inside."""
    n = len(ax)
    lo_out = np.empty(n)
    hi_out = np.empty(n)
    for k in range(n):
        p = piece[k]
        dx = bx[k] - ax[k]
        dy = by[k] - ay[k]
        lo = 0.0
        hi = 1.0
        for j in range(ring_start[p], ring_end[p] - 1):          # closed ring: last point == first
            ex = rx[j + 1] - rx[j]
            ey = ry[j + 1] - ry[j]
            num = sign[p] * (ex * (ay[k] - ry[j]) - ey * (ax[k] - rx[j])) - eps * np.sqrt(ex * ex + ey * ey)
            den = sign[p] * (ex * dy - ey * dx)
            if den == 0.0:
                if num < 0.0:
                    lo, hi = 1.0, 0.0
                    break
                continue
            t = -num / den
            if den > 0.0:
                if t > lo:
                    lo = t
            elif t < hi:
                hi = t
            if lo > hi:
                break
        lo_out[k] = lo
        hi_out[k] = hi
    return lo_out, hi_out


def clip_segments_to_pieces(pieces, piece_idx, a, b, eps: float = 0.0, rings=None):
    """``(lo, hi)``: the parameter range ``t`` in ``[0, 1]`` of each segment ``a[i] + t * (b[i] - a[i])``
    inside ``pieces[piece_idx[i]]`` shrunk by ``eps`` (``lo > hi`` when it misses). Convex pieces are
    clipped in numba; a concave piece falls back to ``shapely.intersection`` (envelope of its parts).
    ``rings`` is ``piece_rings(pieces)`` when already computed."""
    pieces = np.asarray(pieces, dtype=object)
    piece_idx = np.asarray(piece_idx, dtype=np.int64)
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    lo = np.ones(len(a))
    hi = np.zeros(len(a))
    if len(a) == 0:
        return lo, hi
    rs, re, rx, ry, sign, convex = piece_rings(pieces) if rings is None else rings
    on_convex = convex[piece_idx]
    if on_convex.any():
        i = np.flatnonzero(on_convex)
        lo[i], hi[i] = _clip_segments_convex(a[i, 0], a[i, 1], b[i, 0], b[i, 1], piece_idx[i],
                                             rs, re, rx, ry, sign, eps)
    if not on_convex.all():
        i = np.flatnonzero(~on_convex)
        shrunk = shapely.buffer(pieces[piece_idx[i]], -eps) if eps > 0 else pieces[piece_idx[i]]
        inter = shapely.intersection(shrunk, shapely.linestrings(np.stack([a[i], b[i]], axis=1)))
        pts, who = shapely.get_coordinates(inter, return_index=True)
        if len(pts):
            d = b[i] - a[i]
            t = np.einsum("ij,ij->i", pts - a[i][who], d[who]) / np.maximum(np.einsum("ij,ij->i", d, d)[who], 1e-300)
            l2 = np.full(len(i), np.inf); h2 = np.full(len(i), -np.inf)
            np.minimum.at(l2, who, t); np.maximum.at(h2, who, t)
            ok = shapely.length(inter) > 1e-12
            lo[i[ok]], hi[i[ok]] = l2[ok], h2[ok]
    return lo, hi


# ---------------------------------------------------------------------------
# Fused hit kernels: inside test / clipping + distance along the segment + time window, in one pass
# ---------------------------------------------------------------------------

@njit(cache=True)
def _segment_window(s_lo, s_hi, g, vel, xmin, xmax, seg_len, seg_dep):
    """``clip_segment_windows`` for one hit: times the body's reach ``[xmin, xmax]`` spans ``[s_lo, s_hi]``."""
    L = seg_len[g]
    lo = min(max(s_lo - xmax, 0.0), L)
    hi = min(max(s_hi - xmin, 0.0), L)
    if hi < lo:
        hi = lo
    return seg_dep[g] + lo / vel, seg_dep[g] + hi / vel


@njit(cache=True)
def _linear_range(a, b, lo_b, hi_b, t0, t1):
    """Narrow ``[t0, t1]`` to where ``a + b * t`` lies in ``[lo_b, hi_b]``."""
    if b == 0.0:
        if a < lo_b or a > hi_b:
            return 1.0, 0.0
        return t0, t1
    u, v = (lo_b - a) / b, (hi_b - a) / b
    if u > v:
        u, v = v, u
    return max(t0, u), min(t1, v)


@njit(cache=True)
def _line_disc_range(sx, sy, ux, uy, cx, cy, r):
    """Where ``S + t*u`` (``u`` unit) is within ``r`` of ``C``: ``(lo, hi)``, ``lo > hi`` when never."""
    wx, wy = sx - cx, sy - cy
    bq = ux * wx + uy * wy
    disc = bq * bq - (wx * wx + wy * wy - r * r)
    if disc <= 0.0:
        return 1.0, 0.0
    q = np.sqrt(disc)
    return -bq - q, -bq + q


@njit(cache=True)
def _line_capsule_range(sx, sy, ux, uy, ax, ay, bx, by, r):
    """Where ``S + t*u`` (``u`` unit) is within ``r`` of the segment ``AB``: one interval (the capsule is
    convex), the union of the end discs and the band along ``AB``; ``lo > hi`` when never."""
    lo, hi = _line_disc_range(sx, sy, ux, uy, ax, ay, r)
    l2, h2 = _line_disc_range(sx, sy, ux, uy, bx, by, r)
    if h2 > l2:
        if hi > lo:
            lo, hi = min(lo, l2), max(hi, h2)
        else:
            lo, hi = l2, h2
    ex, ey = bx - ax, by - ay
    la = np.sqrt(ex * ex + ey * ey)
    if la > 0.0:
        ex /= la
        ey /= la
        t0, t1 = _linear_range((sx - ax) * ex + (sy - ay) * ey, ux * ex + uy * ey, 0.0, la, -1e300, 1e300)
        t0, t1 = _linear_range((sx - ax) * ey - (sy - ay) * ex, ux * ey - uy * ex, -r, r, t0, t1)
        if t1 > t0:
            if hi > lo:
                lo, hi = min(lo, t0), max(hi, t1)
            else:
                lo, hi = t0, t1
    return lo, hi


@njit(cache=True)
def _point_segment_dist2(px, py, ax, ay, bx, by):
    ex, ey = bx - ax, by - ay
    l2 = ex * ex + ey * ey
    t = 0.0 if l2 == 0.0 else min(max(((px - ax) * ex + (py - ay) * ey) / l2, 0.0), 1.0)
    dx, dy = px - (ax + t * ex), py - (ay + t * ey)
    return dx * dx + dy * dy


@njit(cache=True)
def _sweep_vertex_range(px, py, ux, uy, L, p, rs, re, rx, ry, sign):
    """Swept piece ``p`` (convex ring placed at the segment start, moving ``s`` along unit ``u``, ``s`` in
    ``[0, L]``): the range of ``s`` where the point is inside it. In the piece's frame the point moves
    ``P - s * u``; Cyrus-Beck clips that against the ring (sides inclusive; the ring is already shrunk).
    ``lo > hi`` when it never is."""
    dx, dy = -L * ux, -L * uy
    lo, hi = 0.0, 1.0
    for j in range(rs[p], re[p] - 1):
        ex = rx[j + 1] - rx[j]
        ey = ry[j + 1] - ry[j]
        num = sign[p] * (ex * (py - ry[j]) - ey * (px - rx[j]))
        den = sign[p] * (ex * dy - ey * dx)
        if den == 0.0:
            if num < 0.0:
                return 1.0, 0.0
            continue
        t = -num / den
        if den > 0.0:
            if t > lo:
                lo = t
        elif t < hi:
            hi = t
        if lo > hi:
            return 1.0, 0.0
    return lo * L, hi * L


@njit(cache=True)
def _sweep_edge_range(ax, ay, bx, by, ux, uy, L, p, rs, re, rx, ry, sign):
    """Swept piece ``p`` (as in ``_sweep_vertex_range``) against the edge ``AB``: the range of ``s`` in
    ``[0, L]`` where the moved piece ``Q + s * u`` meets ``AB``, i.e. where ``s * u`` lies in the convex
    polygon ``AB (+) (-Q)``. That polygon's sides are normal to the sides of ``Q`` and to ``AB``; its
    support along a normal ``n`` is ``max(n.A, n.B) + max_q(-n.q)``, so the line ``s * u`` is clipped by
    one half-plane per normal, without building the polygon. ``lo > hi`` when they never meet."""
    lo, hi = 0.0, L
    for j in range(rs[p], re[p] - 1):                            # normals -n_j (n_j: outward normal of side j)
        ex = rx[j + 1] - rx[j]
        ey = ry[j + 1] - ry[j]
        nx, ny = sign[p] * ey, -sign[p] * ex                     # outward, unnormalised (scale cancels)
        c = nx * rx[j] + ny * ry[j]                              # support of Q along n_j
        coef = -(nx * ux + ny * uy)
        rhs = max(-(nx * ax + ny * ay), -(nx * bx + ny * by)) + c
        if coef > 0.0:
            hi = min(hi, rhs / coef)
        elif coef < 0.0:
            lo = max(lo, rhs / coef)
        elif rhs < 0.0:
            return 1.0, 0.0
        if lo > hi:
            return 1.0, 0.0
    wx, wy = ay - by, bx - ax                                    # the two normals of AB: +w and -w
    for sgn in (1.0, -1.0):
        nx, ny = sgn * wx, sgn * wy
        h = -1e300
        for j in range(rs[p], re[p] - 1):                        # support of -Q along n
            h = max(h, -(nx * rx[j] + ny * ry[j]))
        coef = nx * ux + ny * uy
        rhs = nx * ax + ny * ay + h
        if coef > 0.0:
            hi = min(hi, rhs / coef)
        elif coef < 0.0:
            lo = max(lo, rhs / coef)
        elif rhs < 0.0:
            return 1.0, 0.0
        if lo > hi:
            return 1.0, 0.0
    return lo, hi


@njit(cache=True)
def _vertex_hit_times(pv, px, py, kind, cap, rs, re, rx, ry, sign, convex, eps, seg_k, w0, w1, vel, xmin, xmax,
                      seg_sx, seg_sy, seg_ux, seg_uy, seg_len, seg_dep):
    """Per candidate (piece, vertex): status (1 hit, 0 miss, -1 concave ring: test elsewhere) and window.
    Swept pieces (kind 2): the exact window of ``_sweep_vertex_range``.
    Rings: more than ``eps`` inside, window from the reach along travel. Capsules: closer than ``r - eps``;
    on a segment the exact disc window ``s +- sqrt(r'^2 - d^2)`` cut to the body's reach along travel
    (the same for a disc agent; tighter for a polygon in its bounding disc), a turn / goal disc keeps its
    whole window."""
    n = len(pv)
    status = np.zeros(n, dtype=np.int8)
    t1 = np.empty(n)
    t2 = np.empty(n)
    for k in range(n):
        p = pv[k]
        g = seg_k[p]
        if kind[p] == 2:                                         # swept convex piece: exact window
            lo, hi = _sweep_vertex_range(px[k], py[k], seg_ux[g], seg_uy[g], seg_len[g], p, rs, re, rx, ry, sign)
            if hi - lo > 1e-12:
                status[k] = 1
                t1[k] = seg_dep[g] + lo / vel[p]
                t2[k] = seg_dep[g] + hi / vel[p]
            continue
        if kind[p] == 1:
            r = cap[p, 4] - eps
            if g >= 0:
                lo, hi = _line_disc_range(seg_sx[g], seg_sy[g], seg_ux[g], seg_uy[g], px[k], py[k], r)
                s = (px[k] - seg_sx[g]) * seg_ux[g] + (py[k] - seg_sy[g]) * seg_uy[g]
                lo = max(lo, 0.0, s - xmax[p])                   # and within the body's reach along travel
                hi = min(hi, seg_len[g], s - xmin[p])
                if hi > lo:
                    status[k] = 1
                    t1[k] = seg_dep[g] + lo / vel[p]
                    t2[k] = seg_dep[g] + hi / vel[p]
            elif _point_segment_dist2(px[k], py[k], cap[p, 0], cap[p, 1], cap[p, 2], cap[p, 3]) < r * r:
                status[k] = 1
                t1[k] = w0[p]
                t2[k] = w1[p]
            continue
        if not convex[p]:
            status[k] = -1
            continue
        inside = True
        for j in range(rs[p], re[p] - 1):
            ex = rx[j + 1] - rx[j]
            ey = ry[j + 1] - ry[j]
            if sign[p] * (ex * (py[k] - ry[j]) - ey * (px[k] - rx[j])) < eps * np.sqrt(ex * ex + ey * ey):
                inside = False
                break
        if not inside:
            continue
        status[k] = 1
        if g < 0:
            t1[k] = w0[p]
            t2[k] = w1[p]
        else:
            s = (px[k] - seg_sx[g]) * seg_ux[g] + (py[k] - seg_sy[g]) * seg_uy[g]
            t1[k], t2[k] = _segment_window(s, s, g, vel[p], xmin[p], xmax[p], seg_len, seg_dep)
    return status, t1, t2


@njit(cache=True)
def _edge_hit_times(pe, ax, ay, bx, by, kind, cap, rs, re, rx, ry, sign, convex, eps, seg_k, w0, w1, vel, xmin, xmax,
                    seg_sx, seg_sy, seg_ux, seg_uy, seg_len, seg_dep):
    """Per candidate (piece, edge a -> b): status (1 hit, 0 miss, -1 concave ring: test elsewhere) and window.
    Swept pieces (kind 2): the exact window of ``_sweep_edge_range``.
    Rings: positive length inside the ring shrunk by ``eps``, window from the reach along travel. Capsules:
    the edge comes closer than ``r - eps``; on a segment the exact disc window (the travel line clipped by
    the edge's own capsule of radius ``r - eps``) cut to the body's reach along travel, a turn / goal disc
    keeps its whole window."""
    n = len(pe)
    status = np.zeros(n, dtype=np.int8)
    t1 = np.empty(n)
    t2 = np.empty(n)
    for k in range(n):
        p = pe[k]
        g = seg_k[p]
        if kind[p] == 2:                                         # swept convex piece: exact window
            lo, hi = _sweep_edge_range(ax[k], ay[k], bx[k], by[k], seg_ux[g], seg_uy[g], seg_len[g], p,
                                       rs, re, rx, ry, sign)
            if hi - lo > 1e-12:
                status[k] = 1
                t1[k] = seg_dep[g] + lo / vel[p]
                t2[k] = seg_dep[g] + hi / vel[p]
            continue
        if kind[p] == 1:
            r = cap[p, 4] - eps
            if g >= 0:
                lo, hi = _line_capsule_range(seg_sx[g], seg_sy[g], seg_ux[g], seg_uy[g], ax[k], ay[k], bx[k], by[k], r)
                # The part of the edge the moving capsule can reach (the edge clipped by the capsule of the
                # travel segment), projected on the segment, bounds the time by the body's reach along travel.
                ex, ey = bx[k] - ax[k], by[k] - ay[k]
                el = np.sqrt(ex * ex + ey * ey)
                ex, ey = ex / el, ey / el
                gx, gy = seg_sx[g] + seg_len[g] * seg_ux[g], seg_sy[g] + seg_len[g] * seg_uy[g]
                elo, ehi = _line_capsule_range(ax[k], ay[k], ex, ey, seg_sx[g], seg_sy[g], gx, gy, r)
                elo, ehi = max(elo, 0.0), min(ehi, el)
                sa = (ax[k] + elo * ex - seg_sx[g]) * seg_ux[g] + (ay[k] + elo * ey - seg_sy[g]) * seg_uy[g]
                sb = (ax[k] + ehi * ex - seg_sx[g]) * seg_ux[g] + (ay[k] + ehi * ey - seg_sy[g]) * seg_uy[g]
                lo = max(lo, 0.0, min(sa, sb) - xmax[p])         # and within the body's reach along travel
                hi = min(hi, seg_len[g], max(sa, sb) - xmin[p])
                if hi > lo:
                    status[k] = 1
                    t1[k] = seg_dep[g] + lo / vel[p]
                    t2[k] = seg_dep[g] + hi / vel[p]
            elif _point_segment_dist2(cap[p, 0], cap[p, 1], ax[k], ay[k], bx[k], by[k]) < r * r:
                status[k] = 1
                t1[k] = w0[p]
                t2[k] = w1[p]
            continue
        if not convex[p]:
            status[k] = -1
            continue
        dx = bx[k] - ax[k]
        dy = by[k] - ay[k]
        lo = 0.0
        hi = 1.0
        for j in range(rs[p], re[p] - 1):
            ex = rx[j + 1] - rx[j]
            ey = ry[j + 1] - ry[j]
            num = sign[p] * (ex * (ay[k] - ry[j]) - ey * (ax[k] - rx[j])) - eps * np.sqrt(ex * ex + ey * ey)
            den = sign[p] * (ex * dy - ey * dx)
            if den == 0.0:
                if num < 0.0:
                    lo, hi = 1.0, 0.0
                    break
                continue
            t = -num / den
            if den > 0.0:
                if t > lo:
                    lo = t
            elif t < hi:
                hi = t
            if lo > hi:
                break
        if (hi - lo) * np.sqrt(dx * dx + dy * dy) <= 1e-12:
            continue
        status[k] = 1
        if g < 0:
            t1[k] = w0[p]
            t2[k] = w1[p]
        else:
            sa = (ax[k] + lo * dx - seg_sx[g]) * seg_ux[g] + (ay[k] + lo * dy - seg_sy[g]) * seg_uy[g]
            sb = (ax[k] + hi * dx - seg_sx[g]) * seg_ux[g] + (ay[k] + hi * dy - seg_sy[g]) * seg_uy[g]
            t1[k], t2[k] = _segment_window(min(sa, sb), max(sa, sb), g, vel[p], xmin[p], xmax[p], seg_len, seg_dep)
    return status, t1, t2

def convert_to_path_list(paths) -> list:
    """``paths`` as a list of ``(P, 2)`` arrays: one path ``(P, 2)``, same-length paths ``(A, P, 2)``,
    or a list / tuple of paths of any lengths."""
    try:
        arr = np.asarray(paths, dtype=float)
    except ValueError:                                   # ragged: several paths
        return [np.asarray(p, dtype=float) for p in paths]
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return list(arr)
    if arr.ndim == 1 and arr.size == 0:
        return []
    raise ValueError(f"paths: expected (P, 2), (A, P, 2) or a list of (P, 2) paths, got shape {arr.shape}")

def convert_to_shape_list(shapes) -> bool:
    """Whether ``shapes`` is a single shape (a ``Shape``, a radius, a shapely ``Polygon`` or one
    ``(k, 2)`` outline) rather than one per path (a list of shapes, or a 1-D list of radii)."""
    if not isinstance(shapes, (list, tuple, np.ndarray)):
        return True
    try:
        a = np.asarray(shapes, dtype=float)
    except (TypeError, ValueError):
        return False                                     # e.g. a list of Shape objects
    return a.ndim == 2 and a.shape[1] == 2

# ---------------------------------------------------------------------------
# Agent-agent conflicts: exact collision windows between moving bodies
# ---------------------------------------------------------------------------
#
# Each agent's schedule is split into motion pieces over which its body is rigid and moves at constant
# velocity: segments (moving), in-place turns (stationary; the bounding disc, since the body rotates) and
# the goal hold (stationary). For two pieces of different agents that overlap in time, the position of B
# relative to A, d(t) = p_B(t) - p_A(t), moves on a straight line, and the bodies overlap iff d(t) lies in
# A (+) (-B) (the configuration obstacle). Convex pieces of each body are tested pairwise (the union of
# the windows is exact for concave bodies); discs reduce to distance tests.

_RING_CACHE: dict = {}


def body_convex_rings(shape, eps: float = 0.0) -> list:
    """``shrunk_convex_pieces(as_footprint(shape), eps)``, cached per footprint and ``eps`` (empty for a
    disc agent)."""
    if _is_disc(shape):
        return []
    foot = as_footprint(shape)
    key = (shapely.to_wkb(foot), float(eps))
    rings = _RING_CACHE.get(key)
    if rings is None:
        if len(_RING_CACHE) > 1024:
            _RING_CACHE.clear()
        rings = _RING_CACHE[key] = shrunk_convex_pieces(foot, eps)
    return rings


def _group_exclusive_cumsum(x, group_start):
    """Exclusive cumulative sum of ``x`` restarting at every index where ``group_start`` is True."""
    c = np.cumsum(x) - x
    starts = np.flatnonzero(group_start)
    offset = np.repeat(c[starts], np.diff(np.r_[starts, len(x)]))
    return c - offset


def motion_pieces_batch(paths, shapes, velocity, t0, turning_speed, goal_hold, heading, eps: float = 0.0) -> dict:
    """Motion pieces of many agents in one vectorised pass (``paths`` a list of ``(P, 2)`` arrays; the
    other arguments one value per agent; timing as in ``path_schedule`` / ``sweep_rings_timed``).

    A piece is a time window over which the agent's body is rigid and moves at constant velocity:
    every segment (moving), every in-place turn of positive duration (stationary, as its bounding disc)
    and the goal hold when ``goal_hold > 0`` (stationary). Returns flat arrays per piece: ``agent``,
    ``w0``, ``w1``, ``x0``, ``y0`` (position at ``w0``), ``vx``, ``vy``, ``rad`` (``> 0``: a disc piece of
    that radius, reduced by ``eps / 2``; 0: the polygon body), ``rot`` (world rotation of a polygon piece),
    ``radius`` (the agent's bounding radius); and ``rings``, ``ring_lo``, ``ring_cnt``, ``ring_step``: the
    convex pieces of every polygon piece at its rotation (``body_convex_rings`` with ``eps / 2``) as flat
    closed rings ``(rs, re, rx, ry)``, ring ``k`` of piece ``q`` being ``ring_lo[q] + k * ring_step[q]``."""
    n = len(paths)
    for i in range(n):
        v, ts = velocity[i], turning_speed[i]
        if v is not None and not (np.isfinite(v) and v > 0):
            raise ValueError(f"velocity must be positive and finite, or None (1 time unit per segment), got {v}")
        if ts is not None and not ts >= 0:
            raise ValueError(f"turning_speed must be >= 0, inf or None, got {ts}")
    # Per shape object: disc flag, bounding radius, orientation, convex rings.
    uniq, shape_id = {}, np.empty(n, dtype=np.int64)
    for i, sh in enumerate(shapes):
        shape_id[i] = uniq.setdefault(id(sh), len(uniq))
    reps = list({id(sh): sh for sh in shapes}.values())
    s_disc = np.array([_is_disc(sh) for sh in reps], dtype=bool)
    s_R = np.array([_bounding_radius(sh) for sh in reps])
    s_ori = np.array([shape_orientation(sh) for sh in reps])
    s_rings = [body_convex_rings(sh, 0.5 * eps) for sh in reps]
    a_disc, a_R, a_ori = s_disc[shape_id], s_R[shape_id], s_ori[shape_id]
    vel = np.array([np.nan if v is None else float(v) for v in velocity])
    ts = np.array([np.inf if t is None else float(t) for t in turning_speed])
    hold = np.asarray(goal_hold, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    head = np.array([np.nan if h is None else float(h) for h in heading])
    fixed = np.isfinite(head) | (ts == 0)

    # Waypoints without consecutive duplicates, and the segments between them.
    pts = [np.asarray(q, dtype=float).reshape(-1, 2) for q in paths]
    P = np.concatenate(pts) if pts else np.empty((0, 2))
    pa = np.repeat(np.arange(n), [len(q) for q in pts])
    first = np.r_[True, pa[1:] != pa[:-1]] if len(pa) else np.zeros(0, bool)
    keep = first | np.r_[False, np.any(np.diff(P, axis=0) != 0.0, axis=1)] if len(pa) else first
    P, pa, first = P[keep], pa[keep], first[keep]
    is_seg = np.r_[pa[1:] == pa[:-1], False] if len(pa) else first
    k = np.flatnonzero(is_seg)                                  # segment k: P[k] -> P[k + 1]
    sa = pa[k]
    d = P[k + 1] - P[k]
    L = np.hypot(d[:, 0], d[:, 1])
    hd = np.arctan2(d[:, 1], d[:, 0])
    move = np.where(np.isnan(vel[sa]), 1.0, L / np.where(np.isnan(vel[sa]), 1.0, vel[sa]))
    seg_first = np.r_[True, sa[1:] != sa[:-1]] if len(sa) else np.zeros(0, bool)
    seg_last = np.r_[sa[1:] != sa[:-1], True] if len(sa) else np.zeros(0, bool)
    # Turn before segment k (between segment k - 1 and k of the same agent).
    ang = np.zeros(len(sa))
    if len(sa) > 1:
        ang[1:] = np.abs((hd[1:] - hd[:-1] + np.pi) % (2.0 * np.pi) - np.pi)
    ang[seg_first] = 0.0
    instant = np.isinf(ts[sa]) | fixed[sa]
    turn_before = np.where(instant, 0.0, ang / np.where(instant, 1.0, ts[sa]))
    start = t0[sa] + _group_exclusive_cumsum(move + turn_before, seg_first) + turn_before
    end = start + move
    first_hd = hd[np.flatnonzero(seg_first)]
    agent_first_hd = np.zeros(n)
    agent_first_hd[sa[seg_first]] = first_hd
    rot = np.where(np.isfinite(head[sa]), head[sa], np.where(fixed[sa], agent_first_hd[sa], hd)) + a_ori[sa]
    r_body = np.where(a_disc, a_R - 0.5 * eps, 0.0)
    cols = [(sa, start, end, P[k, 0], P[k, 1], d[:, 0] / move, d[:, 1] / move, r_body[sa], rot)]
    tq = np.flatnonzero(turn_before > 0)                       # turns of positive duration
    if len(tq):
        ta = sa[tq]
        cols.append((ta, start[tq] - turn_before[tq], start[tq], P[k[tq], 0], P[k[tq], 1], np.zeros(len(tq)),
                     np.zeros(len(tq)), a_R[ta] - 0.5 * eps, np.zeros(len(tq))))
    # Goal hold: at the last waypoint from arrival (or t0 without segments).
    arrive = t0.copy()
    last_rot = a_ori + np.where(np.isfinite(head), head, 0.0)
    if len(sa):
        arrive[sa[seg_last]] = end[seg_last]
        last_rot[sa[seg_last]] = rot[seg_last]
    goal_xy = np.zeros((n, 2))
    if len(pa):
        last_pt = np.r_[pa[1:] != pa[:-1], True]
        goal_xy[pa[last_pt]] = P[last_pt]
    has_pts = np.zeros(n, dtype=bool)
    has_pts[pa] = True
    gq = np.flatnonzero(has_pts & (hold > 0))
    if len(gq):
        cols.append((gq, arrive[gq], arrive[gq] + hold[gq], goal_xy[gq, 0], goal_xy[gq, 1], np.zeros(len(gq)),
                     np.zeros(len(gq)), r_body[gq], last_rot[gq]))
    out = {name: np.concatenate([c[i] for c in cols]) for i, name in
           enumerate(("agent", "w0", "w1", "x0", "y0", "vx", "vy", "rad", "rot"))}
    out["agent"] = out["agent"].astype(np.int64)
    out["radius"] = a_R[out["agent"]]

    # Convex rings of every polygon piece at its rotation, grouped by shape (ring-major within a group).
    m = len(out["w0"])
    ring_lo = np.full(m, -1, dtype=np.int64)
    ring_cnt = np.zeros(m, dtype=np.int64)
    ring_step = np.zeros(m, dtype=np.int64)
    xy_parts, len_parts, n_rings = [], [], 0
    poly = out["rad"] <= 0.0
    p_shape = shape_id[out["agent"]]
    for u, rings in enumerate(s_rings):
        q = np.flatnonzero(poly & (p_shape == u))
        if len(q) == 0 or not rings:
            continue
        ring_lo[q] = n_rings + np.arange(len(q))
        ring_cnt[q] = len(rings)
        ring_step[q] = len(q)
        for ring in rings:
            xy_parts.append(_place_coords(ring, out["rot"][q], np.zeros((len(q), 2))).reshape(-1, 2))
            len_parts.append(np.full(len(q), len(ring), dtype=np.int64))
        n_rings += len(q) * len(rings)
    if xy_parts:
        counts = np.concatenate(len_parts)
        re_ = np.cumsum(counts)
        xy = np.concatenate(xy_parts)
        out["rings"] = (re_ - counts, re_, np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1]))
    else:
        out["rings"] = (np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1), np.zeros(1))
    out["ring_lo"], out["ring_cnt"], out["ring_step"] = ring_lo, ring_cnt, ring_step
    return out


@njit(cache=True)
def _clip_line(lo, hi, c0, c1, h):
    """Narrow ``[lo, hi]`` to where ``c0 + c1 * tau <= h``; ``lo > hi`` when empty."""
    if c1 > 0.0:
        t = (h - c0) / c1
        if t < hi:
            hi = t
    elif c1 < 0.0:
        t = (h - c0) / c1
        if t > lo:
            lo = t
    elif c0 > h:
        return 1.0, 0.0
    return lo, hi


@njit(cache=True)
def _line_minkowski_range(dx, dy, wx, wy, T, a0, a1, b0, b1, rx, ry):
    """``tau`` in ``[0, T]`` where ``d + w * tau`` lies in ``A (+) (-B)`` (convex counter-clockwise closed
    rings ``A = coords[a0:a1]``, ``B = coords[b0:b1]``): one half-plane per side normal of either ring,
    with support ``h_A(n) + h_B(-n)``."""
    lo, hi = 0.0, T
    for side in range(2):
        s0, s1 = (a0, a1) if side == 0 else (b0, b1)
        sgn = 1.0 if side == 0 else -1.0                 # normals of -B are the negated normals of B
        for j in range(s0, s1 - 1):
            nx = sgn * (ry[j + 1] - ry[j])
            ny = -sgn * (rx[j + 1] - rx[j])
            ha = -1e300
            for i in range(a0, a1 - 1):
                ha = max(ha, nx * rx[i] + ny * ry[i])
            hb = -1e300
            for i in range(b0, b1 - 1):
                hb = max(hb, -(nx * rx[i] + ny * ry[i]))
            lo, hi = _clip_line(lo, hi, nx * dx + ny * dy, nx * wx + ny * wy, ha + hb)
            if lo > hi:
                return 1.0, 0.0
    return lo, hi


@njit(cache=True)
def _line_disc_range_general(dx, dy, wx, wy, T, r):
    """``tau`` in ``[0, T]`` where ``|d + w * tau| < r``."""
    a = wx * wx + wy * wy
    c = dx * dx + dy * dy - r * r
    if a == 0.0:
        return (0.0, T) if c < 0.0 else (1.0, 0.0)
    b = 2.0 * (dx * wx + dy * wy)
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 1.0, 0.0
    q = np.sqrt(disc)
    return max((-b - q) / (2.0 * a), 0.0), min((-b + q) / (2.0 * a), T)


@njit(cache=True)
def _line_offset_ring_range(dx, dy, wx, wy, T, a0, a1, rx, ry, r):
    """``tau`` in ``[0, T]`` where ``d + w * tau`` is within ``r`` of the convex counter-clockwise ring
    ``coords[a0:a1]``: the ring (+) disc is the ring and the capsules of its sides; the line meets that
    convex set in one interval, the hull of the parts."""
    lo, hi = 0.0, T                                      # inside the polygon
    for j in range(a0, a1 - 1):
        nx, ny = ry[j + 1] - ry[j], -(rx[j + 1] - rx[j])
        lo, hi = _clip_line(lo, hi, nx * dx + ny * dy, nx * wx + ny * wy, nx * rx[j] + ny * ry[j])
        if lo > hi:
            break
    L = np.sqrt(wx * wx + wy * wy)
    for j in range(a0, a1 - 1):                          # the capsules of the sides
        if L == 0.0:
            if _point_segment_dist2(dx, dy, rx[j], ry[j], rx[j + 1], ry[j + 1]) < r * r:
                return 0.0, T
            continue
        s0, s1 = _line_capsule_range(dx, dy, wx / L, wy / L, rx[j], ry[j], rx[j + 1], ry[j + 1], r)
        if s1 > s0:
            c0, c1 = max(s0 / L, 0.0), min(s1 / L, T)
            if c1 > c0:
                if hi > lo:
                    lo, hi = min(lo, c0), max(hi, c1)
                else:
                    lo, hi = c0, c1
    return lo, hi


@njit(cache=True, parallel=True)
def _agent_pair_windows(pa, pb, ra, rb, w0, w1, x0, y0, vx, vy, rad, rs, re, rx, ry):
    """For each candidate (piece ``pa``, piece ``pb``, ring ``ra``, ring ``rb``; ring ``-1`` = the piece's
    disc): whether the two bodies overlap for a positive time, and when. Parallel over candidates."""
    n = len(pa)
    hit = np.zeros(n, dtype=np.bool_)
    t1 = np.empty(n)
    t2 = np.empty(n)
    for k in prange(n):
        a, b = pa[k], pb[k]
        s = max(w0[a], w0[b])
        T = min(w1[a], w1[b]) - s
        if not T > 0.0:
            continue
        ax = x0[a] + vx[a] * (s - w0[a])
        ay = y0[a] + vy[a] * (s - w0[a])
        dx = x0[b] + vx[b] * (s - w0[b]) - ax
        dy = y0[b] + vy[b] * (s - w0[b]) - ay
        wx, wy = vx[b] - vx[a], vy[b] - vy[a]
        if ra[k] < 0 and rb[k] < 0:
            lo, hi = _line_disc_range_general(dx, dy, wx, wy, T, rad[a] + rad[b])
        elif ra[k] >= 0 and rb[k] >= 0:
            lo, hi = _line_minkowski_range(dx, dy, wx, wy, T, rs[ra[k]], re[ra[k]], rs[rb[k]], re[rb[k]], rx, ry)
        elif ra[k] >= 0:                                 # A polygon, B disc: d within rad[b] of A
            lo, hi = _line_offset_ring_range(dx, dy, wx, wy, T, rs[ra[k]], re[ra[k]], rx, ry, rad[b])
        else:                                            # A disc, B polygon: -d within rad[a] of B
            lo, hi = _line_offset_ring_range(-dx, -dy, -wx, -wy, T, rs[rb[k]], re[rb[k]], rx, ry, rad[a])
        if hi - lo > 1e-12:
            hit[k] = True
            t1[k] = s + lo
            t2[k] = s + hi
    return hit, t1, t2


@njit(cache=True, parallel=True)
def _sweep_prune_pairs(xmin, xmax, ymin, ymax, w0, w1, agent):
    """Broad phase: index pairs ``(a, b)`` of pieces of different agents whose boxes overlap and whose
    time windows overlap for a positive time. Sort-and-sweep on ``xmin``, parallel over pieces in two
    passes (count, then fill)."""
    n = len(xmin)
    order = np.argsort(xmin)
    sx0, sx1 = xmin[order], xmax[order]
    counts = np.zeros(n, dtype=np.int64)
    for ii in prange(n):
        i = order[ii]
        c = 0
        for jj in range(ii + 1, n):
            if sx0[jj] > sx1[ii]:
                break
            j = order[jj]
            if (agent[i] != agent[j] and ymin[j] <= ymax[i] and ymin[i] <= ymax[j]
                    and max(w0[i], w0[j]) < min(w1[i], w1[j])):
                c += 1
        counts[ii] = c
    offs = np.zeros(n + 1, dtype=np.int64)
    offs[1:] = np.cumsum(counts)
    pa = np.empty(offs[n], dtype=np.int64)
    pb = np.empty(offs[n], dtype=np.int64)
    for ii in prange(n):
        i = order[ii]
        c = offs[ii]
        for jj in range(ii + 1, n):
            if sx0[jj] > sx1[ii]:
                break
            j = order[jj]
            if (agent[i] != agent[j] and ymin[j] <= ymax[i] and ymin[i] <= ymax[j]
                    and max(w0[i], w0[j]) < min(w1[i], w1[j])):
                pa[c] = i
                pb[c] = j
                c += 1
    return pa, pb


@njit(cache=True)
def _expand_ring_pairs(qa, qb, ring_lo, ring_cnt, ring_step):
    """One candidate per pair of convex pieces of two pieces (a disc piece is one candidate with ring
    ``-1``): ``(pa, pb, ra, rb)``."""
    total = 0
    for k in range(len(qa)):
        total += max(ring_cnt[qa[k]], 1) * max(ring_cnt[qb[k]], 1)
    pa = np.empty(total, dtype=np.int64)
    pb = np.empty(total, dtype=np.int64)
    ra = np.empty(total, dtype=np.int64)
    rb = np.empty(total, dtype=np.int64)
    c = 0
    for k in range(len(qa)):
        a, b = qa[k], qb[k]
        na, nb = max(ring_cnt[a], 1), max(ring_cnt[b], 1)
        for x in range(na):
            for y in range(nb):
                pa[c] = a
                pb[c] = b
                ra[c] = ring_lo[a] + x * ring_step[a] if ring_cnt[a] > 0 else -1
                rb[c] = ring_lo[b] + y * ring_step[b] if ring_cnt[b] > 0 else -1
                c += 1
    return pa, pb, ra, rb
