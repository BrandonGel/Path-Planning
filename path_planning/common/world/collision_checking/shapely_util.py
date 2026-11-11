import numpy as np
from shapely import Polygon, Point
from path_planning.common.world.collision_checking.agent_shape import Shape, Circle
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


def _place_coords(xy, rot, origin):
    """Body-frame ``xy`` ``(K, 2)`` rotated by each ``rot`` ``(m,)`` and moved to each ``origin`` ``(m, 2)``."""
    c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
    x = c * xy[None, :, 0] - s * xy[None, :, 1] + origin[:, 0:1]
    y = s * xy[None, :, 0] + c * xy[None, :, 1] + origin[:, 1:2]
    return np.stack([x, y], axis=-1)                                   # (m, K, 2)


def _exact_segment_sweeps(footprint: Polygon, orientation: float, path: np.ndarray) -> list:
    """``footprint (+) segment`` for every segment, vectorised over segments: the footprint's convex
    pieces are found once, then hull(T, T + d) is built for all segments in one call per piece."""
    seg = path[1:] - path[:-1]
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

def path_schedule(path, velocity: float, t0: float = 0.0, turning_speed=None):
    """Timing of an agent along the polyline ``path``: constant ``velocity`` on every segment, turning in
    place at each interior waypoint at ``turning_speed`` rad/s (``None`` or ``inf``: instant turns).

    Returns ``(path, seg_windows, turn_windows, headings)``: the path with consecutive duplicate waypoints
    dropped, ``(m, 2)`` ``(depart, arrive)`` per segment, ``(P - 2, 2)`` ``(start, end)`` per interior
    turn, and the ``(m,)`` direction of travel of each segment."""
    if velocity is None or not np.isfinite(velocity) or velocity <= 0:
        raise ValueError(f"path_schedule: velocity must be positive and finite, got {velocity}")
    if turning_speed is not None and not turning_speed > 0:
        raise ValueError(f"path_schedule: turning_speed must be positive (or None), got {turning_speed}")
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) > 1:
        path = path[np.r_[True, np.any(np.diff(path, axis=0) != 0.0, axis=1)]]
    seg = path[1:] - path[:-1]
    headings = np.arctan2(seg[:, 1], seg[:, 0])
    move = np.hypot(seg[:, 0], seg[:, 1]) / velocity
    angle = np.abs((np.diff(headings) + np.pi) % (2.0 * np.pi) - np.pi)
    turn = angle * 0.0 if turning_speed is None or np.isinf(turning_speed) else angle / turning_speed
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
    """``{key: [(t1, t2), ...]}`` from ``(key, t1, t2)`` items: each key's closed windows sorted and merged
    where they overlap or touch."""
    groups = {}
    for k, a, b in items:
        groups.setdefault(k, []).append((float(a), float(b)))
    for k, ivs in groups.items():
        ivs.sort()
        merged = [ivs[0]]
        for a, b in ivs[1:]:
            if a <= merged[-1][1] + 1e-12:
                if b > merged[-1][1]:
                    merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        groups[k] = merged
    return groups
