"""2D footprint geometry: swept regions (for figures) and the fast front-face sweep of
``ShapelyCollisionHetero`` (timed pieces and the numba kernels that test them against roadmap vertices and
edges). Exact sweeps and agent conflicts are in ``polytope_sweep``."""
import numpy as np
import shapely
from numba import njit
from shapely import Polygon, Point
from shapely import affinity as _affinity

from path_planning.common.world.collision_checking.sweep.agent_shape import Shape, Circle


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------

def as_footprint(shape) -> Polygon:
    """``shape`` as a body-frame ``Polygon`` at orientation 0, centroid at the origin. Accepts a ``Shape``
    (a ``Circle`` becomes a 128-gon), a radius, a ``(K, 2)`` outline or a shapely ``Polygon``."""
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
    """The body's fixed rotation relative to the direction of travel (radians); 0 unless a ``Shape``."""
    return float(shape.orientation()) if isinstance(shape, Shape) else 0.0


def place_footprint(footprint: Polygon, pos, heading: float) -> Polygon:
    """The footprint in the world frame: rotated by ``heading`` (direction of travel + the shape's
    orientation, radians) about the body origin, then moved to ``pos``."""
    poly = _affinity.rotate(footprint, heading, origin=(0.0, 0.0), use_radians=True) if heading != 0.0 else footprint
    return _affinity.translate(poly, float(pos[0]), float(pos[1]))


def convex_pieces(poly: Polygon) -> list:
    """``poly`` itself when convex, else the triangles of its constrained Delaunay triangulation."""
    if abs(poly.convex_hull.area - poly.area) <= 1e-9 * max(1.0, poly.area):
        return [poly]
    return list(shapely.constrained_delaunay_triangles(poly).geoms)


# ---------------------------------------------------------------------------
# Swept region of an agent along a path
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
    """``(m, 4)`` ``(x_min, x_max, y_min, y_max)`` of ``footprint`` rotated by each ``rel`` (rotation
    relative to the direction of travel), in the travel frame."""
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
    """``footprint (+) segment`` for every segment: hull(T, T + d) per convex piece, vectorised over
    segments. The body is rotated by the direction of travel + ``orientation``, or by ``rot`` (``(m,)``)."""
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
    """One rectangle per segment: the body's bounding box along the direction of travel, from its rear
    at the segment start to its front at the segment end. Exact for a travel-aligned rectangle, a superset
    otherwise."""
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
    """``LineString(path).buffer(radius)``, with the radius circumscribed so the GEOS chords never sag
    inside the true disc (2% larger at ``quad_segs = 8``)."""
    geom = shapely.LineString(path) if len(path) > 1 else Point(path[0])
    
    r = radius / np.cos(np.pi / (2 * quad_segs))
    return [geom.buffer(r, quad_segs=quad_segs)]


def _is_disc(shape) -> bool:
    return isinstance(shape, Circle) or isinstance(shape, (int, float, np.integer, np.floating))


def _capsules_segment_sweeps(path: np.ndarray, radius: float, quad_segs: int) -> list:
    """One capsule per segment (``segment.buffer(radius)``, the exact disc sweep), with the radius
    circumscribed as in ``_buffer_segment_sweeps``."""
    r = radius / np.cos(np.pi / (2 * quad_segs))
    if len(path) == 1:
        return [Point(path[0]).buffer(r, quad_segs=quad_segs)]
    lines = shapely.linestrings(np.stack([path[:-1], path[1:]], axis=1))
    return list(shapely.buffer(lines, r, quad_segs=quad_segs))

def _turn_discs_sweeps(path: np.ndarray, radius: float, quad_segs: int) -> list:
    """The bounding disc at every interior waypoint (circumscribed polygon): it covers the turn in place."""
    if len(path) <= 2:
        return []
    r = radius / np.cos(np.pi / (4 * quad_segs))
    return list(shapely.buffer(shapely.points(path[1:-1]), r, quad_segs=quad_segs))

def sweep_region(path, shape, method: str = "front_face", quad_segs: int = 8) -> list:
    """Region covered by ``shape`` moving along ``path`` (``(P, 2)``), facing the direction of travel and
    turning in place at the waypoints. Returns a list of polygons whose union is the region.

    ``method``:

    - ``"buffer"``: the bounding disc swept along the path, one piece (a loose superset).
    - ``"exact"``: the exact footprint sweep on every segment plus the bounding disc at every turn.
    - ``"front_face"``: the body's bounding box along travel on every segment plus the bounding disc at
      every turn. Cheapest; exact for travel-aligned rectangles.

    A disc agent (``Circle`` or a radius) gets one capsule per segment whatever the method. ``quad_segs``
    sets the disc resolution (``4 * quad_segs`` sides)."""
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
# Path timing
# ---------------------------------------------------------------------------

def fixed_orientation(turning_speed) -> bool:
    """Whether ``turning_speed`` means the body never turns (``0``)."""
    return turning_speed is not None and turning_speed == 0


def path_schedule(path, velocity: float, t0: float = 0.0, turning_speed=None):
    """Timing of an agent along ``path``: ``(path, seg_windows, turn_windows, headings)``, i.e. the path
    without consecutive duplicates, ``(depart, arrive)`` per segment, ``(start, end)`` per interior turn and
    each segment's direction of travel.

    ``velocity``: speed; ``None`` means every segment takes 1 time unit. ``turning_speed`` (rad/s): a turn
    takes ``angle / turning_speed``; ``None``, ``inf`` and ``0`` take no time (``0`` also means the body
    never turns, see ``sweep_rings_timed``)."""
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


# ---------------------------------------------------------------------------
# Timed swept pieces
# ---------------------------------------------------------------------------

def _is_convex(pieces) -> np.ndarray:
    """Whether each polygon equals its convex hull (relative 1e-9 in area)."""
    area = shapely.area(pieces)
    return np.abs(shapely.area(shapely.convex_hull(pieces)) - area) <= 1e-9 * np.maximum(area, 1.0)


def piece_rings(pieces, known_convex=None) -> tuple:
    """Flat ring arrays of ``pieces`` for the numba kernels: ``(ring_start, ring_end, rx, ry, sign,
    convex)``. ``sign`` is +1 for a counter-clockwise ring. ``known_convex`` skips the hull check."""
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


PIECE_RING, PIECE_CAPSULE = 0, 1          # piece kinds: polygon ring; capsule (a disc when a == b)


def sweep_rings_timed(path, shape, velocity: float, t0: float = 0.0, turning_speed=None, goal_hold: float = 0.0,
                      heading=None) -> dict:
    """The front-face swept pieces of ``shape`` along ``path`` with their time windows, as flat arrays for
    the numba kernels. Piece kinds: polygon ring and capsule ``(ax, ay, bx, by, r)`` (a disc when a == b).

    - Disc agent: one capsule per segment, a disc per turn and for the goal hold.
    - Polygon agent: per segment, the body's bounding box along travel (a rectangle); per turn, a disc of
      the bounding radius; for the goal hold, the footprint.

    Timing as in ``path_schedule``. With ``turning_speed = 0`` the body keeps its first segment's rotation
    and has no turn discs; ``heading`` (radians) fixes the rotation to ``heading`` + its orientation.

    Returns a dict: ``path``, ``seg_windows``, ``seg_speed`` per segment; per piece ``windows``,
    ``segment`` (-1 for turn / goal pieces), ``kind``, ``cap`` (``(n, 5)`` capsule parameters), ``reach``
    (``(x_min, x_max)`` of the body along travel), ``rings`` (as ``piece_rings``), ``bounds`` and
    ``geoms`` (the polygon, only for a concave goal footprint)."""
    p, seg_w, turn_w, _ = path_schedule(path, velocity, t0, 0 if heading is not None else turning_speed)
    coords, counts, windows, segment, kind, cap, convex, geoms = [], [], [], [], [], [], [], []

    def add_rings(c, cnt, w, seg, cvx=True, g=None):
        coords.append(np.asarray(c, dtype=float).reshape(-1, 2)); counts.append(np.asarray(cnt, dtype=np.int64))
        windows.append(np.asarray(w, dtype=float).reshape(-1, 2)); segment.append(np.asarray(seg, dtype=np.int64))
        kind.append(np.full(len(cnt), PIECE_RING, np.int8)); cap.append(np.zeros((len(cnt), 5)))
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
    if m:
        if disc:
            add_caps(p[:-1], p[1:], radius, seg_w, np.arange(m))
        else:                                                    # front-face rectangles (as _front_face_segment_sweeps)
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
    """Whether each point is inside its convex ring by at least ``eps`` from every side."""
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
    """Whether each point ``xy[i]`` is inside ``pieces[piece_idx[i]]`` by more than ``eps``. Convex pieces
    in numba, concave ones through Shapely."""
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
    """Parameter range ``[lo, hi]`` of each segment ``a -> b`` inside its convex ring shrunk by ``eps``
    (``lo > hi`` when it misses)."""
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
    """``(lo, hi)``: the range of ``t`` in ``[0, 1]`` where ``a[i] + t * (b[i] - a[i])`` is inside
    ``pieces[piece_idx[i]]`` shrunk by ``eps`` (``lo > hi`` when it misses). Convex pieces in numba,
    concave ones through Shapely. ``rings``: ``piece_rings(pieces)`` if already computed."""
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
# Roadmap hit kernels: inside test / clipping and time window in one pass
# ---------------------------------------------------------------------------

@njit(cache=True)
def _segment_window(s_lo, s_hi, g, vel, xmin, xmax, seg_len, seg_dep):
    """The times the body's reach ``[xmin, xmax]`` along segment ``g`` spans ``[s_lo, s_hi]``."""
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
    """Where ``S + t*u`` (``u`` unit) is within ``r`` of segment ``AB``: the union of the end discs and
    the band along ``AB``; ``lo > hi`` when never."""
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
def _vertex_hit_times(pv, px, py, kind, cap, rs, re, rx, ry, sign, convex, eps, seg_k, w0, w1, vel, xmin, xmax,
                      seg_sx, seg_sy, seg_ux, seg_uy, seg_len, seg_dep):
    """Per candidate (piece, vertex): status (1 hit, 0 miss, -1 concave ring: test elsewhere) and window.

    - Ring: more than ``eps`` inside; window from the reach along travel.
    - Capsule: closer than ``r - eps``. On a segment, the exact disc window cut to the body's reach along
      travel; a turn / goal disc keeps its whole window."""
    n = len(pv)
    status = np.zeros(n, dtype=np.int8)
    t1 = np.empty(n)
    t2 = np.empty(n)
    for k in range(n):
        p = pv[k]
        g = seg_k[p]
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
    """Per candidate (piece, edge a -> b): status (1 hit, 0 miss, -1 concave ring: test elsewhere) and
    window.

    - Ring: a positive length inside the ring shrunk by ``eps``; window from the reach along travel.
    - Capsule: the edge comes closer than ``r - eps``. On a segment, the exact disc window cut to the
      body's reach along travel; a turn / goal disc keeps its whole window."""
    n = len(pe)
    status = np.zeros(n, dtype=np.int8)
    t1 = np.empty(n)
    t2 = np.empty(n)
    for k in range(n):
        p = pe[k]
        g = seg_k[p]
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

def convert_to_shape_list(shapes) -> bool:
    """Whether ``shapes`` is a single shape (a ``Shape``, a radius, a ``Polygon`` or one ``(K, 2)``
    outline) rather than one per path."""
    if not isinstance(shapes, (list, tuple, np.ndarray)):
        return True
    try:
        a = np.asarray(shapes, dtype=float)
    except (TypeError, ValueError):
        return False                                     # e.g. a list of Shape objects
    return a.ndim == 2 and a.shape[1] == 2
