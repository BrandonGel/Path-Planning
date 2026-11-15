"""Convex-polytope bodies (2D polygons, 3D polyhedra) swept along whole paths.

Geometry for ``CGALCollisionHetero``. A body is a ball or a union of convex pieces, each stored as vertices
and half-space faces ``n . x <= h`` in the body frame (orientation applied). Over each path segment the
body is rigid and moves at constant velocity, so every test is exact:

- Vertex ``p`` is inside a piece while ``n_k . (p - x0 - v t) < h_k`` for every face: one clip of the
  time window.
- Edge ``A + s (B - A)`` meets a piece while some ``s`` satisfies the same: a polygon clip in the
  ``(s, t)`` plane, whose time range is the window.
- Two bodies overlap while their relative position, which moves on a line, lies in ``A (+) (-B)``. For
  two convex pieces its half-spaces are the face normals of both and, in 3D, the cross products of their
  edge directions. For a ball and a piece it is the piece grown by the radius.

Concave bodies are unions of convex pieces; an element left and re-entered gets separate windows. During
an in-place turn the body is its bounding ball (conservative). Touching is not contact: boundary faces move
inward by ``eps`` for the roadmap tests (seams between pieces stay put), and agents must penetrate by
``eps``.

3D attitude: the travel frame has +x along the direction of travel, +y horizontal and no roll (+y is
world +y for vertical travel). The body is the shape rotated by its orientation, then by the travel frame.
"""
from __future__ import annotations

import json
import math

import numpy as np
import shapely
from numba import njit, prange
from scipy.spatial import ConvexHull
from shapely import Polygon as ShapelyPolygon

from path_planning.common.world.collision_checking.sweep.agent_shape import Shape, Shape3D, Circle, Sphere, euler_rotation
from path_planning.common.world.collision_checking.sweep.disc_sweep import (
    per_agent, as_path_list, pair_windows_output, _path_pieces, _ball_sweep_prune)


# ---------------------------------------------------------------------------
# Body templates: a ball, or convex pieces as half-spaces (body frame, orientation applied)
# ---------------------------------------------------------------------------

def _rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _footprint_convex_pieces(outline: np.ndarray) -> list:
    """Convex pieces of the 2D outline ``(K, 2)``: ``[(ring, on_boundary), ...]`` with ``ring`` ``(k, 2)``
    counter-clockwise and ``on_boundary[i]`` whether side ``i`` lies on the outline (not an internal seam)."""
    from shapely.geometry.polygon import orient
    from path_planning.common.world.collision_checking.shapely.shapely_util import convex_pieces
    foot = ShapelyPolygon(outline)
    if not foot.is_valid or foot.area <= 0.0:
        raise ValueError(f"the outline is not a valid polygon ({shapely.is_valid_reason(foot)})")
    boundary = foot.boundary
    scale = max(1.0, float(np.sqrt(foot.area)))
    out = []
    for piece in convex_pieces(foot):
        xy = np.asarray(orient(piece, 1.0).exterior.coords)[:-1]
        prev, nxt = np.roll(xy, 1, axis=0), np.roll(xy, -1, axis=0)
        turn = (xy[:, 0] - prev[:, 0]) * (nxt[:, 1] - xy[:, 1]) - (xy[:, 1] - prev[:, 1]) * (nxt[:, 0] - xy[:, 0])
        xy = xy[np.abs(turn) > 1e-12 * scale * scale]                   # drop collinear vertices
        mid = shapely.points(0.5 * (xy + np.roll(xy, -1, axis=0)))
        out.append((xy, shapely.distance(boundary, mid) <= 1e-9 * scale))
    return out


def _convex_piece(V: np.ndarray, boundary_fn) -> dict:
    """Half-space description of the convex hull of ``V`` (``(k, D)``): vertices ``V``, faces (unit normal
    ``N``, offset ``H``, boundary flag ``B`` from ``boundary_fn(normal, offset, centroid)``), per face the
    "column" planes through its ridges (``cols``), the ridges as segments ``RA -> RB`` (3D: edges; 2D:
    vertices) and, in 3D, the distinct edge directions ``E``."""
    D = V.shape[1]
    hull = ConvexHull(V)
    V = V[hull.vertices]
    remap = -np.ones(len(hull.points), dtype=np.int64)
    remap[hull.vertices] = np.arange(len(hull.vertices))
    simp = remap[hull.simplices]
    eq = hull.equations                                                 # n . x + d <= 0 inside
    scale = max(1.0, float(np.abs(V).max()))
    # Group the facets (triangles / segments) into faces by their plane.
    face_of = -np.ones(len(eq), dtype=np.int64)
    normals, offsets = [], []
    for i in range(len(eq)):
        for f, (n, h) in enumerate(zip(normals, offsets)):
            if np.abs(eq[i, :D] - n).max() < 1e-9 and abs(-eq[i, D] - h) < 1e-9 * scale:
                face_of[i] = f
                break
        else:
            face_of[i] = len(normals)
            normals.append(eq[i, :D].copy())
            offsets.append(float(-eq[i, D]))
    nf = len(normals)
    face_vertices = [np.unique(simp[face_of == f].ravel()) for f in range(nf)]
    centroids = [V[fv].mean(axis=0) for fv in face_vertices]
    # Ridges between two different faces, with the faces they bound.
    ridges = {}
    if D == 2:
        for i, s in enumerate(simp):
            for v in s:
                ridges.setdefault((int(v), int(v)), set()).add(int(face_of[i]))
    else:
        for i, s in enumerate(simp):
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
                ridges.setdefault((int(min(a, b)), int(max(a, b))), set()).add(int(face_of[i]))
        ridges = {k: f for k, f in ridges.items() if len(f) == 2}       # edges inside a face are not ridges
    cols = [[] for _ in range(nf)]
    for (a, b), faces in ridges.items():
        for f in faces:
            n = normals[f]
            if D == 2:
                other = [v for v in face_vertices[f] if v != a][0]
                m = V[a] - V[other]
            else:
                m = np.cross(V[b] - V[a], n)
                if m @ (centroids[f] - V[a]) > 0:
                    m = -m
            m = m / np.linalg.norm(m)
            cols[f].append((m, float(m @ V[a])))
    edge_dirs = []
    if D == 3:
        for (a, b) in ridges:
            e = V[b] - V[a]
            e = e / np.linalg.norm(e)
            if not any(abs(abs(e @ q) - 1.0) < 1e-12 for q in edge_dirs):
                edge_dirs.append(e)
    return {"V": V, "N": np.array(normals), "H": np.array(offsets),
            "B": np.array([bool(boundary_fn(normals[f], offsets[f], centroids[f])) for f in range(nf)]),
            "cols": cols, "RA": np.array([V[a] for a, _ in ridges]), "RB": np.array([V[b] for _, b in ridges]),
            "E": np.array(edge_dirs).reshape(-1, D)}


class Body:
    """One agent body: a ball (``ball_r > 0``) or convex ``pieces`` (``_convex_piece`` dicts, orientation
    applied). ``radius``: the bounding radius about the reference point."""
    def __init__(self, dim, radius, ball_r=0.0, pieces=()):
        self.dim, self.radius, self.ball_r, self.pieces = dim, float(radius), float(ball_r), list(pieces)


_BODY_CACHE: dict = {}


def _body_key(shape, dim):
    if isinstance(shape, Shape):
        return (type(shape).__name__, json.dumps(shape.to_dict(), sort_keys=True, default=str), dim)
    if isinstance(shape, (int, float, np.integer, np.floating)):
        return ("radius", float(shape), dim)
    if isinstance(shape, ShapelyPolygon):
        return ("polygon", shape.wkb, dim)
    return ("points", np.asarray(shape, dtype=float).tobytes(), np.shape(shape), dim)


def make_body(shape, dim: int) -> Body:
    """The ``Body`` of ``shape`` in ``dim`` dimensions (cached). A number, ``Circle`` (2D) or ``Sphere`` is a
    ball; a 2D ``Shape`` a polygon (2D) or, with ``z_height``, a prism (3D); a ``Shape3D`` a convex polyhedron.
    Raw points are used as given, about their own origin: a ``(K, 2)`` outline or shapely ``Polygon`` (2D), or
    ``(K, 3)`` points (their convex hull)."""
    key = _body_key(shape, dim)
    body = _BODY_CACHE.get(key)
    if body is None:
        if len(_BODY_CACHE) > 1024:
            _BODY_CACHE.clear()
        body = _BODY_CACHE[key] = _make_body(shape, dim)
    return body


def _all_boundary(n, h, c):
    return True


def _make_body(shape, dim):
    if isinstance(shape, (int, float, np.integer, np.floating)):
        r = float(shape)
        if not (np.isfinite(r) and r > 0):
            raise ValueError(f"a ball body needs a positive, finite radius, got {shape}")
        return Body(dim, r, ball_r=r)
    if dim not in (2, 3):
        raise ValueError(f"bodies are 2D or 3D, got dim={dim}")
    if isinstance(shape, Sphere) or (isinstance(shape, Circle) and dim == 2) or type(shape) is Shape:
        return Body(dim, shape.radius(), ball_r=shape.radius())
    if not isinstance(shape, Shape):                         # raw points: used as given, about their own origin
        if isinstance(shape, ShapelyPolygon):
            shape = np.asarray(shape.exterior.coords)[:-1]
        pts = np.asarray(shape, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != dim or len(pts) < dim + 1:
            raise ValueError(f"a {dim}D body: expected a Shape, a radius or (K, {dim}) points, got shape {pts.shape}")
        if dim == 3:
            return Body(3, np.linalg.norm(pts, axis=1).max(), pieces=[_convex_piece(pts, _all_boundary)])
        outline, zh = pts, None
    elif isinstance(shape, Shape3D):
        if dim != 3:
            raise ValueError(f"{type(shape).__name__} is a 3D shape; the roadmap / paths are {dim}D")
        V = shape.vertices() @ shape.rotation_matrix().T
        piece = _convex_piece(V, _all_boundary)
        return Body(3, np.linalg.norm(V, axis=1).max(), pieces=[piece])
    else:                                                    # a 2D Shape: a polygon in 2D, a prism in 3D
        if isinstance(shape, Circle):
            k = 32
            outline = (shape.radius() / math.cos(math.pi / k)) * np.column_stack(
                [np.cos(2 * math.pi * np.arange(k) / k), np.sin(2 * math.pi * np.arange(k) / k)])
        else:
            outline = shape.outline()
        outline = outline @ _rot2(shape.orientation()).T
        zh = shape.z_height()
    parts = _footprint_convex_pieces(outline)
    if dim == 2:
        pieces = []
        for ring, on_b in parts:
            mids = 0.5 * (ring + np.roll(ring, -1, axis=0))
            pieces.append(_convex_piece(ring, lambda n, h, c, mids=mids, on_b=on_b:
                                        bool(on_b[np.argmin(np.linalg.norm(mids - c, axis=1))])))
        return Body(2, np.linalg.norm(outline, axis=1).max(), pieces=pieces)
    if zh is None:
        raise ValueError(f"{type(shape).__name__} is a 2D outline: give it a z_height to use it in 3D "
                         f"(or use Sphere / Box / ConvexPolyhedron)")
    pieces = []
    for ring, on_b in parts:
        mids = 0.5 * (ring + np.roll(ring, -1, axis=0))
        V = np.vstack([np.column_stack([ring, np.full(len(ring), -0.5 * zh)]),
                       np.column_stack([ring, np.full(len(ring), 0.5 * zh)])])
        pieces.append(_convex_piece(V, lambda n, h, c, mids=mids, on_b=on_b:
                                    abs(n[2]) > 0.5 or bool(on_b[np.argmin(np.linalg.norm(mids - c[:2], axis=1))])))
    R2 = np.linalg.norm(outline, axis=1).max()
    return Body(3, math.hypot(R2, 0.5 * zh), pieces=pieces)


def _flatten_bodies(bodies: list, dim: int) -> tuple:
    """The convex pieces of ``bodies`` as flat arrays for the kernels: ``(body_lo, body_cnt, geo)``. Body
    ``b`` owns pieces ``body_lo[b] : body_lo[b] + body_cnt[b]`` (none for a ball); ``geo`` holds the vertex,
    face, column-plane, ridge and edge-direction arrays and each piece's ranges into them."""
    V, FN, FH, FB, FCL, FCH, CN, CH, RA, RB, E = ([] for _ in range(11))
    p_rng = []
    body_lo = np.full(len(bodies), -1, dtype=np.int64)
    body_cnt = np.zeros(len(bodies), dtype=np.int64)
    nv = nf = nc = nr = ne = npc = 0
    for b, body in enumerate(bodies):
        if body.ball_r > 0:
            continue
        body_lo[b], body_cnt[b] = npc, len(body.pieces)
        for p in body.pieces:
            p_rng.append((nv, nv + len(p["V"]), nf, nf + len(p["N"]), nr, nr + len(p["RA"]), ne, ne + len(p["E"])))
            V.append(p["V"]); FN.append(p["N"]); FH.append(p["H"]); FB.append(p["B"].astype(float))
            for cl in p["cols"]:
                FCL.append(nc); nc += len(cl); FCH.append(nc)
                CN.extend(m for m, _ in cl); CH.extend(h for _, h in cl)
            RA.append(p["RA"]); RB.append(p["RB"]); E.append(p["E"])
            nv += len(p["V"]); nf += len(p["N"]); nr += len(p["RA"]); ne += len(p["E"]); npc += 1
    cat = lambda xs, shape: np.ascontiguousarray(np.concatenate(xs)) if xs else np.zeros(shape)
    pr = np.array(p_rng, dtype=np.int64).reshape(-1, 8)
    geo = (cat(V, (1, dim)), cat(FN, (1, dim)), cat(FH, (1,)), cat(FB, (1,)),
           np.array(FCL or [0], dtype=np.int64), np.array(FCH or [0], dtype=np.int64),
           np.ascontiguousarray(np.array(CN).reshape(-1, dim)) if CN else np.zeros((1, dim)),
           np.array(CH or [0.0]), cat(RA, (1, dim)), cat(RB, (1, dim)), cat(E, (1, dim)),
           np.ascontiguousarray(pr) if len(pr) else np.zeros((1, 8), dtype=np.int64))
    return body_lo, body_cnt, geo


# ---------------------------------------------------------------------------
# Motion pieces: rigid body, constant velocity
# ---------------------------------------------------------------------------

def _travel_frames(u: np.ndarray) -> np.ndarray:
    """``(m, D, D)`` rotations whose first column is each unit direction ``u`` (3D: no roll)."""
    m, D = u.shape
    if D == 2:
        c, s = u[:, 0], u[:, 1]
        return np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)
    y = np.column_stack([-u[:, 1], u[:, 0], np.zeros(m)])
    ny = np.linalg.norm(y, axis=1)
    vertical = ny < 1e-9
    y[vertical] = [0.0, 1.0, 0.0]
    ny[vertical] = 1.0
    y /= ny[:, None]
    z = np.cross(u, y)
    return np.stack([u, y, z], axis=2)


def _heading_matrix(h, D: int) -> np.ndarray:
    """A fixed heading as a rotation: 2D an angle; 3D a yaw or a 3x3 rotation matrix."""
    a = np.asarray(h, dtype=float)
    if a.ndim == 0:
        return _rot2(float(a)) if D == 2 else euler_rotation(float(a), 0.0, 0.0)
    if D == 3 and a.shape == (3, 3):
        return a
    raise ValueError(f"heading: expected an angle{' or a 3x3 rotation' if D == 3 else ''}, got {h}")


def per_agent_heading(heading, n: int, D: int) -> list:
    """``heading`` as one value per agent (a 3x3 matrix in 3D is one heading for all)."""
    if heading is None:
        return [None] * n
    if D == 3:
        try:
            one = np.asarray(heading, dtype=float).shape == (3, 3)
        except (TypeError, ValueError):                 # a mixed per-agent list (None, angles, matrices)
            one = False
        if one:
            return [heading] * n
    return per_agent(heading, n)


def body_motion_pieces(paths, bodies, velocity, t0, turning_speed, goal_hold, heading, dim) -> dict:
    """Motion pieces of every agent: each segment (the body at the segment's attitude), each in-place turn
    (the bounding ball) and the goal hold (the body at its last attitude). Timing as
    ``disc_sweep._path_pieces``. Flat arrays per piece: ``agent``, ``w0``, ``w1``, ``x0`` (position at
    ``w0``), ``v``, ``rot``, ``body`` (-1 for a turn ball), ``ball_r`` (0 for a polytope), ``radius``."""
    n = len(paths)
    heads = per_agent_heading(heading, n, dim)
    has_head = np.array([h is not None for h in heads])
    ts = per_agent(turning_speed, n)
    ts = [None if has_head[i] else ts[i] for i in range(n)]
    seg, still = _path_pieces(paths, 1.0, velocity, t0, ts, goal_hold)
    ts_zero = np.array([t is not None and t == 0 for t in ts])
    I = np.eye(dim)
    head_R = np.stack([_heading_matrix(h, dim) if h is not None else I for h in heads]) if n else np.zeros((0, dim, dim))
    # Attitude per segment.
    sa = seg["agent"]
    d = seg["b"] - seg["a"]
    L = np.linalg.norm(d, axis=1)
    R = _travel_frames(d / np.where(L > 0, L, 1.0)[:, None]) if len(sa) else np.zeros((0, dim, dim))
    first_seg = np.full(n, -1, dtype=np.int64)
    last_seg = np.full(n, -1, dtype=np.int64)
    if len(sa):
        u, idx = np.unique(sa, return_index=True)
        first_seg[u] = idx
        last_seg[u] = np.r_[idx[1:], len(sa)] - 1
        R = np.where(ts_zero[sa][:, None, None], R[first_seg[sa]], R)
        R = np.where(has_head[sa][:, None, None], head_R[sa], R)
    b_ball = np.array([b.ball_r for b in bodies], dtype=float)
    b_rad = np.array([b.radius for b in bodies], dtype=float)
    # Stationary pieces.
    st_a, kind = still["agent"], still["kind"]
    st_R = np.repeat(I[None], len(st_a), axis=0)
    hold = kind == 1
    if hold.any():
        ha = st_a[hold]
        last = last_seg[ha]
        st_R[hold] = np.where((last >= 0)[:, None, None], R[np.maximum(last, 0)] if len(R) else head_R[ha],
                              head_R[ha])
    st_body = np.where(hold, st_a, -1)
    st_ball = np.where(hold, b_ball[st_a], b_rad[st_a])
    tdur = seg["tdur"]
    return {"agent": np.concatenate([sa, st_a]).astype(np.int64),
            "w0": np.concatenate([seg["depart"], still["w0"]]),
            "w1": np.concatenate([seg["depart"] + tdur, still["w1"]]),
            "x0": np.ascontiguousarray(np.concatenate([seg["a"], still["p"]]).reshape(-1, dim)),
            "v": np.ascontiguousarray(np.concatenate([d / tdur[:, None], np.zeros((len(st_a), dim))]).reshape(-1, dim)),
            "rot": np.ascontiguousarray(np.concatenate([R, st_R]).reshape(-1, dim, dim)),
            "body": np.concatenate([sa, st_body]).astype(np.int64),
            "ball_r": np.concatenate([b_ball[sa], st_ball]),
            "radius": np.concatenate([b_rad[sa], b_rad[st_a]])}


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _rotv(Rm, q, x, j):
    """``Rm[q] @ x[j]`` as a new array."""
    D = x.shape[1]
    out = np.empty(D)
    for k in range(D):
        s = 0.0
        for l in range(D):
            s += Rm[q, k, l] * x[j, l]
        out[k] = s
    return out


@njit(cache=True, inline="always")
def _dot(a, b):
    s = 0.0
    for k in range(a.shape[0]):
        s += a[k] * b[k]
    return s


@njit(cache=True, inline="always")
def _clip(lo, hi, a, c):
    """``[lo, hi]`` cut to ``a * t < c``."""
    if abs(a) < 1e-300:
        if c <= 0.0:
            return 1.0, 0.0
        return lo, hi
    if a > 0.0:
        hi = min(hi, c / a)
    else:
        lo = max(lo, c / a)
    return lo, hi


@njit(cache=True)
def _quad_range(A, B, C, T):
    """``t`` in ``[0, T]`` with ``A t^2 + B t + C < 0`` (``A >= 0``). Empty when ``lo >= hi``."""
    if A < 1e-300:
        if abs(B) < 1e-300:
            return (0.0, T) if C < 0.0 else (1.0, 0.0)
        return _clip(0.0, T, B, -C)
    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        return 1.0, 0.0
    q = math.sqrt(disc)
    return max(0.0, (-B - q) / (2.0 * A)), min(T, (-B + q) / (2.0 * A))


@njit(cache=True)
def _capsule_range(d0, w, T, a, b, r):
    """``t`` in ``[0, T]`` with ``d0 + w t`` strictly within ``r`` of segment ``a -> b`` (a ball when
    ``a == b``): the union of the end balls and the finite cylinder."""
    D = d0.shape[0]
    lo, hi = 1.0, 0.0
    ww = _dot(w, w)
    for end in range(2):
        c = a if end == 0 else b
        q = d0 - c
        l, h = _quad_range(ww, 2.0 * _dot(w, q), _dot(q, q) - r * r, T)
        if l < h:
            lo, hi = (l, h) if lo >= hi else (min(lo, l), max(hi, h))
    e = b - a
    L2 = _dot(e, e)
    if L2 > 1e-300:
        q = d0 - a
        qe, we = _dot(q, e) / L2, _dot(w, e) / L2
        qp = q - qe * e
        wp = w - we * e
        l, h = _quad_range(_dot(wp, wp), 2.0 * _dot(wp, qp), _dot(qp, qp) - r * r, T)
        l, h = _clip(l, h, -we, qe)              # s(t) = qe + we t > 0
        l, h = _clip(l, h, we, 1.0 - qe)         # s(t) < 1
        if l < h:
            lo, hi = (l, h) if lo >= hi else (min(lo, l), max(hi, h))
    return lo, hi


@njit(cache=True)
def _piece_range(d0, w, T, Rm, q, c, V, FN, FH, FB, eps, pr):
    """``t`` in ``[0, T]`` with ``d0 + w t`` strictly inside convex piece ``c`` rotated by ``Rm[q]``
    (boundary faces moved in by ``eps``)."""
    lo, hi = 0.0, T
    for f in range(pr[c, 2], pr[c, 3]):
        n = _rotv(Rm, q, FN, f)
        lo, hi = _clip(lo, hi, _dot(n, w), FH[f] - eps * FB[f] - _dot(n, d0))
        if lo >= hi:
            break
    return lo, hi


@njit(cache=True, parallel=True)
def _vertex_windows(pq, pc, vid, P, x0, v, T, Rm, ball_r, V, FN, FH, FB, FCL, FCH, CN, CH, RA, RB, E, pr, eps):
    """Row ``k``: when roadmap vertex ``vid[k]`` is strictly inside motion piece ``pq[k]`` (convex piece
    ``pc[k]``, or its ball when ``pc[k] < 0``). Times relative to the piece start."""
    m = len(pq)
    t1 = np.empty(m)
    t2 = np.empty(m)
    hit = np.zeros(m, dtype=np.bool_)
    for k in prange(m):
        q = pq[k]
        d0 = P[vid[k]] - x0[q]
        w = -v[q]
        if pc[k] < 0:
            lo, hi = _capsule_range(-d0, v[q], T[q], np.zeros_like(d0), np.zeros_like(d0), ball_r[q] - eps)
        else:
            lo, hi = _piece_range(d0, w, T[q], Rm, q, pc[k], V, FN, FH, FB, eps, pr)
        if hi - lo > 1e-12:
            hit[k] = True
            t1[k], t2[k] = lo, hi
    return hit, t1, t2


@njit(cache=True)
def _clip_polygon(px, py, n, a, b, c, qx, qy):
    """Polygon ``(px, py)[:n]`` cut to ``a x + b y <= c`` into ``(qx, qy)``; returns the vertex count."""
    m = 0
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        di = a * px[i] + b * py[i] - c
        dj = a * px[j] + b * py[j] - c
        if di <= 0.0:
            qx[m], qy[m] = px[i], py[i]
            m += 1
        if (di < 0.0 < dj) or (dj < 0.0 < di):
            f = di / (di - dj)
            qx[m], qy[m] = px[i] + f * (px[j] - px[i]), py[i] + f * (py[j] - py[i])
            m += 1
    return m


@njit(cache=True, parallel=True)
def _edge_windows(pq, pc, eid, EA, EB, x0, v, T, Rm, ball_r, V, FN, FH, FB, FCL, FCH, CN, CH, RA, RB, E, pr, eps):
    """Row ``k``: when a positive length of roadmap edge ``eid[k]`` is inside motion piece ``pq[k]`` (convex
    piece ``pc[k]``, or its ball when ``pc[k] < 0``). Times relative to the piece start."""
    m = len(pq)
    t1 = np.empty(m)
    t2 = np.empty(m)
    hit = np.zeros(m, dtype=np.bool_)
    for k in prange(m):
        q, e, c = pq[k], eid[k], pc[k]
        A, B = EA[e], EB[e]
        if c < 0:
            lo, hi = _capsule_range(x0[q], v[q], T[q], A, B, ball_r[q] - eps)
            if hi - lo > 1e-12:
                hit[k] = True
                t1[k], t2[k] = lo, hi
            continue
        rel = A - x0[q]
        ab = B - A
        seg_len = math.sqrt(_dot(ab, ab))
        moving = _dot(v[q], v[q]) > 0.0
        if not moving:                          # the s-range at any time
            lo, hi = 0.0, 1.0
            for f in range(pr[c, 2], pr[c, 3]):
                n = _rotv(Rm, q, FN, f)
                lo, hi = _clip(lo, hi, _dot(n, ab), FH[f] - eps * FB[f] - _dot(n, rel))
                if lo >= hi:
                    break
            if (hi - lo) * seg_len > 1e-12:
                hit[k] = True
                t1[k], t2[k] = 0.0, T[q]
            continue
        nf = pr[c, 3] - pr[c, 2]
        cap = 4 + nf + 2
        px = np.empty(cap); py = np.empty(cap); qx = np.empty(cap); qy = np.empty(cap)
        px[0], py[0], px[1], py[1], px[2], py[2], px[3], py[3] = 0.0, 0.0, 1.0, 0.0, 1.0, T[q], 0.0, T[q]
        cnt = 4
        for f in range(pr[c, 2], pr[c, 3]):    # n . (rel + s ab - t v) <= h
            n = _rotv(Rm, q, FN, f)
            cnt = _clip_polygon(px, py, cnt, _dot(n, ab), -_dot(n, v[q]), FH[f] - eps * FB[f] - _dot(n, rel), qx, qy)
            px, qx = qx, px
            py, qy = qy, py
            if cnt < 3:
                break
        if cnt >= 3:
            smin, smax, tmin, tmax = px[0], px[0], py[0], py[0]
            for i in range(1, cnt):
                smin = min(smin, px[i]); smax = max(smax, px[i])
                tmin = min(tmin, py[i]); tmax = max(tmax, py[i])
            if tmax - tmin > 1e-12 and (smax - smin) * seg_len > 1e-12:
                hit[k] = True
                t1[k], t2[k] = tmin, tmax
    return hit, t1, t2


@njit(cache=True)
def _offset_piece_range(d0, w, T, Rm, q, c, r, V, FN, FH, FCL, FCH, CN, CH, RA, RB, pr):
    """``t`` in ``[0, T]`` with ``d0 + w t`` strictly within ``r`` of convex piece ``c`` rotated by
    ``Rm[q]``: the piece, a slab over each face (bounded by its column planes) and a capsule along each
    ridge, which together are the grown piece."""
    lo, hi = _piece_range(d0, w, T, Rm, q, c, V, FN, FH, FH, 0.0, pr)
    for f in range(pr[c, 2], pr[c, 3]):
        n = _rotv(Rm, q, FN, f)
        nd, nw = _dot(n, d0), _dot(n, w)
        l, h = _clip(0.0, T, nw, FH[f] + r - nd)
        l, h = _clip(l, h, -nw, nd - FH[f])
        for j in range(FCL[f], FCH[f]):
            if l >= h:
                break
            mm = _rotv(Rm, q, CN, j)
            l, h = _clip(l, h, _dot(mm, w), CH[j] - _dot(mm, d0))
        if l < h:
            lo, hi = (l, h) if lo >= hi else (min(lo, l), max(hi, h))
    for j in range(pr[c, 4], pr[c, 5]):
        a = _rotv(Rm, q, RA, j)
        b = _rotv(Rm, q, RB, j)
        l, h = _capsule_range(d0, w, T, a, b, r)
        if l < h:
            lo, hi = (l, h) if lo >= hi else (min(lo, l), max(hi, h))
    return lo, hi


@njit(cache=True)
def _minkowski_range(d0, w, T, Rm, qa, ca, qb, cb, V, FN, E, pr, eps):
    """``t`` in ``[0, T]`` with ``d0 + w t`` in ``A (+) (-B)`` shrunk by ``eps``, for convex pieces ``ca``
    of motion piece ``qa`` and ``cb`` of ``qb``: one half-space per face normal of either and, in 3D, per
    cross product of their edge directions."""
    D = d0.shape[0]
    va0, va1, vb0, vb1 = pr[ca, 0], pr[ca, 1], pr[cb, 0], pr[cb, 1]
    VA = np.empty((va1 - va0, D))
    VB = np.empty((vb1 - vb0, D))
    for i in range(va0, va1):
        VA[i - va0] = _rotv(Rm, qa, V, i)
    for i in range(vb0, vb1):
        VB[i - vb0] = _rotv(Rm, qb, V, i)
    lo, hi = 0.0, T
    nfa = pr[ca, 3] - pr[ca, 2]
    nfb = pr[cb, 3] - pr[cb, 2]
    nea = pr[ca, 7] - pr[ca, 6]
    neb = pr[cb, 7] - pr[cb, 6]
    total = nfa + nfb + (2 * nea * neb if D == 3 else 0)
    n = np.empty(D)
    for k in range(total):
        if k < nfa:
            n = _rotv(Rm, qa, FN, pr[ca, 2] + k)
        elif k < nfa + nfb:
            n = -_rotv(Rm, qb, FN, pr[cb, 2] + k - nfa)
        else:
            j = k - nfa - nfb
            ea = _rotv(Rm, qa, E, pr[ca, 6] + (j // 2) // neb)
            eb = _rotv(Rm, qb, E, pr[cb, 6] + (j // 2) % neb)
            n[0] = ea[1] * eb[2] - ea[2] * eb[1]
            n[1] = ea[2] * eb[0] - ea[0] * eb[2]
            n[2] = ea[0] * eb[1] - ea[1] * eb[0]
            nn = math.sqrt(_dot(n, n))
            if nn < 1e-9:
                continue
            n /= nn if j % 2 == 0 else -nn
        ha = -1e300
        for i in range(VA.shape[0]):
            ha = max(ha, _dot(n, VA[i]))
        hb = 1e300
        for i in range(VB.shape[0]):
            hb = min(hb, _dot(n, VB[i]))
        lo, hi = _clip(lo, hi, _dot(n, w), ha - hb - eps - _dot(n, d0))
        if lo >= hi:
            break
    return lo, hi


@njit(cache=True, parallel=True)
def _body_pair_windows(qa, qb, ca, cb, w0, w1, x0, v, Rm, ball_r,
                       V, FN, FH, FB, FCL, FCH, CN, CH, RA, RB, E, pr, eps):
    """Row ``k``: when motion pieces ``qa[k]`` and ``qb[k]`` (convex pieces ``ca[k]`` / ``cb[k]``, a ball
    when negative) overlap by more than ``eps``. Absolute times."""
    m = len(qa)
    t1 = np.empty(m)
    t2 = np.empty(m)
    hit = np.zeros(m, dtype=np.bool_)
    for k in prange(m):
        a, b = qa[k], qb[k]
        s = max(w0[a], w0[b])
        e = min(w1[a], w1[b])
        if not e > s:
            continue
        T = e - s
        pa = x0[a] + v[a] * (s - w0[a])
        pb = x0[b] + v[b] * (s - w0[b])
        d0 = pb - pa
        w = v[b] - v[a]
        z = np.zeros_like(d0)
        if ca[k] < 0 and cb[k] < 0:
            lo, hi = _capsule_range(d0, w, T, z, z, ball_r[a] + ball_r[b] - eps)
        elif cb[k] < 0:
            lo, hi = _offset_piece_range(d0, w, T, Rm, a, ca[k], ball_r[b] - eps, V, FN, FH, FCL, FCH, CN, CH,
                                         RA, RB, pr)
        elif ca[k] < 0:
            lo, hi = _offset_piece_range(-d0, -w, T, Rm, b, cb[k], ball_r[a] - eps, V, FN, FH, FCL, FCH, CN, CH,
                                         RA, RB, pr)
        else:
            lo, hi = _minkowski_range(d0, w, T, Rm, a, ca[k], b, cb[k], V, FN, E, pr, eps)
        if hi - lo > 1e-12:
            hit[k] = True
            t1[k], t2[k] = s + lo, s + hi
    return hit, t1, t2


def _expand_pieces(rows, sub_lo, sub_cnt):
    """Each motion piece ``rows[k]`` once per convex piece of its body: ``(index into rows, piece)``,
    with piece -1 for a ball."""
    cnt = np.maximum(sub_cnt[rows], 1)
    idx = np.repeat(np.arange(len(rows)), cnt)
    start = np.cumsum(cnt) - cnt
    offs = np.arange(len(idx)) - np.repeat(start, cnt)
    sub = np.where(sub_cnt[rows][idx] > 0, sub_lo[rows][idx] + offs, -1)
    return idx, sub.astype(np.int64)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _prepare(paths, shapes, velocity, t0, turning_speed, goal_hold, heading, dim):
    """Bodies, motion pieces and flat piece geometry of all agents."""
    n = len(paths)
    single = not isinstance(shapes, (list, tuple, np.ndarray))
    if not single:
        try:
            a = np.asarray(shapes, dtype=float)
            single = a.ndim == 2 and a.shape[1] == dim
        except (TypeError, ValueError):
            single = False
    if not single and len(shapes) != n:
        raise ValueError(f"shapes: expected one shape or one per path ({n}), got {len(shapes)}")
    shape_list = [shapes] * n if single else list(shapes)
    bodies = [make_body(s, dim) for s in shape_list]
    mp = body_motion_pieces(paths, bodies, velocity, t0, turning_speed, goal_hold, heading, dim)
    body_lo, body_cnt, geo = _flatten_bodies(bodies, dim)
    has_body = mp["body"] >= 0
    sub_lo = np.where(has_body, body_lo[np.maximum(mp["body"], 0)], -1)
    sub_cnt = np.where(has_body, body_cnt[np.maximum(mp["body"], 0)], 0)
    return mp, sub_lo.astype(np.int64), sub_cnt.astype(np.int64), geo


def body_roadmap_hits(backend, paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                      heading=None, eps: float = 1e-9):
    """Hits of every agent's body against ``backend``'s roadmap, as ``(v_parts, e_parts)`` lists of
    ``(agent, element, t1, t2)`` arrays for ``disc_sweep.assemble``. ``backend`` as in
    ``disc_sweep.sweep_paths``."""
    dim = backend.vertex_positions.shape[1]
    paths = [np.asarray(p, dtype=float).reshape(-1, dim) for p in as_path_list(paths)]
    mp, sub_lo, sub_cnt, geo = _prepare(paths, shapes, velocity, t0, turning_speed, goal_hold, heading, dim)
    if len(mp["w0"]) == 0:
        return [], []
    T = mp["w1"] - mp["w0"]
    moving = np.any(mp["v"] != 0.0, axis=1)
    end = mp["x0"] + np.where(moving, T, 0.0)[:, None] * mp["v"]
    v_parts, e_parts = [], []
    for kind in ("vertex", "edge"):
        if kind == "vertex":
            rows, el = backend._vertex_candidates(mp["x0"], end, mp["radius"])
        else:
            rows, el = backend._edge_candidates(mp["x0"], end, mp["radius"])
        if len(rows) == 0:
            continue
        idx, sub = _expand_pieces(rows, sub_lo, sub_cnt)
        pq, el = rows[idx], el[idx]
        if kind == "vertex":
            hit, t1, t2 = _vertex_windows(pq, sub, el, backend.vertex_positions, mp["x0"], mp["v"], T, mp["rot"],
                                          mp["ball_r"], *geo, eps)
            v_parts.append((mp["agent"][pq[hit]], el[hit], mp["w0"][pq[hit]] + t1[hit], mp["w0"][pq[hit]] + t2[hit]))
        else:
            hit, t1, t2 = _edge_windows(pq, sub, el, backend.edge_src_positions, backend.edge_tgt_positions,
                                        mp["x0"], mp["v"], T, mp["rot"], mp["ball_r"], *geo, eps)
            e_parts.append((mp["agent"][pq[hit]], el[hit], mp["w0"][pq[hit]] + t1[hit], mp["w0"][pq[hit]] + t2[hit]))
    return v_parts, e_parts


def body_agent_conflicts(paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0, heading=None,
                         pairs=None, first_only: bool = False, as_arrays: bool = False, eps: float = 1e-9):
    """``CGALCollisionHetero.agent_conflicts``: when two agents' bodies overlap (2D or 3D)."""
    paths = as_path_list(paths)
    n = len(paths)
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), np.empty(0))
    if n == 0:
        return empty if as_arrays else {}
    dim = np.asarray(paths[0]).shape[-1]
    paths = [np.asarray(p, dtype=float).reshape(-1, dim) for p in paths]
    mp, sub_lo, sub_cnt, geo = _prepare(paths, shapes, velocity, t0, turning_speed, goal_hold, heading, dim)
    if len(mp["w0"]) == 0:
        return empty if as_arrays else {}
    w0, w1, x0, v, R = mp["w0"], mp["w1"], mp["x0"], mp["v"], mp["radius"]
    end_t = np.where(np.isfinite(w1), w1, w0)
    xe = x0 + (end_t - w0)[:, None] * v
    qa, qb = _ball_sweep_prune(np.minimum(x0, xe) - R[:, None], np.maximum(x0, xe) + R[:, None], w0, w1,
                               mp["agent"])
    if pairs is not None:
        allowed = np.zeros((n, n), dtype=bool)
        for i, j in pairs:
            allowed[i, j] = allowed[j, i] = True
        ok = allowed[mp["agent"][qa], mp["agent"][qb]]
        qa, qb = qa[ok], qb[ok]
    if len(qa) == 0:
        return empty if as_arrays else {}
    # One row per pair of convex pieces.
    ia, ca = _expand_pieces(qa, sub_lo, sub_cnt)
    qa2, qb2 = qa[ia], qb[ia]
    ib, cb = _expand_pieces(qb2, sub_lo, sub_cnt)
    qa3, ca3, qb3 = qa2[ib], ca[ib], qb2[ib]
    hit, t1, t2 = _body_pair_windows(qa3, qb3, ca3, cb, w0, w1, x0, v, mp["rot"], mp["ball_r"], *geo, eps)
    return pair_windows_output(n, mp["agent"][qa3[hit]], mp["agent"][qb3[hit]], t1[hit], t2[hit],
                               first_only, as_arrays)
