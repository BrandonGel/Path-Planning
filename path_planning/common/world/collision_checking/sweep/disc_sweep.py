"""Ball agents (discs in 2D, spheres in 3D) swept along whole paths, in any dimension.

The geometry behind ``collision_base.CollisionBase`` (the base of ``ShapelyCollision`` and
``CGALCollision``). A backend only supplies candidate queries (a superset of the elements near each
segment); the exact distance test, timing, assembly and agent conflicts are here:

- ``sweep_paths``: when each agent covers each roadmap vertex and directed edge.
- ``disc_agent_conflicts``: when two agents overlap.

The window formulas and tolerances match the former per-move queries, so a one-move query returns what
it used to.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Directed edges
# ---------------------------------------------------------------------------

class EdgeDir(IntEnum):
    """Which directions of the undirected roadmap edge ``(i, j)``, ``i < j``, may be traversed."""
    LOW = 0     # only i -> j
    HIGH = 1    # only j -> i
    BOTH = 2    # both


_LOW, _HIGH = 1, 2          # direction bits used while folding


def fold_directed_edges(edges: np.ndarray):
    """Directed ``(src, tgt)`` pairs -> one entry per undirected edge: ``(lo, hi, dirs)`` with ``lo < hi``
    and ``dirs`` its ``EdgeDir``. Self-loops are dropped."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, np.empty(0, dtype=np.int8)
    lo, hi = np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])
    n = int(hi.max()) + 1
    keys, inv = np.unique(lo * n + hi, return_inverse=True)
    bits = np.zeros(len(keys), dtype=np.int8)
    np.bitwise_or.at(bits, inv, np.where(edges[:, 0] < edges[:, 1], _LOW, _HIGH).astype(np.int8))
    dirs = np.where(bits == _LOW, EdgeDir.LOW, np.where(bits == _HIGH, EdgeDir.HIGH, EdgeDir.BOTH)).astype(np.int8)
    return keys // n, keys % n, dirs


# ---------------------------------------------------------------------------
# Interval merging
# ---------------------------------------------------------------------------

@njit(cache=True)
def _merge_sorted_intervals(ids, t1, t2, eps):
    """Merge overlapping or touching windows of each id; input sorted by ``(id, t1)``."""
    n = len(ids)
    out_i = np.empty(n, dtype=np.int64)
    out_a = np.empty(n)
    out_b = np.empty(n)
    m = -1
    for k in range(n):
        if m >= 0 and ids[k] == out_i[m] and t1[k] <= out_b[m] + eps:
            if t2[k] > out_b[m]:
                out_b[m] = t2[k]
        else:
            m += 1
            out_i[m] = ids[k]
            out_a[m] = t1[k]
            out_b[m] = t2[k]
    return out_i[:m + 1], out_a[:m + 1], out_b[:m + 1]


def merge_intervals(ids, t1, t2):
    """``(ids, t1, t2)`` sorted by ``(id, t1)``, with each id's overlapping or touching windows merged."""
    ids = np.asarray(ids, dtype=np.int64)
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    if len(ids) == 0:
        return ids, t1, t2
    order = np.lexsort((t1, ids))
    return _merge_sorted_intervals(ids[order], t1[order], t2[order], 1e-12)


def intervals_to_dict(ids, t1, t2, keys=None) -> dict:
    """``{key: [(t1, t2), ...]}`` from ``merge_intervals`` output. ``keys`` (optional) gives each row's
    dict key, e.g. a directed edge tuple."""
    ids = np.asarray(ids)
    if len(ids) == 0:
        return {}
    ks = ids.tolist() if keys is None else keys
    pairs = list(zip(np.asarray(t1).tolist(), np.asarray(t2).tolist()))
    first = np.empty(len(ids), dtype=bool)
    first[0] = True
    first[1:] = ids[1:] != ids[:-1]
    if first.all():                                  # common case: one window per element
        return {k: [w] for k, w in zip(ks, pairs)}
    starts = np.flatnonzero(first).tolist()
    ends = starts[1:] + [len(ids)]
    return {ks[i]: pairs[i:j] for i, j in zip(starts, ends)}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def check_velocity(velocity):
    """``velocity`` as a float, or None (1 time unit per segment). Raises unless None or positive."""
    if velocity is None:
        return None
    if not (np.isfinite(velocity) and velocity > 0):
        raise ValueError(f"velocity must be positive and finite, or None (1 time unit per segment), got {velocity}")
    return float(velocity)


def per_agent(x, n: int) -> list:
    """``x`` as one value per agent: a length-``n`` list / tuple / array as is, anything else repeated."""
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) and len(x) == n else [x] * n


def as_path_list(paths) -> list:
    """``paths`` as a list of ``(P, D)`` arrays, from one path, ``(A, P, D)`` paths or a list of paths."""
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
    raise ValueError(f"paths: expected (P, D), (A, P, D) or a list of (P, D) paths, got shape {arr.shape}")


# ---------------------------------------------------------------------------
# Roadmap contact windows
# ---------------------------------------------------------------------------

def vertex_windows(r0, vel, r, tdur):
    """Per row, when ``|r0 + t * vel| < r`` for ``t`` in ``[0, tdur]``: ``(t1, t2, valid)``. ``r0`` is the
    agent's start minus the vertex. Slack: ``+1e-10`` on the distance, ``+1e-9`` on the exit time."""
    a = np.einsum("ij,ij->i", vel, vel)
    b = 2.0 * np.einsum("ij,ij->i", vel, r0)
    c = np.einsum("ij,ij->i", r0, r0) - r * r + 1e-10
    disc = b * b - 4.0 * a * c
    t1 = np.zeros(len(a))
    t2 = np.zeros(len(a))
    moving = a > 0.0
    valid = moving & (disc >= 0.0)
    stationary_inside = (~moving) & (c < 0.0)
    t2[stationary_inside] = tdur[stationary_inside]
    q = np.sqrt(np.where(valid, disc, 0.0))
    sa = np.where(moving, 2.0 * a, 1.0)
    t1 = np.where(valid, np.clip((-b - q) / sa, 0.0, tdur), t1)
    t2 = np.where(valid, np.clip((-b + q) / sa + 1e-9, 0.0, tdur), t2)
    return t1, t2, valid | stationary_inside


def edge_windows(start, vel, r, tdur, a_pos, b_pos, end_lo, end_hi):
    """Per row, when the agent (``start + t * vel``, ``t`` in ``[0, tdur]``) is within ``r`` of the edge
    ``a_pos -> b_pos``: ``(t_start, t_end, hit)``. Joins the endpoint-ball windows ``[end_lo, end_hi]``
    (``inf`` / ``-inf`` when no endpoint is hit) with the finite-cylinder window."""
    a_to_b = b_pos - a_pos
    seg_len_sq = np.einsum("ij,ij->i", a_to_b, a_to_b) + 1e-12
    vdot = np.einsum("ij,ij->i", a_to_b, vel) / seg_len_sq
    v_perp = vel - vdot[:, None] * a_to_b
    rel = start - a_pos
    pdot = np.einsum("ij,ij->i", rel, a_to_b) / seg_len_sq
    pos_perp = rel - pdot[:, None] * a_to_b
    A = np.einsum("ij,ij->i", v_perp, v_perp)
    B = 2.0 * np.einsum("ij,ij->i", v_perp, pos_perp)
    C = np.einsum("ij,ij->i", pos_perp, pos_perp) - r * r
    LARGE = 1e18
    moving_perp = A > 1e-12
    disc = B * B - 4.0 * A * C
    has_roots = moving_perp & (disc >= 0.0)
    q = np.sqrt(np.maximum(0.0, disc))
    safe_A = np.where(moving_perp, A, 1.0)
    t_perp_enter = np.where(has_roots, (-B - q) / (2.0 * safe_A), LARGE)
    t_perp_exit = np.where(has_roots, (-B + q) / (2.0 * safe_A), -LARGE)
    inside = (~moving_perp) & (C <= 0.0)
    t_perp_enter = np.where(inside, -LARGE, t_perp_enter)
    t_perp_exit = np.where(inside, LARGE, t_perp_exit)
    moving_proj = np.abs(vdot) > 1e-12
    safe_vdot = np.where(moving_proj, vdot, 1.0)
    t_at_a = -pdot / safe_vdot
    t_at_b = (1.0 - pdot) / safe_vdot
    proj_in = (~moving_proj) & (pdot >= 0.0) & (pdot <= 1.0)
    t_proj_enter = np.where(moving_proj, np.minimum(t_at_a, t_at_b), np.where(proj_in, -LARGE, LARGE))
    t_proj_exit = np.where(moving_proj, np.maximum(t_at_a, t_at_b), np.where(proj_in, LARGE, -LARGE))
    t_first = np.maximum(t_perp_enter, t_proj_enter)
    t_leave = np.minimum(t_perp_exit, t_proj_exit)
    has_cyl = t_first < t_leave
    starts = np.where(has_cyl, np.minimum(end_lo, t_first), end_lo)
    ends = np.where(has_cyl, np.maximum(end_hi, t_leave), end_hi)
    tau_start = np.clip(starts, 0.0, tdur)
    tau_end = np.clip(ends, 0.0, tdur)
    hit = ~((tau_start >= tau_end) | np.isinf(starts))
    return tau_start, tau_end, hit


def _agent_values(x, n: int, check, none_value: float) -> np.ndarray:
    """``x`` (one value or one per agent) as a float array, validated by ``check``; None -> ``none_value``."""
    conv = lambda v: none_value if v is None else float(check(v))
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) == n:
        return np.array([conv(v) for v in x], dtype=float)
    return np.full(n, conv(x), dtype=float)


def _check_turning_speed(ts):
    if ts is not None and not ts >= 0:
        raise ValueError(f"turning_speed must be >= 0, inf or None, got {ts}")
    return ts


def _check_radius(r):
    r = float(r)
    if not (np.isfinite(r) and r > 0):
        raise ValueError(f"radius must be positive and finite, got {r}")
    return r


@njit(cache=True)
def _schedule_times(seg_start, seg_count, move, turn, t0):
    """Per segment ``(depart, arrive, turn_end)``, accumulated agent by agent (``t0 + cumsum``)."""
    m = len(move)
    depart = np.empty(m)
    arrive = np.empty(m)
    turn_end = np.empty(m)
    for i in range(len(seg_start)):
        acc = 0.0
        for j in range(seg_count[i]):
            k = seg_start[i] + j
            depart[k] = t0[i] if j == 0 else t0[i] + acc
            acc += move[k]
            arrive[k] = t0[i] + acc
            if j + 1 < seg_count[i]:
                acc += turn[k]
            turn_end[k] = t0[i] + acc
    return depart, arrive, turn_end


def _path_pieces(paths, radius, velocity, t0, turning_speed, goal_hold):
    """Moving segments and stationary pieces of every agent, as ``(seg, still)`` dicts of flat arrays
    (``still["kind"]``: 0 for an in-place turn, 1 for the goal hold). Consecutive duplicate waypoints are
    dropped; ``velocity`` and ``turning_speed`` as in ``CollisionBase.overlapping_interval_paths``."""
    n = len(paths)
    rad = _agent_values(radius, n, _check_radius, np.nan)
    if np.isnan(rad).any():
        raise ValueError("radius must be positive and finite, got None")
    vel = _agent_values(velocity, n, check_velocity, np.nan)            # nan: 1 time unit per segment
    ts = _agent_values(turning_speed, n, _check_turning_speed, np.inf)  # inf: instant turns
    t0s = _agent_values(t0, n, float, 0.0)
    hold = _agent_values(goal_hold, n, float, 0.0)

    pp = [np.asarray(p, dtype=float) for p in paths]
    D = pp[0].shape[-1]
    pp = [p.reshape(-1, D) for p in pp]
    lens = np.fromiter((len(p) for p in pp), dtype=np.int64, count=n)
    X = np.concatenate(pp) if lens.sum() else np.empty((0, D))
    pa = np.repeat(np.arange(n), lens)
    # Drop consecutive duplicate waypoints of the same agent.
    if len(X) > 1:
        keep = np.ones(len(X), dtype=bool)
        keep[1:] = (pa[1:] != pa[:-1]) | np.any(X[1:] != X[:-1], axis=1)
        X, pa = X[keep], pa[keep]
    npts = np.bincount(pa, minlength=n)

    # Segments: consecutive points of the same agent.
    sk = np.flatnonzero(pa[1:] == pa[:-1]) if len(X) > 1 else np.empty(0, np.int64)
    s_agent = pa[sk]
    A, B = X[sk], X[sk + 1]
    d = B - A
    length = np.linalg.norm(d, axis=1)
    sv = vel[s_agent]
    move = np.where(np.isnan(sv), 1.0, length / np.where(np.isnan(sv), 1.0, sv))
    seg_count = np.maximum(npts - 1, 0)
    seg_start = np.concatenate([[0], np.cumsum(seg_count)[:-1]]).astype(np.int64)
    # Turn after segment k (k not the agent's last segment).
    turn = np.zeros(len(sk))
    has_next = np.zeros(len(sk), dtype=bool)
    if len(sk) > 1:
        has_next[:-1] = s_agent[1:] == s_agent[:-1]
        tsk = ts[s_agent]
        timed = has_next & np.isfinite(tsk) & (tsk > 0)
        if timed.any():
            k = np.flatnonzero(timed)
            u = d / length[:, None]
            angle = np.arccos(np.clip(np.sum(u[k + 1] * u[k], axis=1), -1.0, 1.0))
            turn[k] = angle / tsk[k]
    depart, arrive, turn_end = _schedule_times(seg_start, seg_count, move, turn, t0s)

    # Stationary pieces: turns (at B of segment k), then the goal hold, ordered by agent.
    q = np.flatnonzero(has_next & (turn_end > arrive))
    last_pt = np.cumsum(npts) - 1
    h_ag = np.flatnonzero((npts > 0) & (hold > 0))
    last_seg = seg_start + seg_count - 1
    h_arrive = np.where(seg_count[h_ag] > 0, arrive[np.maximum(last_seg[h_ag], 0)] if len(arrive) else 0.0,
                        t0s[h_ag])
    st_agent = np.concatenate([s_agent[q], h_ag]).astype(np.int64)
    order = np.argsort(st_agent, kind="stable")
    still = {"agent": st_agent[order],
             "p": np.concatenate([B[q], X[last_pt[h_ag]]]).reshape(-1, D)[order],
             "w0": np.concatenate([arrive[q], h_arrive]).astype(float)[order],
             "w1": np.concatenate([turn_end[q], h_arrive + hold[h_ag]]).astype(float)[order],
             "kind": np.concatenate([np.zeros(len(q), np.int64), np.ones(len(h_ag), np.int64)])[order]}
    still["r"] = rad[still["agent"]]
    seg = {"agent": s_agent.astype(np.int64), "a": A, "b": B, "depart": depart, "tdur": arrive - depart,
           "r": rad[s_agent]}
    return seg, still


# ---------------------------------------------------------------------------
# Exact distance filters
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _pt_seg_d2(P, i, SA, SB, j):
    """Squared distance from point ``P[i]`` to segment ``SA[j] -> SB[j]``."""
    D = P.shape[1]
    dd = 0.0
    dp = 0.0
    for k in range(D):
        e = SB[j, k] - SA[j, k]
        dd += e * e
        dp += (P[i, k] - SA[j, k]) * e
    t = 0.0
    if dd >= 1e-20:
        t = min(max(dp / dd, 0.0), 1.0)
    s = 0.0
    for k in range(D):
        x = SA[j, k] + t * (SB[j, k] - SA[j, k]) - P[i, k]
        s += x * x
    return s


@njit(cache=True, parallel=True)
def _point_segment_mask(pi, si, ri, P, SA, SB, r2):
    """Pair ``k``: point ``P[pi[k]]`` strictly within ``sqrt(r2[ri[k]])`` of segment ``si[k]``."""
    out = np.empty(len(pi), dtype=np.bool_)
    for k in prange(len(pi)):
        out[k] = _pt_seg_d2(P, pi[k], SA, SB, si[k]) < r2[ri[k]]
    return out


@njit(cache=True, parallel=True)
def _segment_segment_mask(qi, ei, QA, QB, EA, EB, r2):
    """Pair ``k``: segment ``qi[k]`` strictly within ``sqrt(r2[qi[k]])`` of segment ``ei[k]``. The
    minimum distance is at an endpoint (four point-segment distances) or at the interior closest points of
    the two lines."""
    out = np.empty(len(qi), dtype=np.bool_)
    D = QA.shape[1]
    for k in prange(len(qi)):
        q, e = qi[k], ei[k]
        best = min(_pt_seg_d2(QA, q, EA, EB, e), _pt_seg_d2(QB, q, EA, EB, e),
                   _pt_seg_d2(EA, e, QA, QB, q), _pt_seg_d2(EB, e, QA, QB, q))
        aa = 0.0
        ee = 0.0
        bd = 0.0
        c = 0.0
        f = 0.0
        for j in range(D):
            d1 = QB[q, j] - QA[q, j]
            d2 = EB[e, j] - EA[e, j]
            rr = QA[q, j] - EA[e, j]
            aa += d1 * d1
            ee += d2 * d2
            bd += d1 * d2
            c += d1 * rr
            f += d2 * rr
        den = aa * ee - bd * bd
        if aa > 1e-20 and ee > 1e-20 and den > 1e-12 * aa * ee:
            s = (bd * f - c * ee) / den
            t = (aa * f - bd * c) / den
            if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
                dist = 0.0
                for j in range(D):
                    x = QA[q, j] + s * (QB[q, j] - QA[q, j]) - EA[e, j] - t * (EB[e, j] - EA[e, j])
                    dist += x * x
                best = min(best, dist)
        out[k] = best < r2[q]
    return out


def segment_candidates(backend, a, b, r):
    """Vertices and edges strictly within ``r[k]`` of each segment ``a[k] -> b[k]``:
    ``((seg, vertex), (seg, edge))``. Backend candidates, then the exact distance test."""
    a = np.ascontiguousarray(a, dtype=float)
    b = np.ascontiguousarray(b, dtype=float)
    r = np.broadcast_to(np.asarray(r, dtype=float), (len(a),))
    sv, vid = backend._vertex_candidates(a, b, r)
    se, eid = backend._edge_candidates(a, b, r)
    r2 = r * r
    if len(sv):
        m = _point_segment_mask(vid, sv, sv, backend.vertex_positions, a, b, r2)
        sv, vid = sv[m], vid[m]
    if len(se):
        m = _segment_segment_mask(se, eid, a, b, backend.edge_src_positions, backend.edge_tgt_positions, r2)
        se, eid = se[m], eid[m]
    return (sv, vid), (se, eid)


def point_candidates(backend, p, r):
    """Vertices strictly within ``r[k] - 1e-10`` and edges strictly within ``r[k]`` of each point
    ``p[k]``: ``((point, vertex), (point, edge))``."""
    p = np.ascontiguousarray(p, dtype=float)
    r = np.broadcast_to(np.asarray(r, dtype=float), (len(p),))
    sv, vid = backend._vertex_candidates(p, p, r)
    se, eid = backend._edge_candidates(p, p, r)
    if len(sv):
        rv = np.maximum(r - 1e-10, 0.0)
        m = _point_segment_mask(vid, sv, sv, backend.vertex_positions, p, p, rv * rv)
        sv, vid = sv[m], vid[m]
    if len(se):
        m = _point_segment_mask(se, eid, se, p, backend.edge_src_positions, backend.edge_tgt_positions, r * r)
        se, eid = se[m], eid[m]
    return (sv, vid), (se, eid)


def sweep_paths(backend, paths, radius, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                as_arrays: bool = True) -> list:
    """``CollisionBase.overlapping_interval_paths`` for ``backend`` (a ``CollisionBase``)."""
    paths = as_path_list(paths)
    n = len(paths)
    if n == 0:
        return []
    seg, still = _path_pieces(paths, radius, velocity, t0, turning_speed, goal_hold)
    V = backend.vertex_positions
    nv = max(len(V), 1)
    v_parts, e_parts = [], []                                # (agent, id, t1, t2)

    if len(seg["agent"]):
        A, Bp, dep, tdur, r = seg["a"], seg["b"], seg["depart"], seg["tdur"], seg["r"]
        vel = (Bp - A) / tdur[:, None]
        (sv, vid), (se, eid) = segment_candidates(backend, A, Bp, r)
        t1, t2, valid = vertex_windows(A[sv] - V[vid], vel[sv], r[sv], tdur[sv])
        sv, vid, t1, t2 = sv[valid], vid[valid], t1[valid], t2[valid]
        v_parts.append((seg["agent"][sv], vid, dep[sv] + t1, dep[sv] + t2))
        if len(se):
            # Endpoint balls: the same segment's window of each endpoint vertex, if it was hit.
            key = sv * nv + vid
            order = np.argsort(key)
            key_s, lo_s, hi_s = key[order], t1[order], t2[order]

            def lookup(vtx):
                k = se * nv + vtx
                if len(key_s) == 0:
                    return np.full(len(k), np.inf), np.full(len(k), -np.inf)
                pos = np.minimum(np.searchsorted(key_s, k), len(key_s) - 1)
                found = key_s[pos] == k
                return np.where(found, lo_s[pos], np.inf), np.where(found, hi_s[pos], -np.inf)

            src, tgt = backend.edge_src_array[eid], backend.edge_tgt_array[eid]
            lo_a, hi_a = lookup(src)
            lo_b, hi_b = lookup(tgt)
            ts_, te_, hit = edge_windows(A[se], vel[se], r[se], tdur[se], V[src], V[tgt],
                                         np.minimum(lo_a, lo_b), np.maximum(hi_a, hi_b))
            e_parts.append((seg["agent"][se[hit]], eid[hit], dep[se[hit]] + ts_[hit], dep[se[hit]] + te_[hit]))

    if len(still["agent"]):
        (pv, vid), (pe, eid) = point_candidates(backend, still["p"], still["r"])
        v_parts.append((still["agent"][pv], vid, still["w0"][pv], still["w1"][pv]))
        e_parts.append((still["agent"][pe], eid, still["w0"][pe], still["w1"][pe]))
    return assemble(backend, n, v_parts, e_parts, as_arrays)


def assemble(backend, n_agents, v_parts, e_parts, as_arrays=False) -> list:
    """Expand canonical edge hits to their allowed directions, merge each (agent, element)'s windows and
    split per agent. ``v_parts`` / ``e_parts``: lists of ``(agent, element, t1, t2)`` arrays."""
    nv = max(len(backend.vertex_positions) if backend.vertex_positions is not None else 0, 1)
    cat = lambda parts, i, dt: (np.concatenate([np.asarray(q[i]) for q in parts]).astype(dt)
                                if parts else np.empty(0, dt))
    va, vid, vt1, vt2 = (cat(v_parts, i, dt) for i, dt in enumerate((np.int64, np.int64, float, float)))
    ea, ec, et1, et2 = (cat(e_parts, i, dt) for i, dt in enumerate((np.int64, np.int64, float, float)))
    d = np.frombuffer(backend.edge_dir, dtype=np.uint8)[ec] if len(ec) else np.empty(0, np.uint8)
    fwd, bwd = d != EdgeDir.HIGH, d != EdgeDir.LOW
    lo_v = backend.edge_src_array[ec] if len(ec) else ec
    hi_v = backend.edge_tgt_array[ec] if len(ec) else ec
    e_agent = np.concatenate([ea[fwd], ea[bwd]])
    e_src = np.concatenate([lo_v[fwd], hi_v[bwd]])
    e_tgt = np.concatenate([hi_v[fwd], lo_v[bwd]])
    e_t1 = np.concatenate([et1[fwd], et1[bwd]])
    e_t2 = np.concatenate([et2[fwd], et2[bwd]])
    vm = merge_intervals(va * nv + vid, vt1, vt2)
    em = merge_intervals((e_agent * nv + e_src) * nv + e_tgt, e_t1, e_t2)
    v_cut = np.searchsorted(vm[0] // nv, np.arange(n_agents + 1))
    e_cut = np.searchsorted(em[0] // (nv * nv), np.arange(n_agents + 1))
    results = []
    for i in range(n_agents):
        vs, es = slice(v_cut[i], v_cut[i + 1]), slice(e_cut[i], e_cut[i + 1])
        vids, code = vm[0][vs] % nv, em[0][es]
        src, tgt = (code // nv) % nv, code % nv
        if as_arrays:
            results.append(((vids, vm[1][vs], vm[2][vs]), (src, tgt, em[1][es], em[2][es])))
        else:
            results.append((intervals_to_dict(vids, vm[1][vs], vm[2][vs]),
                            intervals_to_dict(code, em[1][es], em[2][es], keys=list(zip(src.tolist(), tgt.tolist())))))
    return results


# ---------------------------------------------------------------------------
# Agent-agent conflicts
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _ball_sweep_prune(bmin, bmax, w0, w1, agent):
    """Broad phase: pairs of pieces of different agents whose boxes (``(n, D)`` min / max) and time
    windows overlap. Sort-and-sweep on axis 0, in two parallel passes (count, then fill)."""
    n, D = bmin.shape
    order = np.argsort(bmin[:, 0])
    counts = np.zeros(n, dtype=np.int64)
    for ii in prange(n):
        i = order[ii]
        c = 0
        for jj in range(ii + 1, n):
            j = order[jj]
            if bmin[j, 0] > bmax[i, 0]:
                break
            if agent[i] == agent[j] or not max(w0[i], w0[j]) < min(w1[i], w1[j]):
                continue
            ok = True
            for k in range(1, D):
                if bmin[j, k] > bmax[i, k] or bmin[i, k] > bmax[j, k]:
                    ok = False
                    break
            if ok:
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
            j = order[jj]
            if bmin[j, 0] > bmax[i, 0]:
                break
            if agent[i] == agent[j] or not max(w0[i], w0[j]) < min(w1[i], w1[j]):
                continue
            ok = True
            for k in range(1, D):
                if bmin[j, k] > bmax[i, k] or bmin[i, k] > bmax[j, k]:
                    ok = False
                    break
            if ok:
                pa[c] = i
                pb[c] = j
                c += 1
    return pa, pb


@njit(cache=True, parallel=True)
def _ball_pair_windows(pa, pb, w0, w1, x0, v, rad):
    """Per candidate pair: whether the two balls overlap for a positive time, and when. Piece ``k`` is at
    ``x0[k] + v[k] * (t - w0[k])`` during ``[w0[k], w1[k]]``."""
    n = len(pa)
    D = x0.shape[1]
    hit = np.zeros(n, dtype=np.bool_)
    t1 = np.empty(n)
    t2 = np.empty(n)
    for k in prange(n):
        a, b = pa[k], pb[k]
        s = max(w0[a], w0[b])
        T = min(w1[a], w1[b]) - s
        if not T > 0.0:
            continue
        aa = 0.0
        bb = 0.0
        cc = 0.0
        for j in range(D):
            d = (x0[b, j] + v[b, j] * (s - w0[b])) - (x0[a, j] + v[a, j] * (s - w0[a]))
            w = v[b, j] - v[a, j]
            aa += w * w
            bb += 2.0 * d * w
            cc += d * d
        r = rad[a] + rad[b]
        cc -= r * r
        if aa == 0.0:
            if cc < 0.0:
                lo, hi = 0.0, T
            else:
                continue
        else:
            disc = bb * bb - 4.0 * aa * cc
            if disc <= 0.0:
                continue
            q = np.sqrt(disc)
            lo = max((-bb - q) / (2.0 * aa), 0.0)
            hi = min((-bb + q) / (2.0 * aa), T)
        if hi - lo > 1e-12:
            hit[k] = True
            t1[k] = s + lo
            t2[k] = s + hi
    return hit, t1, t2


def pair_windows_output(n, i_ag, j_ag, t1, t2, first_only=False, as_arrays=False):
    """Agent-pair windows as ``{(i, j): [(t1, t2), ...]}`` (``i < j``, merged), ``{(i, j): first time}``
    with ``first_only``, or ``(i, j, t1, t2)`` arrays with ``as_arrays``."""
    lo_ag, hi_ag = np.minimum(i_ag, j_ag), np.maximum(i_ag, j_ag)
    key, a1, a2 = merge_intervals(lo_ag * n + hi_ag, t1, t2)
    if first_only:
        first = np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, dtype=bool)
        key, a1 = key[first], a1[first]
        if as_arrays:
            return key // n, key % n, a1, a1.copy()
        return {(int(k // n), int(k % n)): float(t) for k, t in zip(key.tolist(), a1.tolist())}
    if as_arrays:
        return key // n, key % n, a1, a2
    return intervals_to_dict(key, a1, a2, keys=[(int(k // n), int(k % n)) for k in key.tolist()])


def disc_agent_conflicts(paths, radius, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0, pairs=None,
                         first_only: bool = False, as_arrays: bool = False, eps: float = 1e-9):
    """``CollisionBase.agent_conflicts``, in any dimension.
    Each radius is reduced by ``eps / 2``, so touching is not contact."""
    paths = as_path_list(paths)
    n = len(paths)
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), np.empty(0))
    if n == 0:
        return empty if as_arrays else {}
    seg, still = _path_pieces(paths, radius, velocity, t0, turning_speed, goal_hold)
    D = seg["a"].shape[1] if len(seg["a"]) else still["p"].shape[1]
    agent = np.concatenate([seg["agent"], still["agent"]])
    if len(agent) == 0:
        return empty if as_arrays else {}
    w0 = np.concatenate([seg["depart"], still["w0"]])
    w1 = np.concatenate([seg["depart"] + seg["tdur"], still["w1"]])
    x0 = np.concatenate([seg["a"].reshape(-1, D), still["p"].reshape(-1, D)])
    v = np.concatenate([(seg["b"] - seg["a"]).reshape(-1, D) / seg["tdur"][:, None], np.zeros((len(still["agent"]), D))])
    rad = np.concatenate([seg["r"], still["r"]]) - 0.5 * eps
    xe = np.concatenate([seg["b"].reshape(-1, D), still["p"].reshape(-1, D)])
    bmin = np.minimum(x0, xe) - rad[:, None]
    bmax = np.maximum(x0, xe) + rad[:, None]
    qa, qb = _ball_sweep_prune(bmin, bmax, w0, w1, agent)
    if pairs is not None:
        allowed = np.zeros((n, n), dtype=bool)
        for i, j in pairs:
            allowed[i, j] = allowed[j, i] = True
        ok = allowed[agent[qa], agent[qb]]
        qa, qb = qa[ok], qb[ok]
    if len(qa) == 0:
        return empty if as_arrays else {}
    hit, t1, t2 = _ball_pair_windows(qa, qb, w0, w1, x0, v, rad)
    return pair_windows_output(n, agent[qa[hit]], agent[qb[hit]], t1[hit], t2[hit], first_only, as_arrays)
