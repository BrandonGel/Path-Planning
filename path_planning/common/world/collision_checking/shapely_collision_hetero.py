"""
Shapely (GEOS) spatial-sweep collision checker over a ``GraphSampler`` roadmap.

``ShapelyCollisionHetero(config)`` checks sweeps against a roadmap handed to
``set_graph(vertices, edges)``: the node positions (``graph_map.position``,
roadmap index order) and the directed ``(src, tgt)`` pairs of
``graph_map.edges`` (the CSR roadmap's ``(E, 2)`` pairs), so the vertex
indices it returns are the map's node indices; the map's
``set_constraint_sweep`` does that call. The two query flags live in
``ShapelyCollisionConfig`` (``record_sweep``, ``use_exact_collision_check``),
which round-trips through ``to_dict`` / ``from_dict`` like the map's other
configs.

Agents of any 2D shape (an ``agent_shape.Shape``, a radius, an outline or a
polygon) are swept along whole paths, many agents at once, with
``overlapping_interval_paths``: when each agent's body covers each roadmap vertex
and directed edge; ``agent_conflicts`` gives the exact collision windows between
agents. Both override the disc-only versions of ``ShapelyCollision`` (which the
map's ``get_constraint_sweep`` uses) and also accept discs.

Shapely/GEOS is planar, so this backend is 2D only (``set_graph`` raises for 3D).
"""

from __future__ import annotations
import numpy as np
import shapely
from path_planning.common.world.collision_checking.shapely_util import (
    merge_intervals, intervals_to_dict,
    clip_segments_to_pieces, points_in_pieces, sweep_rings_timed,
    _vertex_hit_times, _edge_hit_times, convert_to_shape_list, convert_to_path_list,
    motion_pieces_batch, _agent_pair_windows, _sweep_prune_pairs, _expand_ring_pairs)
from path_planning.common.world.collision_checking.shapely_collision import (
    ShapelyCollisionConfig, ShapelyCollision, EdgeDir)


class ShapelyCollisionHetero(ShapelyCollision):
    """``ShapelyCollision`` for agents of different sizes and shapes: footprint sweeps along
    whole paths (``overlapping_interval_paths``). Vertex indices in the results are the map's
    node indices; edge keys are directed ``(src, tgt)`` node-index pairs. The disc-only
    per-move queries are inherited from ``ShapelyCollision``."""

    def __init__(self, config: "ShapelyCollisionConfig | dict | None" = None):
        super().__init__(config)

    # ------------------------------------------------------------ arbitrary footprints
    #: Inward shrink applied before every containment test on a footprint
    #: region: contact along a region's boundary (a vertex sliding on a
    #: footprint edge, an edge collinear with it) is tangential, not a
    #: collision, and float rotation noise must not promote it to one.
    STRICT_EPS = 1e-9

    @classmethod
    def _strict(cls, region):
        """``region`` (a polygon or an array of them) shrunk by ``STRICT_EPS``
        so boundary-only contact drops out."""
        return shapely.buffer(region, -cls.STRICT_EPS)

    @staticmethod
    def _as_pieces(region) -> np.ndarray:
        """``region`` (one polygon or a list/array of pieces) as a 1-D object array."""
        return np.atleast_1d(np.asarray(region, dtype=object))

    def _covered_vertices(self, region) -> np.ndarray:
        """Vertex indices strictly inside ``region``: one polygon or a list of pieces
        (e.g. from ``sweep_region``), queried in one batched call. Boundary contact is
        not a collision. Sorted, each vertex once."""
        pieces = self._as_pieces(region)
        if self.vertex_tree is None or len(pieces) == 0:
            return np.empty(0, dtype=np.int64)
        _, idx = self.vertex_tree.query(pieces, predicate="contains_properly")
        return np.unique(idx).astype(np.int64)

    def _crossed_edges(self, region) -> np.ndarray:
        """Canonical edge indices whose geometry runs through the interior of
        ``region`` (one polygon or a list of pieces): segments with a positive
        length of overlap with some piece (a segment that only touches the
        boundary is not a collision). Sorted, each edge once."""
        pieces = self._as_pieces(region)
        if self.edge_tree is None or len(pieces) == 0:
            return np.empty(0, dtype=np.int64)
        strict = self._strict(pieces)
        p, e = self.edge_tree.query(strict, predicate="intersects")
        if len(e) == 0:
            return np.empty(0, dtype=np.int64)
        keep = shapely.length(shapely.intersection(strict[p], self.edge_tree.geometries[e])) > 1e-12
        return np.unique(e[keep]).astype(np.int64)

    # ------------------------------------------------------------ timed overlapping interval sweeps
    def overlapping_interval_paths(self, paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                                   method: str = "front_face", exact=False, as_arrays: bool = True,
                                   heading=None) -> list:
        """When each agent's body, moving along its path, covers each roadmap element:
        ``[((vertex, t1, t2), (src, tgt, t1, t2)), ...]``, one entry per path: NumPy arrays, one row per
        window, sorted by element then time, absolute times from ``t0``, each element's windows merged; edge
        rows are the directed edges the roadmap allows. ``as_arrays=False`` gives dicts instead,
        ``[({vertex: [(t1, t2), ...]}, {(src, tgt): [(t1, t2), ...]}), ...]``. All paths go into one piece set, the
        roadmap trees are queried once and the hits are timed in one numba pass, then split per agent.
        Touching is not contact, and contacts of zero duration (e.g. at an instant turn) are left out.

        ``exact=False``: the pieces of ``sweep_rings_timed(..., method)``. Polygon pieces (``front_face``
        rectangles, ``exact`` hulls) are clipped to when the body's reach along travel spans the element
        (exact for a travel-aligned rectangle, longer otherwise, never shorter). Capsules (every move of a
        disc agent, ``"buffer"`` moves) use the exact disc window cut to the body's reach. ``exact=True``:
        each move of a polygon agent uses the footprint's convex pieces moved along the segment, giving
        the exact windows; a concave body that leaves an element and comes back gets separate windows.
        Either way, turns use the bounding disc and the goal hold the footprint, each for its whole
        window, and a disc agent (``Circle`` or a radius) is exact.

        ``paths``: one path ``(P, 2)``, same-length paths ``(A, P, 2)`` or a list / tuple of paths of any
        lengths; the result is always a list, one entry per path. ``shapes`` (one shape, including a single
        ``(k, 2)`` outline, or one per path), ``velocity``, ``t0``, ``turning_speed``, ``goal_hold``, ``exact``
        and ``heading`` are one value for all agents or a list / array with one value per path.

        ``velocity = None``: every segment takes 1 time unit, whatever its length. ``turning_speed``: ``None``
        or ``inf`` turns instantly at each waypoint; ``0`` never turns, so the body keeps the first segment's
        rotation (its direction + the shape's orientation) along the whole path; a positive value turns in
        place at that many rad/s. ``heading`` (radians, default None) fixes the body's rotation instead: it
        faces ``heading`` + its orientation along the whole path, never turning (``turning_speed`` is
        then ignored)."""
        paths = convert_to_path_list(paths)
        n = len(paths)
        if n == 0:
            return []
        def per(x):
            return list(x) if isinstance(x, (list, tuple, np.ndarray)) and len(x) == n else [x] * n
        shapes = [shapes] * n if convert_to_shape_list(shapes) else per(shapes)
        vels, t0s, turns, holds, exacts = per(velocity), per(t0), per(turning_speed), per(goal_hold), per(exact)
        heads = per(heading)
        sets = []
        for i in range(n):
            sr = sweep_rings_timed(paths[i], shapes[i], vels[i], method, t0s[i], turns[i], holds[i],
                                   exact=bool(exacts[i]), eps=self.STRICT_EPS, heading=heads[i])
            sets.append(self._piece_set(sr))
        return self._assemble(n, *self._piece_hits(self._concat_piece_sets(sets)), as_arrays=as_arrays)

    # Overlapping Interval Paths Step 1: Get each agent piece
    @staticmethod
    def _piece_set(sr: dict) -> dict:
        """One path's pieces (``sweep_rings_timed`` output) with what the fused kernels need per piece
        (speed of its segment, reach along travel) and per segment (start, unit direction, length,
        departure time)."""
        n = len(sr["windows"])
        p, seg_w, sg = sr["path"], sr["seg_windows"], sr["segment"]
        if len(p) >= 2:
            d = p[1:] - p[:-1]
            L = np.hypot(d[:, 0], d[:, 1])
            seg = (p[:-1, 0].copy(), p[:-1, 1].copy(), d[:, 0] / L, d[:, 1] / L, L, seg_w[:, 0].copy())
        else:
            seg = tuple(np.empty(0) for _ in range(6))
        vel = np.ones(n)                                     # turn / goal pieces keep their whole window
        vel[sg >= 0] = sr["seg_speed"][sg[sg >= 0]]
        return {"rings": sr["rings"], "kind": sr["kind"], "cap": sr["cap"], "bounds": sr["bounds"],
                "geoms": sr["geoms"], "windows": sr["windows"], "segment": sg,
                "vel": vel, "xmin": sr["reach"][:, 0].copy(), "xmax": sr["reach"][:, 1].copy(),
                "seg": seg, "agent": np.zeros(n, dtype=np.int64)}

    # Overlapping Interval Paths Step 2: Concat all the agent pieces
    @staticmethod
    def _concat_piece_sets(sets: list) -> dict:
        """Several agents' piece sets as one: ring and segment indices offset, ``agent`` tags each piece."""
        if len(sets) == 1:
            return sets[0]
        rs, re, rx, ry, sign, cvx, segs, agents = [], [], [], [], [], [], [], []
        seg_parts = [[] for _ in range(6)]
        c_off = g_off = 0
        for i, s in enumerate(sets):
            r = s["rings"]
            rs.append(r[0] + c_off); re.append(r[1] + c_off); rx.append(r[2]); ry.append(r[3])
            sign.append(r[4]); cvx.append(r[5]); c_off += len(r[2])
            segs.append(np.where(s["segment"] >= 0, s["segment"] + g_off, -1)); g_off += len(s["seg"][0])
            for j in range(6):
                seg_parts[j].append(s["seg"][j])
            agents.append(np.full(len(s["windows"]), i, dtype=np.int64))
        cat = np.concatenate
        return {"rings": (cat(rs), cat(re), cat(rx), cat(ry), cat(sign), cat(cvx)),
                "kind": cat([s["kind"] for s in sets]), "cap": cat([s["cap"] for s in sets]),
                "bounds": cat([s["bounds"] for s in sets]), "geoms": cat([s["geoms"] for s in sets]),
                "windows": cat([s["windows"] for s in sets]), "segment": cat(segs),
                "vel": cat([s["vel"] for s in sets]), "xmin": cat([s["xmin"] for s in sets]),
                "xmax": cat([s["xmax"] for s in sets]), "seg": tuple(cat(q) for q in seg_parts), "agent": cat(agents)}

    # Overlapping Interval Paths Step 3: Find where each agent pieces are touching and when
    def _piece_hits(self, ps: dict):
        """Raw hits of a piece set: ``(v_agent, v_id, v_t1, v_t2), (e_agent, e_edge, e_t1, e_t2)`` with
        canonical edge indices. Bounding-box candidates from the trees, then the fused numba kernels
        (inside test / clipping by ``STRICT_EPS`` + time window); a concave goal footprint goes through
        shapely and keeps its whole window. Contacts of zero duration are left out, as in the disc classes and
        ``CGALCollisionHetero``: pieces whose window has no length (an instant turn, ``goal_hold = 0``) are not
        queried at all, and hits whose window has no length (a grazing touch) are dropped."""
        empty_i, empty_f = np.empty(0, np.int64), np.empty(0)
        none = (empty_i, empty_i, empty_f, empty_f)
        if len(ps["windows"]) == 0 or self.vertex_tree is None:
            return none, none
        rings, seg, eps = ps["rings"], ps["seg"], self.STRICT_EPS
        common = (ps["segment"], ps["windows"][:, 0].copy(), ps["windows"][:, 1].copy(), ps["vel"], ps["xmin"],
                  ps["xmax"]) + seg
        live = np.flatnonzero(ps["windows"][:, 1] - ps["windows"][:, 0] > 1e-12)   # pieces with a duration
        b = ps["bounds"][live]
        boxes = shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3])
        out = []
        for tree, is_vertex in ((self.vertex_tree, True), (self.edge_tree, False)):
            pl, ic = tree.query(boxes)
            pc = live[pl]
            if is_vertex:
                xy = self.vertex_positions[ic]
                st, t1, t2 = _vertex_hit_times(pc, xy[:, 0].copy(), xy[:, 1].copy(), ps["kind"], ps["cap"],
                                               *rings, eps, *common)
            else:
                A = self.vertex_positions[self.edge_src_array[ic]]
                B = self.vertex_positions[self.edge_tgt_array[ic]]
                st, t1, t2 = _edge_hit_times(pc, A[:, 0].copy(), A[:, 1].copy(), B[:, 0].copy(), B[:, 1].copy(),
                                             ps["kind"], ps["cap"], *rings, eps, *common)
            fb = np.flatnonzero(st == -1)                  # concave pieces: waypoint pieces, whole window
            if len(fb):
                if is_vertex:
                    ok = points_in_pieces(ps["geoms"], pc[fb], self.vertex_positions[ic[fb]], eps, rings)
                else:
                    lo, hi = clip_segments_to_pieces(ps["geoms"], pc[fb], A[fb], B[fb], eps, rings)
                    ok = (hi - lo) * np.hypot(*(B[fb] - A[fb]).T) > 1e-12
                st[fb[ok]] = 1
                t1[fb[ok]], t2[fb[ok]] = ps["windows"][pc[fb[ok]], 0], ps["windows"][pc[fb[ok]], 1]
            h = st == 1
            h[h] = t2[h] - t1[h] > 1e-12                    # drop zero-duration hits (only real hits have times)
            out.append((ps["agent"][pc[h]], ic[h].astype(np.int64), t1[h], t2[h]))
        return out[0], out[1]

    # Overlapping Interval Paths Step 4: Get the final per-agent hit results
    def _assemble(self, n_agents, v_hits, e_hits, as_arrays=False) -> list:
        """Expand edges to their allowed directions, merge each (agent, element)'s windows and split the
        result per agent, as dicts or arrays."""
        nv = max(len(self.vertex_positions) if self.vertex_positions is not None else 0, 1)
        va, vid, vt1, vt2 = v_hits
        ea, ec, et1, et2 = e_hits
        d = np.frombuffer(self.edge_dir, dtype=np.uint8)[ec] if len(ec) else np.empty(0, np.uint8)
        fwd, bwd = d != EdgeDir.HIGH, d != EdgeDir.LOW
        lo_v, hi_v = (self.edge_src_array[ec], self.edge_tgt_array[ec]) if len(ec) else (ec, ec)
        e_agent = [ea[fwd], ea[bwd]]; e_src = [lo_v[fwd], hi_v[bwd]]; e_tgt = [hi_v[fwd], lo_v[bwd]]
        e_t1 = [et1[fwd], et1[bwd]]; e_t2 = [et2[fwd], et2[bwd]]
        cat = np.concatenate
        vm = merge_intervals(va * nv + vid, vt1, vt2)
        em = merge_intervals((cat(e_agent) * nv + cat(e_src)) * nv + cat(e_tgt), cat(e_t1), cat(e_t2))
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

    # ------------------------------------------------------------ check agents overlapping sweeps
    def agent_conflicts(self, paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                        heading=None, pairs=None, first_only: bool = False, as_arrays: bool = False):
        """When the bodies of every two agents overlap, exactly (touching is not contact).

        Same inputs as ``overlapping_interval_paths`` (``paths``, ``shapes``, ``velocity``, ``t0``,
        ``turning_speed``, ``goal_hold``, ``heading``; each one value or one per agent). Returns
        ``{(i, j): [(t1, t2), ...]}`` (``i < j``, windows sorted and merged) for the pairs that collide;
        ``first_only=True`` gives ``{(i, j): first contact time}``; ``as_arrays=True`` gives
        ``(i, j, t1, t2)`` arrays (one row per window, or per pair with ``first_only``). ``pairs``: only
        check these ``(i, j)`` agent pairs (e.g. one replanned agent against the others).

        Each agent's schedule is split into rigid pieces with constant velocity (segments, turns, goal
        hold). For two pieces of different agents that overlap in time, B's position relative to A moves
        on a line, and the bodies overlap while it lies in ``A (+) (-B)``: one clip per pair of convex
        pieces (exact, including concave bodies); discs reduce to distance tests. During an in-place
        turn the agent is its bounding disc (conservative). An agent exists from ``t0`` until it arrives,
        then for ``goal_hold`` (``inf``: it stays). Candidates come from a sort-and-sweep over each
        piece's box over its window (different agents, overlapping windows); both it and the exact test
        run in parallel in numba (``prange``)."""
        paths = convert_to_path_list(paths)
        n = len(paths)
        def per(x):
            return list(x) if isinstance(x, (list, tuple, np.ndarray)) and len(x) == n else [x] * n
        shapes = [shapes] * n if convert_to_shape_list(shapes) else per(shapes)
        vels, t0s, turns, holds, heads = per(velocity), per(t0), per(turning_speed), per(goal_hold), per(heading)

        empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), np.empty(0))
        if n == 0:
            return empty if as_arrays else {}
        mp = motion_pieces_batch(paths, shapes, vels, t0s, turns, holds, heads, self.STRICT_EPS)
        if len(mp["w0"]) == 0:
            return empty if as_arrays else {}
        agent, w0, w1, x0, y0, vx, vy, rad = (mp[k] for k in ("agent", "w0", "w1", "x0", "y0", "vx", "vy", "rad"))
        rs_, re_, rx, ry = mp["rings"]
        ring_lo, ring_cnt, ring_step = mp["ring_lo"], mp["ring_cnt"], mp["ring_step"]
        # Box over each window: the reference point's path, grown by the bounding radius.
        end_t = np.where(np.isfinite(w1), w1, w0)
        xe, ye = x0 + vx * (end_t - w0), y0 + vy * (end_t - w0)
        R = mp["radius"]
        box = np.column_stack([np.minimum(x0, xe) - R, np.minimum(y0, ye) - R,
                               np.maximum(x0, xe) + R, np.maximum(y0, ye) + R])

        # Broad phase: pieces of different agents whose boxes and windows overlap (sort-and-sweep).
        qa, qb = _sweep_prune_pairs(box[:, 0], box[:, 2], box[:, 1], box[:, 3], w0, w1, agent)
        if pairs is not None:
            allowed = np.zeros((n, n), dtype=bool)
            for i, j in pairs:
                allowed[i, j] = allowed[j, i] = True
            ok = allowed[agent[qa], agent[qb]]
            qa, qb = qa[ok], qb[ok]
        if len(qa) == 0:
            return empty if as_arrays else {}
        # Narrow phase: one candidate per pair of convex pieces, exact windows in parallel.
        pa, pb, ra, rb = _expand_ring_pairs(qa, qb, ring_lo, ring_cnt, ring_step)
        hit, t1, t2 = _agent_pair_windows(pa, pb, ra, rb, w0, w1, x0, y0, vx, vy, rad, rs_, re_, rx, ry)

        i_ag, j_ag = agent[pa[hit]], agent[pb[hit]]
        lo_ag, hi_ag = np.minimum(i_ag, j_ag), np.maximum(i_ag, j_ag)
        key, a1, a2 = merge_intervals(lo_ag * n + hi_ag, t1[hit], t2[hit])
        if first_only:
            first = np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, dtype=bool)
            key, a1 = key[first], a1[first]
            if as_arrays:
                return key // n, key % n, a1, a1.copy()
            return {(int(k // n), int(k % n)): float(t) for k, t in zip(key.tolist(), a1.tolist())}
        if as_arrays:
            return key // n, key % n, a1, a2
        return intervals_to_dict(key, a1, a2, keys=[(int(k // n), int(k % n)) for k in key.tolist()])
