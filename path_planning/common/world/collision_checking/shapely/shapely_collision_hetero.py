"""Polygon sweep collision checker over a ``GraphSampler`` roadmap (2D, Shapely / GEOS).

``ShapelyCollisionHetero`` is ``ShapelyCollision`` for agents of any 2D shape (a ``Shape``, a radius or an
outline), each with its own speed. ``overlapping_interval_paths`` reports when each agent's body covers
each roadmap vertex and directed edge, with a fast front-face sweep (``shapely_util``) or exactly
(``polytope_sweep``); ``agent_conflicts`` gives the exact collision windows between agents
(``polytope_sweep``). For 3D use ``CGALCollisionHetero``.
"""

from __future__ import annotations
import numpy as np
import shapely
from path_planning.common.world.collision_checking.sweep.disc_sweep import assemble, as_path_list, per_agent
from path_planning.common.world.collision_checking.sweep.polytope_sweep import body_roadmap_hits, body_agent_conflicts
from path_planning.common.world.collision_checking.shapely.shapely_util import (
    clip_segments_to_pieces, points_in_pieces, sweep_rings_timed, convert_to_shape_list,
    _vertex_hit_times, _edge_hit_times)
from path_planning.common.world.collision_checking.shapely.shapely_collision import ShapelyCollision


class ShapelyCollisionHetero(ShapelyCollision):
    """``ShapelyCollision`` for agents of different shapes and sizes, swept along whole paths."""

    #: Inward shrink of a footprint before every containment test, so that tangential contact and float
    #: noise are not collisions.
    STRICT_EPS = 1e-9

    # ------------------------------------------------------------ timed overlapping interval sweeps
    def overlapping_interval_paths(self, paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                                   exact=False, as_arrays: bool = True, heading=None) -> list:
        """When each agent's body covers each roadmap element along its path. One entry per path:
        ``((vertex, t1, t2), (src, tgt, t1, t2))`` arrays, one row per window, sorted by element then time,
        absolute times, windows merged. ``as_arrays=False``: ``({vertex: [(t1, t2), ...]},
        {(src, tgt): [(t1, t2), ...]})`` dicts instead. Touching is not contact, and zero-duration contacts
        are left out.

        ``paths``: one ``(P, 2)`` path, ``(A, P, 2)`` paths or a list of paths. The other arguments are one
        value or one per path:

        - ``shapes``: a ``Shape``, a radius, a ``(K, 2)`` outline or a shapely ``Polygon``.
        - ``velocity``: speed; ``None`` means every segment takes 1 time unit.
        - ``t0``: start time.
        - ``turning_speed`` (rad/s): ``None`` or ``inf`` turns instantly; ``0`` never turns (the body keeps
          its first segment's rotation); a positive value turns in place at that rate.
        - ``goal_hold``: time spent at the last waypoint (``inf``: forever).
        - ``heading`` (radians): fixes the body's rotation to ``heading`` + its orientation (no turning).
        - ``exact=False``: the fast front-face sweep (each move covers the body's bounding box along
          travel). Exact for discs and travel-aligned rectangles, never shorter than the true windows
          otherwise. ``exact=True``: the exact sweep of ``polytope_sweep``; a concave body that leaves an
          element and returns gives separate windows.

        Either way, turns use the bounding disc and the goal hold the footprint, each for its whole window."""
        paths = as_path_list(paths)
        n = len(paths)
        if n == 0:
            return []
        per = lambda x: per_agent(x, n)
        shapes = [shapes] * n if convert_to_shape_list(shapes) else per(shapes)
        vels, t0s, turns, holds, heads = per(velocity), per(t0), per(turning_speed), per(goal_hold), per(heading)
        is_exact = np.array([bool(e) for e in per(exact)]) & (self.vertex_positions is not None)
        results = [None] * n
        take = lambda xs, idx: [xs[i] for i in idx]
        idx = np.flatnonzero(is_exact).tolist()
        if idx:
            v_parts, e_parts = body_roadmap_hits(self, take(paths, idx), take(shapes, idx), take(vels, idx),
                                                 take(t0s, idx), take(turns, idx), take(holds, idx),
                                                 take(heads, idx), self.STRICT_EPS)
            for i, r in zip(idx, assemble(self, len(idx), v_parts, e_parts, as_arrays)):
                results[i] = r
        idx = np.flatnonzero(~is_exact).tolist()
        if idx:
            sets = [self._piece_set(sweep_rings_timed(paths[i], shapes[i], vels[i], t0s[i], turns[i], holds[i],
                                                      heading=heads[i])) for i in idx]
            hits = self._piece_hits(self._concat_piece_sets(sets))
            for i, r in zip(idx, self._assemble(len(idx), *hits, as_arrays=as_arrays)):
                results[i] = r
        return results

    # Overlapping Interval Paths Step 1: Get each agent piece
    @staticmethod
    def _piece_set(sr: dict) -> dict:
        """One path's pieces (``sweep_rings_timed`` output) plus what the kernels need: per piece its
        segment's speed and reach along travel; per segment its start, unit direction, length and departure."""
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
        """Several agents' piece sets as one: ring and segment indices offset, pieces tagged by ``agent``."""
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
        """Hits of a piece set: ``(agent, vertex, t1, t2), (agent, edge, t1, t2)`` with canonical edge
        indices. Box candidates from the trees, then the numba kernels (inside test / clipping + time window).
        A concave goal footprint goes through Shapely and keeps its whole window. Zero-duration pieces are
        not queried, and zero-duration hits are dropped."""
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
        """Expand edges to their allowed directions, merge each (agent, element)'s windows and split per
        agent (``disc_sweep.assemble``)."""
        return assemble(self, n_agents, [v_hits], [e_hits], as_arrays)

    # ------------------------------------------------------------ check agents overlapping sweeps
    def agent_conflicts(self, paths, shapes, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                        heading=None, pairs=None, first_only: bool = False, as_arrays: bool = False):
        """When two agents' bodies overlap, exactly: ``{(i, j): [(t1, t2), ...]}`` with ``i < j``.
        ``first_only``: ``{(i, j): first contact time}``. ``as_arrays``: ``(i, j, t1, t2)`` arrays.
        ``pairs``: only check these agent pairs. Inputs as in ``overlapping_interval_paths``; no roadmap is
        needed. An agent exists from ``t0`` until it arrives, then for ``goal_hold``. See
        ``polytope_sweep`` for the method."""
        paths = as_path_list(paths)
        if convert_to_shape_list(shapes):
            shapes = [shapes] * len(paths)
        return body_agent_conflicts(paths, shapes, velocity, t0, turning_speed, goal_hold, heading, pairs,
                                    first_only, as_arrays, self.STRICT_EPS)
