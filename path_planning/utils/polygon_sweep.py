"""Convex-polygon footprint support shared by ``CGAL_Sweep`` and ``ShapelySweep``.

The two backends differ only in how they find *candidate* roadmap vertices /
edges near a query region; the exact contact math is geometry-library-agnostic
NumPy. This mixin implements the polygon versions of
``overlapping_graph_elements_cgal`` / ``overlapping_interval_cgal`` once, on
top of a single backend hook:

    _region_candidates(region: ConvexPolygon) -> (vertex_idx, edge_idx)

which must return (possibly over-complete) index arrays of roadmap vertices /
canonical edges that may intersect ``region``; the mixin applies the exact
strict test afterwards.

Geometry. The polygon ``P`` handed in as the ``r`` argument is the combined
footprint ``P_i (+) (-P_j)`` (see ``agent_shape.pair_clearance``) in a fixed
world orientation. For the query agent moving ``c(t) = u + t * vel``:

- a roadmap vertex ``q`` is in contact iff ``q - c(t)`` is strictly inside
  ``P``, i.e. the point ``x(t) = (q - u) - t * vel`` is inside ``P``;
- a roadmap edge ``ab`` is in contact iff ``(a - u) - t * vel`` is inside the
  Minkowski sum ``Q = P (+) segment(0, a - b)``. ``Q`` is convex and is cut
  exactly by ``P``'s half-planes (offsets raised by ``max(0, n . (a - b))``)
  plus the two half-planes perpendicular to the edge, whose offsets are the
  support of ``P`` in those directions.

Both are Cyrus-Beck line clips (``agent_shape.moving_point_interval``), so
contact *sets* are direction-agnostic and contact *intervals* flip as
``t -> tdur - t`` under query reversal -- exactly the symmetry the disc code
exploits, so the same cache / bidirectional-reuse structure is mirrored here.
The exact velocity refinement of the elements query (two simultaneously
moving agents) is likewise the polygon analogue of the disc one and remains
direction-dependent.

Heading-aligned footprints (``AlignedClearance``). When the element agent's
footprint is an ``AlignedShape`` its orientation depends on the roadmap element
it occupies: on a directed edge ``a -> b`` it faces ``b - a``; at a vertex it
faces one of the headings of the edges incident to that vertex (it arrived
along one of them and will leave along one of them -- turns are instantaneous).
The sweep therefore evaluates one combined polygon ``Q (+) (-E(theta))`` per
distinct heading ``theta``, tests every directed edge separately and returns
vertex results keyed ``(vertex_index, heading)``; a self-loop ``(i, i)`` (the
element waiting with unknown heading) uses the element's envelope disc. The
query footprint faces its own direction of travel (``AlignedClearance``
resolves it, including the stationary case). An aligned *query* with a
disc / fixed-polygon element simply resolves to one combined polygon and takes
the fixed path above.

Polygons are planar, so this path is 2D only.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from path_planning.utils.agent_shape import (
    AlignedClearance,
    ConvexPolygon,
    Disc,
    halfplanes_of_vertex_rows,
    heading_key,
    heading_of,
    minkowski_batch,
    minkowski_sum,
    moving_point_interval,
)


class PolygonSweepMixin:
    """Polygon-footprint query implementations; see module docstring."""

    # Backend attributes used here (both backends define them in set_graph):
    #   vertex_positions (n, 2) float, edge_src_array / edge_tgt_array (E,),
    #   edge_indices {idx: (src, tgt)}, edge_directed_map {idx: [(s, t), ...]},
    #   record_sweep, use_exact_collision_check, overlapping_sweep,
    #   overlapping_interval_sweep, segment_spatial_cache, and the helpers
    #   _expand_directed_pairs / _expand_directed_intervals /
    #   convert_bidirectional_interval.

    def _region_candidates(self, region: ConvexPolygon):  # pragma: no cover - abstract
        raise NotImplementedError

    def _check_polygon_supported(self):
        vp = getattr(self, "vertex_positions", None)
        if vp is None:
            raise RuntimeError("set_graph must be called before querying the sweep.")
        if vp.shape[1] != 2:
            raise ValueError(
                f"Polygon footprints are 2D only (roadmap dim={vp.shape[1]}); "
                f"use a float radius (sphere) for 3D sweeps."
            )

    # ------------------------------------------------------------------ geometry

    def _polygon_edge_halfplanes(self, edge_idx: np.ndarray, poly: ConvexPolygon):
        """Half-planes of ``Q_e = P (+) segment(0, a_e - b_e)`` for every
        canonical edge in ``edge_idx``: ``normals (K, m+2, 2)``, ``offsets (K, m+2)``."""
        a = self.vertex_positions[self.edge_src_array[edge_idx]]
        b = self.vertex_positions[self.edge_tgt_array[edge_idx]]
        w = a - b  # (K, 2)
        K = w.shape[0]
        n = poly.normals  # (m, 2)
        d = poly.offsets  # (m,)
        off = d[None, :] + np.maximum(0.0, w @ n.T)  # (K, m)

        wlen = np.linalg.norm(w, axis=1)
        ok = wlen > 1e-12
        safe_len = np.where(ok, wlen, 1.0)
        n_w = np.stack([w[:, 1], -w[:, 0]], axis=1) / safe_len[:, None]
        n_w[~ok] = 0.0  # self-loop edge: Q = P; pad with always-true constraints
        h_plus = np.where(ok, poly.support(n_w), 1.0)
        h_minus = np.where(ok, poly.support(-n_w), 1.0)

        normals = np.concatenate(
            [np.broadcast_to(n[None, :, :], (K,) + n.shape), n_w[:, None, :], -n_w[:, None, :]],
            axis=1,
        )
        offsets = np.concatenate([off, h_plus[:, None], h_minus[:, None]], axis=1)
        return normals, offsets

    def _polygon_hits(self, u_arr: np.ndarray, v_arr: np.ndarray, poly: ConvexPolygon):
        """Exact strict vertex / edge hit index arrays for the sweep ``u -> v``
        (speed-independent: parameterised with ``tdur = 1``)."""
        region = poly.swept(u_arr, v_arr)
        vert_cand, edge_cand = self._region_candidates(region)
        vert_cand = np.asarray(vert_cand, dtype=np.int64).reshape(-1)
        edge_cand = np.asarray(edge_cand, dtype=np.int64).reshape(-1)
        vel = v_arr - u_arr
        if len(vert_cand):
            x0 = self.vertex_positions[vert_cand] - u_arr
            _, _, valid = moving_point_interval(x0, -vel, poly.normals, poly.offsets, 1.0)
            vert_cand = vert_cand[valid]
        if len(edge_cand):
            normals, offsets = self._polygon_edge_halfplanes(edge_cand, poly)
            x0 = self.vertex_positions[self.edge_src_array[edge_cand]] - u_arr
            _, _, valid = moving_point_interval(x0, -vel, normals, offsets, 1.0)
            edge_cand = edge_cand[valid]
        return vert_cand, edge_cand

    def _polygon_point_hits(self, u_arr: np.ndarray, poly: ConvexPolygon):
        """Exact strict vertex / edge hits of the stationary footprint ``P + u``."""
        region = poly.translated(u_arr)
        vert_cand, edge_cand = self._region_candidates(region)
        vert_cand = np.asarray(vert_cand, dtype=np.int64).reshape(-1)
        edge_cand = np.asarray(edge_cand, dtype=np.int64).reshape(-1)
        zero = np.zeros(2)
        if len(vert_cand):
            x0 = self.vertex_positions[vert_cand] - u_arr
            _, _, valid = moving_point_interval(x0, zero, poly.normals, poly.offsets, 1.0)
            vert_cand = vert_cand[valid]
        if len(edge_cand):
            normals, offsets = self._polygon_edge_halfplanes(edge_cand, poly)
            x0 = self.vertex_positions[self.edge_src_array[edge_cand]] - u_arr
            _, _, valid = moving_point_interval(x0, zero, normals, offsets, 1.0)
            edge_cand = edge_cand[valid]
        return vert_cand, edge_cand

    def _polygon_segment_spatial_hits(self, u, v, poly: ConvexPolygon):
        """Direction-agnostic, cached (when ``record_sweep``) hit arrays for the
        sweep ``u -> v`` with footprint ``poly`` (mirror of
        ``_segment_spatial_hits``; callers must not mutate the result)."""
        key = (u, v, poly) if u <= v else (v, u, poly)
        if self.record_sweep and key in self.segment_spatial_cache:
            return self.segment_spatial_cache[key]
        cu = np.asarray(key[0], dtype=float)
        cv = np.asarray(key[1], dtype=float)
        result = self._polygon_hits(cu, cv, poly)
        if self.record_sweep:
            self.segment_spatial_cache[key] = result
        return result

    @staticmethod
    def _query_motion(u_arr, v_arr, velocity):
        """Shared query-agent motion: ``(vel_vec, tdur)`` with the disc backends'
        conventions (``velocity == 0`` -> unit-time traversal)."""
        u_to_v = v_arr - u_arr
        dist_uv = float(np.linalg.norm(u_to_v))
        if velocity == 0.0:
            return u_to_v, 1.0
        if dist_uv > 0.0:
            return velocity * u_to_v / dist_uv, dist_uv / velocity
        return np.zeros_like(u_to_v), 0.0

    # ------------------------------------------------------------------ elements

    def _polygon_overlapping_graph_elements(self, u, v, velocity: float, poly: ConvexPolygon):
        self._check_polygon_supported()
        key = (u, v, velocity, poly)
        if self.record_sweep and key in self.overlapping_sweep:
            return self.overlapping_sweep[key]

        if self.record_sweep and not self.use_exact_collision_check and u != v:
            rev = self.overlapping_sweep.get((v, u, velocity, poly))
            if rev is not None:
                overlapping_edges = set(rev)
                self.overlapping_sweep[key] = overlapping_edges
                return overlapping_edges

        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        overlapping_edges: set[tuple[int, int]] = set()

        if u == v:
            _, edge_hits = self._polygon_point_hits(u_arr, poly)
            for edge_idx in edge_hits:
                overlapping_edges.update(self._expand_directed_pairs(int(edge_idx)))
            if self.record_sweep:
                self.overlapping_sweep[key] = overlapping_edges
            return overlapping_edges

        vertex_hits, edge_hits = self._polygon_segment_spatial_hits(u, v, poly)
        overlapping_vertices = set(int(i) for i in vertex_hits)
        surviving = set(int(i) for i in edge_hits)
        if self.use_exact_collision_check and surviving:
            # Polygon analogue of the disc refinement: drop edges whose
            # hypothetical agent, moving a->b while the query agent moves u->v,
            # never brings the relative position strictly inside the combined
            # footprint within the (relative-motion) duration.
            crossing_idxs = [
                i
                for i in surviving
                if (
                    self.edge_indices[i][0] not in overlapping_vertices
                    or self.edge_indices[i][1] not in overlapping_vertices
                )
            ]
            if crossing_idxs:
                crossing_src = np.array([self.edge_indices[i][0] for i in crossing_idxs], dtype=np.int64)
                crossing_tgt = np.array([self.edge_indices[i][1] for i in crossing_idxs], dtype=np.int64)
                a_pos = self.vertex_positions[crossing_src]
                b_pos = self.vertex_positions[crossing_tgt]

                u_to_v = v_arr - u_arr
                a_to_b = b_pos - a_pos
                ro1 = a_pos - u_arr
                rel = a_to_b - u_to_v
                rel_norm = np.linalg.norm(rel, axis=1)
                if velocity == 0.0:
                    vel = rel
                    tdur = np.ones(rel.shape[0])
                else:
                    safe = rel_norm > 0.0
                    scale = np.zeros_like(rel_norm)
                    scale[safe] = velocity / rel_norm[safe]
                    vel = rel * scale[:, None]
                    tdur = np.where(safe, rel_norm / velocity, 0.0)
                _, _, keep = moving_point_interval(ro1, vel, poly.normals, poly.offsets, tdur)
                for k, i in zip(keep, crossing_idxs):
                    if not k:
                        surviving.discard(i)

        for i in surviving:
            overlapping_edges.update(self._expand_directed_pairs(i))
        if self.record_sweep:
            self.overlapping_sweep[key] = overlapping_edges
        return overlapping_edges

    # ------------------------------------------------------------------ intervals

    def _polygon_overlapping_interval(self, u, v, velocity: float, poly: ConvexPolygon, get_time_interval: bool):
        self._check_polygon_supported()
        key = (u, v, velocity, poly)
        if self.record_sweep and key in self.overlapping_interval_sweep:
            return self.overlapping_interval_sweep[key]

        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)

        # Reverse-query reuse (see module docstring): same hit sets, intervals
        # flipped with t -> tdur - t.
        if self.record_sweep and u != v:
            rev = self.overlapping_interval_sweep.get((v, u, velocity, poly))
            if rev is not None:
                rev_vertices, rev_edges = rev
                if (
                    isinstance(rev_vertices, dict) == get_time_interval
                    and isinstance(rev_edges, dict) == get_time_interval
                ):
                    if get_time_interval:
                        _, tdur = self._query_motion(u_arr, v_arr, velocity)
                        overlapping_vertices = {
                            idx: self.convert_bidirectional_interval(t1, t2, tdur)
                            for idx, (t1, t2) in rev_vertices.items()
                        }
                        overlapping_edges = {
                            k: self.convert_bidirectional_interval(t1, t2, tdur)
                            for k, (t1, t2) in rev_edges.items()
                        }
                    else:
                        overlapping_vertices = set(rev_vertices)
                        overlapping_edges = set(rev_edges)
                    self.overlapping_interval_sweep[key] = (overlapping_vertices, overlapping_edges)
                    return overlapping_vertices, overlapping_edges

        if u == v:
            vertex_hits, edge_hits = self._polygon_point_hits(u_arr, poly)
            if get_time_interval:
                overlapping_vertices = {int(idx): (0.0, float("inf")) for idx in vertex_hits}
                overlapping_edges = {}
                for i in edge_hits:
                    overlapping_edges.update(self._expand_directed_intervals(int(i), 0.0, float("inf"), None))
            else:
                overlapping_vertices = set(int(idx) for idx in vertex_hits)
                overlapping_edges = set()
                for i in edge_hits:
                    overlapping_edges.update(self._expand_directed_pairs(int(i)))
            if self.record_sweep:
                self.overlapping_interval_sweep[key] = (overlapping_vertices, overlapping_edges)
            return overlapping_vertices, overlapping_edges

        vertex_hits, edge_hits = self._polygon_segment_spatial_hits(u, v, poly)
        vel_vec, tdur = self._query_motion(u_arr, v_arr, velocity)

        if not get_time_interval:
            overlapping_vertices = set(int(i) for i in vertex_hits)
            overlapping_edges = set()
            for i in edge_hits:
                overlapping_edges.update(self._expand_directed_pairs(int(i)))
        else:
            overlapping_vertices = {}
            if len(vertex_hits):
                x0 = self.vertex_positions[vertex_hits] - u_arr
                t1, t2, valid = moving_point_interval(x0, -vel_vec, poly.normals, poly.offsets, tdur)
                for i, idx in enumerate(vertex_hits.tolist()):
                    if valid[i]:
                        overlapping_vertices[int(idx)] = (float(t1[i]), float(t2[i]))
            overlapping_edges = {}
            if len(edge_hits):
                normals, offsets = self._polygon_edge_halfplanes(edge_hits, poly)
                x0 = self.vertex_positions[self.edge_src_array[edge_hits]] - u_arr
                t1, t2, valid = moving_point_interval(x0, -vel_vec, normals, offsets, tdur)
                for i, idx in enumerate(edge_hits.tolist()):
                    if valid[i]:
                        overlapping_edges.update(
                            self._expand_directed_intervals(int(idx), float(t1[i]), float(t2[i]), tdur)
                        )

        if self.record_sweep:
            self.overlapping_interval_sweep[key] = (overlapping_vertices, overlapping_edges)
        return overlapping_vertices, overlapping_edges

    # ------------------------------------------------------------------ aligned

    def _aligned_tables(self):
        """Per-graph tables for heading-aligned elements, rebuilt when
        ``set_graph`` changes the vertex array: the headings of the directed
        edges incident to each vertex (CSR ``v_offsets`` / ``v_headings``, both
        directions of every edge) and the directed rows of each canonical edge
        (CSR ``d_offsets`` into ``d_src`` / ``d_tgt`` / ``d_heading``; ``-1``
        marks a self-loop)."""
        vp = self.vertex_positions
        tbl = getattr(self, "_aligned_tbl", None)
        if tbl is not None and tbl.vp is vp:
            return tbl
        n = vp.shape[0]
        inc = [set() for _ in range(n)]
        d_src, d_tgt, d_heading, d_counts = [], [], [], []
        for ei, (s_i, t_i) in enumerate(zip(self.edge_src_array.tolist(), self.edge_tgt_array.tolist())):
            directed = self.edge_directed_map[ei]
            d_counts.append(len(directed))
            for a, b in directed:
                d_src.append(a)
                d_tgt.append(b)
                if a == b:
                    d_heading.append(-1.0)
                else:
                    # Reverse heading via atan2 of the negated vector (not h + pi):
                    # every caller then derives bit-identical keys from the points.
                    h = heading_of(vp[a], vp[b])
                    hr = heading_of(vp[b], vp[a])
                    d_heading.append(h)
                    inc[a].add(h)
                    inc[a].add(hr)
                    inc[b].add(h)
                    inc[b].add(hr)
        counts = np.array([len(x) for x in inc], dtype=np.int64)
        v_offsets = np.zeros(n + 1, dtype=np.int64)
        v_offsets[1:] = np.cumsum(counts)
        d_offsets = np.zeros(len(d_counts) + 1, dtype=np.int64)
        d_offsets[1:] = np.cumsum(np.asarray(d_counts, dtype=np.int64))
        tbl = SimpleNamespace(
            vp=vp,
            v_offsets=v_offsets,
            v_headings=np.array([h for x in inc for h in sorted(x)], dtype=float),
            d_offsets=d_offsets,
            d_src=np.asarray(d_src, dtype=np.int64),
            d_tgt=np.asarray(d_tgt, dtype=np.int64),
            d_heading=np.asarray(d_heading, dtype=float),
        )
        self._aligned_tbl = tbl
        return tbl

    @staticmethod
    def _csr_expand(offsets, idx):
        """Flat row indices of CSR groups ``idx`` and the group each row came from."""
        counts = offsets[idx + 1] - offsets[idx]
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        starts = np.repeat(offsets[idx], counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        return starts + within, np.repeat(idx, counts)

    def _aligned_combos(self, clr: AlignedClearance, query_fp, headings: np.ndarray, outer: ConvexPolygon):
        """Half-planes / vertices of ``Q (+) (-E(h))`` for every heading in
        ``headings`` (``-1`` -> the envelope polygon ``outer``), padded to one
        width: ``normals (K, M, 2)``, ``offsets (K, M)``, ``points (K, M, 2)``."""
        cache = self.__dict__.setdefault("_aligned_combo_cache", {})
        key = (query_fp, clr.element, clr.n_sides, headings.tobytes())
        hit = cache.get(key)
        if hit is not None:
            return hit
        if isinstance(query_fp, Disc):
            query_fp = query_fp.as_polygon(n_sides=clr.n_sides)
        body_refl = clr.element.body.reflected()
        is_loop = headings < 0.0
        K = headings.shape[0]
        m = query_fp.points.shape[0] + body_refl.points.shape[0]
        M = max(m, outer.points.shape[0])
        normals = np.zeros((K, M, 2))
        offsets = np.ones((K, M))
        points = np.zeros((K, M, 2))
        if (~is_loop).any():
            verts = minkowski_batch(query_fp, body_refl, headings[~is_loop])
            n, d = halfplanes_of_vertex_rows(verts)
            normals[~is_loop, :m] = n
            offsets[~is_loop, :m] = d
            points[~is_loop, :m] = verts
            points[~is_loop, m:] = verts[:, :1]
        if is_loop.any():
            k = outer.points.shape[0]
            normals[is_loop, :k] = outer.normals
            offsets[is_loop, :k] = outer.offsets
            points[is_loop, :k] = outer.points
            points[is_loop, k:] = outer.points[0]
        if len(cache) > 4096:
            cache.clear()
        hit = cache[key] = (normals, offsets, points)
        return hit

    def _aligned_rows(self, u_arr, v_arr, clr: AlignedClearance, stationary: bool):
        """Candidate rows of an aligned-element query ``u -> v`` with their
        half-planes: vertex rows ``(vertex, heading)`` tested against
        ``Q (+) (-E(heading))`` and directed-edge rows ``(s, t)`` tested against
        ``Q (+) (-E(heading(s->t))) (+) segment(0, a - b)``."""
        tbl = self._aligned_tables()
        vp = tbl.vp
        motion_heading = None if stationary else heading_of(u_arr, v_arr)
        query_fp = clr.query_footprint(motion_heading)
        outer = minkowski_sum(query_fp, clr.element.envelope, n_sides=clr.n_sides)
        if isinstance(outer, Disc):
            outer = outer.as_polygon(n_sides=clr.n_sides)
        region = outer.translated(u_arr) if stationary else outer.swept(u_arr, v_arr)
        vert_cand, edge_cand = self._region_candidates(region)
        vert_cand = np.asarray(vert_cand, dtype=np.int64).reshape(-1)
        edge_cand = np.asarray(edge_cand, dtype=np.int64).reshape(-1)

        v_rows, row_vertex = self._csr_expand(tbl.v_offsets, vert_cand)
        row_vheading = tbl.v_headings[v_rows]
        d_rows, _ = self._csr_expand(tbl.d_offsets, edge_cand)
        e_src = tbl.d_src[d_rows]
        e_tgt = tbl.d_tgt[d_rows]
        e_heading = tbl.d_heading[d_rows]

        all_headings = np.concatenate([row_vheading, e_heading])
        uniq, inv = np.unique(all_headings, return_inverse=True)
        normals, offsets, points = self._aligned_combos(clr, query_fp, uniq, outer)
        v_inv = inv[: row_vheading.shape[0]]
        e_inv = inv[row_vheading.shape[0]:]

        rows = SimpleNamespace(query_fp=query_fp)
        rows.v_idx = row_vertex
        rows.v_heading = row_vheading
        rows.v_normals = normals[v_inv]
        rows.v_offsets = offsets[v_inv]
        rows.v_x0 = vp[row_vertex] - u_arr

        a = vp[e_src]
        b = vp[e_tgt]
        w = a - b
        cn = normals[e_inv]
        cd = offsets[e_inv]
        cp = points[e_inv]
        off = cd + np.maximum(0.0, np.einsum("rmd,rd->rm", cn, w))
        wlen = np.linalg.norm(w, axis=1)
        ok = wlen > 1e-12
        safe_len = np.where(ok, wlen, 1.0)
        n_w = np.stack([w[:, 1], -w[:, 0]], axis=1) / safe_len[:, None]
        n_w[~ok] = 0.0
        h_plus = np.where(ok, np.max(np.einsum("rd,rkd->rk", n_w, cp), axis=1), 1.0)
        h_minus = np.where(ok, np.max(np.einsum("rd,rkd->rk", -n_w, cp), axis=1), 1.0)
        rows.e_src = e_src
        rows.e_tgt = e_tgt
        rows.e_loop = ~ok
        rows.e_combo_normals = cn
        rows.e_combo_offsets = cd
        rows.e_normals = np.concatenate([cn, n_w[:, None, :], -n_w[:, None, :]], axis=1)
        rows.e_offsets = np.concatenate([off, h_plus[:, None], h_minus[:, None]], axis=1)
        rows.e_x0 = a - u_arr
        return rows

    def _aligned_overlapping_graph_elements(self, u, v, velocity: float, clr: AlignedClearance):
        self._check_polygon_supported()
        if not clr.aligned_element:
            # Only the query is heading-aligned: one fixed combined clearance.
            combined = clr.resolve(None if u == v else heading_of(u, v), None)
            return self.overlapping_graph_elements_cgal(u, v, velocity, combined)
        key = (u, v, velocity, clr)
        if self.record_sweep and key in self.overlapping_sweep:
            return self.overlapping_sweep[key]
        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        stationary = u == v
        rows = self._aligned_rows(u_arr, v_arr, clr, stationary)
        vel = np.zeros(2) if stationary else v_arr - u_arr
        _, _, v_valid = moving_point_interval(rows.v_x0, -vel, rows.v_normals, rows.v_offsets, 1.0)
        _, _, e_valid = moving_point_interval(rows.e_x0, -vel, rows.e_normals, rows.e_offsets, 1.0)
        hit_vertices = set(rows.v_idx[v_valid].tolist())
        keep = e_valid.copy()
        if self.use_exact_collision_check and not stationary and keep.any():
            # Polygon analogue of the disc refinement (see the fixed path): an
            # element agent moving s -> t while the query moves u -> v must bring
            # the relative position strictly inside Q (+) (-E(theta_st)).
            hit_arr = np.fromiter(hit_vertices, dtype=np.int64, count=len(hit_vertices))
            both_hit = np.isin(rows.e_src, hit_arr) & np.isin(rows.e_tgt, hit_arr)
            idx = np.flatnonzero(keep & ~rows.e_loop & ~both_hit)
            if idx.size:
                a_pos = rows.e_x0[idx] + u_arr
                b_pos = self.vertex_positions[rows.e_tgt[idx]]
                rel = (b_pos - a_pos) - (v_arr - u_arr)
                rel_norm = np.linalg.norm(rel, axis=1)
                if velocity == 0.0:
                    rel_vel = rel
                    tdur = np.ones(idx.size)
                else:
                    safe = rel_norm > 0.0
                    scale = np.zeros_like(rel_norm)
                    scale[safe] = velocity / rel_norm[safe]
                    rel_vel = rel * scale[:, None]
                    tdur = np.where(safe, rel_norm / velocity, 0.0)
                _, _, ok = moving_point_interval(
                    rows.e_x0[idx], rel_vel, rows.e_combo_normals[idx], rows.e_combo_offsets[idx], tdur
                )
                keep[idx[~ok]] = False
        overlapping_edges = {
            (int(s_i), int(t_i)) for s_i, t_i in zip(rows.e_src[keep].tolist(), rows.e_tgt[keep].tolist())
        }
        if self.record_sweep:
            self.overlapping_sweep[key] = overlapping_edges
        return overlapping_edges

    def _aligned_overlapping_interval(self, u, v, velocity: float, clr: AlignedClearance, get_time_interval: bool):
        self._check_polygon_supported()
        if not clr.aligned_element:
            combined = clr.resolve(None if u == v else heading_of(u, v), None)
            return self.overlapping_interval_cgal(u, v, velocity, combined, get_time_interval=get_time_interval)
        key = (u, v, velocity, clr)
        if self.record_sweep and key in self.overlapping_interval_sweep:
            return self.overlapping_interval_sweep[key]
        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        stationary = u == v
        rows = self._aligned_rows(u_arr, v_arr, clr, stationary)
        if stationary:
            zero = np.zeros(2)
            _, _, v_valid = moving_point_interval(rows.v_x0, zero, rows.v_normals, rows.v_offsets, 1.0)
            _, _, e_valid = moving_point_interval(rows.e_x0, zero, rows.e_normals, rows.e_offsets, 1.0)
            vt1 = np.zeros(v_valid.shape[0])
            vt2 = np.full(v_valid.shape[0], float("inf"))
            et1 = np.zeros(e_valid.shape[0])
            et2 = np.full(e_valid.shape[0], float("inf"))
        else:
            vel_vec, tdur = self._query_motion(u_arr, v_arr, velocity)
            vt1, vt2, v_valid = moving_point_interval(rows.v_x0, -vel_vec, rows.v_normals, rows.v_offsets, tdur)
            et1, et2, e_valid = moving_point_interval(rows.e_x0, -vel_vec, rows.e_normals, rows.e_offsets, tdur)
        v_keys = list(zip(rows.v_idx.tolist(), rows.v_heading.tolist()))
        e_keys = list(zip(rows.e_src.tolist(), rows.e_tgt.tolist()))
        if get_time_interval:
            overlapping_vertices = {
                (int(vi), float(h)): (float(vt1[i]), float(vt2[i]))
                for i, (vi, h) in enumerate(v_keys) if v_valid[i]
            }
            overlapping_edges = {
                (int(s_i), int(t_i)): (float(et1[i]), float(et2[i]))
                for i, (s_i, t_i) in enumerate(e_keys) if e_valid[i]
            }
        else:
            overlapping_vertices = {(int(vi), float(h)) for i, (vi, h) in enumerate(v_keys) if v_valid[i]}
            overlapping_edges = {(int(s_i), int(t_i)) for i, (s_i, t_i) in enumerate(e_keys) if e_valid[i]}
        if self.record_sweep:
            self.overlapping_interval_sweep[key] = (overlapping_vertices, overlapping_edges)
        return overlapping_vertices, overlapping_edges
