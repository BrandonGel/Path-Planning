"""
Shapely (GEOS) spatial-sweep backend — a drop-in alternative to ``CGAL_Sweep``
(``path_planning/utils/cgal_sweep.py``) for benchmarking.

Mirrors ``CGAL_Sweep``'s public API and return shapes exactly, so it can be
swapped in via ``GraphSampler(sweep_backend="shapely")`` and compared head-to-head
(see ``scripts/roadmaps/benchmark_sweep.py``). The spatial overlap detection
(which vertices/edges a radius-``r`` disk sweeping ``u->v`` touches) uses Shapely's
``STRtree.query(predicate="dwithin")`` instead of rtree + CGAL ``squared_distance``.
The continuous-time interval math and the velocity-based exact refinement are
geometry-library-agnostic NumPy, copied verbatim from ``CGAL_Sweep`` so both
backends return matching intervals — only the spatial layer differs. The
bidirectional-reuse machinery (undirected-canonical spatial-hit cache shared by
both query methods, reverse-query interval flipping via
``convert_bidirectional_interval``) is likewise mirrored between the backends.

Shapely/GEOS is planar, so this backend is 2D only (``set_graph`` raises for 3D).
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import LineString, Point, STRtree


class ShapelySweep:
    """Spatial sweep queries against roadmap vertices/edges using Shapely (GEOS)."""

    def __init__(self, record_sweep: bool = True, use_exact_collision_check: bool = True):
        self.reset()
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check

    def reset(self):
        self.vertices = []
        self.vertex_positions = None
        self.edge_indices = {}
        self.edge_directed_map = {}
        self.overlapping_sweep = {}
        self.overlapping_interval_sweep = {}
        # Undirected-canonical cache of raw spatial candidate hits, shared
        # across query directions and across the elements/interval methods.
        self.segment_spatial_cache = {}
        self.vertex_tree = None
        self.edge_tree = None
        self.edge_src_array = None
        self.edge_tgt_array = None
        self.edge_src_positions = None
        self.edge_tgt_positions = None

    def clear_caches(self):
        """Drop memoized query results but keep the spatial indices / geometry
        (the roadmap is unchanged; only the queries' relevance has expired)."""
        self.overlapping_sweep = {}
        self.overlapping_interval_sweep = {}
        self.segment_spatial_cache = {}

    @staticmethod
    def _undirected_key(src: int, tgt: int) -> tuple[int, int]:
        """Canonical undirected key; self-loops stay ``(i, i)``."""
        if src == tgt:
            return (src, tgt)
        return (src, tgt) if src < tgt else (tgt, src)

    @staticmethod
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

    def _expand_directed_pairs(self, edge_idx: int):
        """Directed ``(src, tgt)`` keys to emit for a canonical spatial hit."""
        return self.edge_directed_map[int(edge_idx)]

    def _expand_directed_intervals(self, edge_idx: int, t1: float, t2: float, tdur=None):
        """Expand a canonical contact interval to all directed edge keys.

        Same query-agent frame for every directed key (undirected capsule).
        ``tdur`` is unused here; reverse-*query* remapping uses
        :meth:`convert_bidirectional_interval`.
        """
        _ = tdur
        edge_idx = int(edge_idx)
        iv = (float(t1), float(t2))
        return {directed: iv for directed in self.edge_directed_map[edge_idx]}

    def set_graph(
        self,
        vertices: list[tuple[float, float]],
        edges: list[tuple[int, int]],
        default_radius: float = 1.0,
    ):
        _ = default_radius  # kept for API compatibility
        self.reset()
        assert len(vertices) > 0, "Vertices cannot be empty"
        assert len(edges) > 0, "Edges cannot be empty"
        dim = len(vertices[0])
        if dim != 2:
            raise ValueError(
                f"ShapelySweep supports 2D only (got dim={dim}); Shapely/GEOS is planar. "
                f"Use the CGAL backend for 3D."
            )

        self.vertex_positions = np.asarray([list(v) for v in vertices], dtype=float)
        # Vertex STRtree (point geometries) for vertex-overlap queries.
        self.vertex_tree = STRtree([Point(p) for p in self.vertex_positions])

        # Deduplicate undirected edges: index geometry once, expand to every
        # directed key that appears in the input. Self-loops (i,i) once each.
        directed_by_undirected: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for e in edges:
            a, b = int(e[0]), int(e[1])
            if a == b:
                continue
            key = self._undirected_key(a, b)
            directed_by_undirected.setdefault(key, set()).add((a, b))

        n_verts = self.vertex_positions.shape[0]
        canonical_edges = list(directed_by_undirected.keys()) + [(i, i) for i in range(n_verts)]
        src_list = [e[0] for e in canonical_edges]
        tgt_list = [e[1] for e in canonical_edges]
        self.edge_src_array = np.asarray(src_list, dtype=np.int64)
        self.edge_tgt_array = np.asarray(tgt_list, dtype=np.int64)
        self.edge_src_positions = self.vertex_positions[self.edge_src_array]
        self.edge_tgt_positions = self.vertex_positions[self.edge_tgt_array]

        for edge_idx, (src, tgt) in enumerate(canonical_edges):
            self.edge_indices[edge_idx] = (src, tgt)
            if src == tgt:
                self.edge_directed_map[edge_idx] = [(src, tgt)]
            else:
                self.edge_directed_map[edge_idx] = sorted(directed_by_undirected[(src, tgt)])

        # Edge STRtree: LineString for proper edges; Point for self-loops
        # (degenerate LineString(p,p) is invalid in GEOS).
        edge_geoms = []
        for s, t in zip(src_list, tgt_list):
            if s == t:
                edge_geoms.append(Point(self.vertex_positions[s]))
            else:
                edge_geoms.append(
                    LineString([self.vertex_positions[s], self.vertex_positions[t]])
                )
        self.edge_tree = STRtree(edge_geoms)

    # ------------------------------------------------------------ spatial helpers

    def _vertices_within(self, geom, r):
        """Indices of roadmap vertices strictly within distance r of ``geom``.

        GEOS ``dwithin`` is inclusive (distance <= r), which would report a vertex
        that merely *touches* the swept disc (distance == r) as a contact. Two
        discs of radius r/2 at exactly distance r are tangent, not colliding
        (the solution checker and the CGAL backend both use strict ``<``), so the
        candidates are post-filtered with an exact strict distance test.
        """
        if self.vertex_tree is None:
            return np.empty(0, dtype=np.int64)
        hits = np.asarray(self.vertex_tree.query(geom, predicate="dwithin", distance=r), dtype=np.int64)
        if len(hits) == 0:
            return hits
        d = shapely.distance(geom, self.vertex_tree.geometries[hits])
        return hits[d < r]

    def _edges_within(self, geom, r):
        """Indices (into the edge arrays) of roadmap edges strictly within distance r of ``geom``
        (see :meth:`_vertices_within` for why the inclusive ``dwithin`` hits are re-filtered)."""
        if self.edge_tree is None:
            return np.empty(0, dtype=np.int64)
        hits = np.asarray(self.edge_tree.query(geom, predicate="dwithin", distance=r), dtype=np.int64)
        if len(hits) == 0:
            return hits
        d = shapely.distance(geom, self.edge_tree.geometries[hits])
        return hits[d < r]

    def _segment_spatial_hits(self, u, v, r):
        """Raw candidate hits ``(vertex_hits, edge_hits)`` for the capsule sweep u->v (u != v).

        Direction-agnostic: the geometry is evaluated in canonical undirected
        orientation and cached (when ``record_sweep``) under the canonical
        ``(min(u, v), max(u, v), r)`` key, so forward/reverse queries and the
        elements/interval methods share one spatial computation. Callers must
        not mutate the returned containers.
        """
        key = (u, v, r) if u <= v else (v, u, r)
        if self.record_sweep and key in self.segment_spatial_cache:
            return self.segment_spatial_cache[key]
        seg = LineString([key[0], key[1]])
        vertex_hits = self._vertices_within(seg, r)
        edge_hits = self._edges_within(seg, r)
        result = (vertex_hits, edge_hits)
        if self.record_sweep:
            self.segment_spatial_cache[key] = result
        return result

    # ------------------------------------------------------------ overlap (edges)

    def overlapping_graph_elements_cgal(
        self, u: tuple[float, float], v: tuple[float, float], velocity: float = 0.0, r: float = 0.5
    ):
        if self.record_sweep and (u, v, velocity, r) in self.overlapping_sweep:
            return self.overlapping_sweep[u, v, velocity, r]

        # Without the direction-dependent exact velocity refinement the result
        # is fully symmetric in (u, v): reuse the reverse query's directed-key
        # set. With the refinement on, only the spatial layer is shared (below).
        if self.record_sweep and not self.use_exact_collision_check and u != v:
            rev = self.overlapping_sweep.get((v, u, velocity, r))
            if rev is not None:
                overlapping_edges = set(rev)
                self.overlapping_sweep[u, v, velocity, r] = overlapping_edges
                return overlapping_edges

        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        overlapping_edges = set()

        # Special case: point query (stationary agent).
        if u == v:
            for edge_idx in self._edges_within(Point(u_arr), r):
                overlapping_edges.update(self._expand_directed_pairs(int(edge_idx)))
            if self.record_sweep:
                self.overlapping_sweep[u, v, velocity, r] = overlapping_edges
            return overlapping_edges

        # Regular segment query (moving agent).
        vertex_hits, edge_hits = self._segment_spatial_hits(u, v, r)
        overlapping_vertices = set[int](int(i) for i in vertex_hits)
        hit_idxs = [int(i) for i in edge_hits]
        # Exact refinement on canonical undirected hits only, then expand.
        surviving = set(hit_idxs)
        if self.use_exact_collision_check and surviving:
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

                vel_dot_vel = np.sum(vel * vel, axis=1)
                vel_dot_ro1 = np.sum(vel * ro1, axis=1)
                tmin = np.clip(-vel_dot_ro1 / (vel_dot_vel + 1e-10), 0.0, tdur)
                vec = ro1 + vel * tmin[:, None]
                miss = np.sum(vec * vec, axis=1) > r * r
                for keep, i in zip(~miss, crossing_idxs):
                    if not keep:
                        surviving.discard(i)

        for i in surviving:
            overlapping_edges.update(self._expand_directed_pairs(i))

        if self.record_sweep:
            self.overlapping_sweep[u, v, velocity, r] = overlapping_edges
        return overlapping_edges

    # ------------------------------------------------------------ interval math

    def get_interval_from_quadratic_equation(self, r0: np.ndarray, vel: np.ndarray, r: float, tdur: float):
        """Solve ||r0 + t*vel||^2 = r^2 for t in [0, tdur]. Copied from CGAL_Sweep.

        Returns ``(t1, t2, valid)``. ``valid[i]`` is False when the relative motion never
        brings the point strictly inside radius r (no real roots, e.g. pure tangency);
        such rows carry the placeholder window ``(0, tdur)`` and must be ignored by
        callers -- the inclusive spatial pre-query can hand in tangent candidates."""
        r0 = np.asarray(r0)
        vel = np.asarray(vel)
        if r0.ndim == 1:
            r0 = r0.reshape(1, -1)

        if vel.ndim == 1 or (vel.ndim == 2 and vel.shape[0] == 1):
            v = vel.reshape(-1)
            a = np.dot(v, v)
            b = 2.0 * (r0 @ v)
            c = np.einsum("ij,ij->i", r0, r0) - r**2 + 1e-10
            disc = b**2 - 4.0 * a * c
            t1 = np.zeros_like(b, dtype=float)
            t2 = tdur * np.ones_like(b, dtype=float)
            tdur_arr = np.broadcast_to(tdur, b.shape).astype(float)
            # Stationary relative motion (a == 0): inside iff c < 0 for all t.
            valid = np.broadcast_to(c < 0.0, b.shape).copy()
            if a > 0.0:
                valid = disc >= 0.0
                if np.any(valid):
                    sqrt_disc = np.sqrt(disc[valid])
                    t1_raw = (-b[valid] - sqrt_disc) / (2.0 * a)
                    t2_raw = (-b[valid] + sqrt_disc) / (2.0 * a) + 1e-9
                    t1[valid] = np.clip(t1_raw, 0.0, tdur_arr[valid])
                    t2[valid] = np.clip(t2_raw, 0.0, tdur_arr[valid])
        elif vel.ndim == 2:
            if vel.shape != r0.shape:
                raise ValueError("For per-row velocities, r0 and vel must have the same shape.")
            a = np.einsum("ij,ij->i", vel, vel)
            b = 2.0 * np.einsum("ij,ij->i", vel, r0)
            c = np.einsum("ij,ij->i", r0, r0) - r**2 + 1e-10
            disc = b**2 - 4.0 * a * c
            t1 = np.zeros_like(a, dtype=float)
            t2 = np.zeros_like(a, dtype=float)
            tdur_arr = np.broadcast_to(tdur, a.shape).astype(float)
            moving = a > 0.0
            valid = moving & (disc >= 0.0)
            stationary_inside = (~moving) & (c < 0.0)
            t2[stationary_inside] = tdur_arr[stationary_inside]
            valid = valid | stationary_inside
            if np.any(valid):
                sqrt_disc = np.sqrt(disc[valid])
                t1_raw = (-b[valid] - sqrt_disc) / (2.0 * a[valid])
                t2_raw = (-b[valid] + sqrt_disc) / (2.0 * a[valid]) + 1e-9
                t1[valid] = np.clip(t1_raw, 0.0, tdur_arr[valid])
                t2[valid] = np.clip(t2_raw, 0.0, tdur_arr[valid])
        else:
            raise ValueError("vel must be either a 1D or 2D array.")
        return t1, t2, valid

    def overlapping_interval_cgal(
        self,
        u: tuple[float, float],
        v: tuple[float, float],
        velocity: float = 0.0,
        r: float = 0.5,
        get_time_interval: bool = False,
    ):
        if self.record_sweep and (u, v, velocity, r) in self.overlapping_interval_sweep:
            return self.overlapping_interval_sweep[u, v, velocity, r]

        # Reverse-query reuse: the spatial hit set is symmetric in (u, v) and
        # contact intervals against static geometry map onto the reverse
        # traversal exactly via t -> tdur - t (same tdur both ways; the
        # velocity == 0 case uses the tdur = 1.0 parameterization in both
        # directions). Flipped intervals sit 1e-9 below directly-computed ones
        # (the one-sided exit slack in get_interval_from_quadratic_equation
        # lands on the entry side after flipping) — far below downstream
        # tolerances.
        if self.record_sweep and u != v:
            rev = self.overlapping_interval_sweep.get((v, u, velocity, r))
            if rev is not None:
                rev_vertices, rev_edges = rev
                # The cache key omits get_time_interval, so only reuse when
                # the cached container shape matches the request.
                if (
                    isinstance(rev_vertices, dict) == get_time_interval
                    and isinstance(rev_edges, dict) == get_time_interval
                ):
                    if get_time_interval:
                        dist_uv = float(
                            np.linalg.norm(np.asarray(v, dtype=float) - np.asarray(u, dtype=float))
                        )
                        if velocity == 0.0:
                            tdur = 1.0
                        elif dist_uv > 0.0:
                            tdur = dist_uv / velocity
                        else:
                            tdur = 0.0
                        overlapping_vertices = {
                            idx: self.convert_bidirectional_interval(t1, t2, tdur)
                            for idx, (t1, t2) in rev_vertices.items()
                        }
                        overlapping_edges = {
                            key: self.convert_bidirectional_interval(t1, t2, tdur)
                            for key, (t1, t2) in rev_edges.items()
                        }
                    else:
                        overlapping_vertices = set(rev_vertices)
                        overlapping_edges = set(rev_edges)
                    self.overlapping_interval_sweep[u, v, velocity, r] = (
                        overlapping_vertices,
                        overlapping_edges,
                    )
                    return overlapping_vertices, overlapping_edges

        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)

        # Special case: point query (stationary agent).
        if u == v:
            v_hits = self._vertices_within(Point(u_arr), r - 1e-10)
            if get_time_interval:
                overlapping_vertices = {int(idx): (0.0, float("inf")) for idx in v_hits}
            else:
                overlapping_vertices = set(int(idx) for idx in v_hits)

            e_hits = self._edges_within(Point(u_arr), r)
            if len(e_hits):
                if get_time_interval:
                    overlapping_edges = {}
                    for i in e_hits:
                        overlapping_edges.update(
                            self._expand_directed_intervals(int(i), 0.0, float("inf"), None)
                        )
                else:
                    overlapping_edges = set()
                    for i in e_hits:
                        overlapping_edges.update(self._expand_directed_pairs(int(i)))
            else:
                overlapping_edges = {} if get_time_interval else set()

            if self.record_sweep:
                self.overlapping_interval_sweep[u, v, velocity, r] = (
                    overlapping_vertices,
                    overlapping_edges,
                )
            return overlapping_vertices, overlapping_edges

        # Regular segment query (moving agent).
        vertex_hits, edge_hits = self._segment_spatial_hits(u, v, r)

        u_to_v = v_arr - u_arr
        dist_uv = float(np.linalg.norm(u_to_v))
        if velocity == 0.0:
            vel_vec = u_to_v
            tdur = 1.0
        elif dist_uv > 0.0:
            vel_vec = velocity * u_to_v / dist_uv
            tdur = dist_uv / velocity
        else:
            vel_vec = np.zeros_like(u_to_v)
            tdur = 0.0

        # Dense per-vertex interval tables used for vectorized endpoint gather
        # when get_time_interval=True (None / unused otherwise).
        vtx_t_lo = None
        vtx_t_hi = None
        if len(vertex_hits):
            if get_time_interval:
                r0 = u_arr - self.vertex_positions[vertex_hits]
                t1, t2, valid = self.get_interval_from_quadratic_equation(r0, vel_vec, r, tdur)
                # Tangent / never-inside candidates are not contacts: leave them out
                # of the vertex windows and of the endpoint-sphere tables below.
                vertex_hits = vertex_hits[valid]
                t1 = t1[valid]
                t2 = t2[valid]
                vtx_t_lo = np.full(self.vertex_positions.shape[0], np.inf)
                vtx_t_hi = np.full(self.vertex_positions.shape[0], -np.inf)
                vtx_t_lo[vertex_hits] = t1
                vtx_t_hi[vertex_hits] = t2
                overlapping_vertices = {
                    int(idx): (float(t1[i]), float(t2[i])) for i, idx in enumerate(vertex_hits.tolist())
                }
            else:
                overlapping_vertices = set(int(i) for i in vertex_hits)
        else:
            overlapping_vertices = {} if get_time_interval else set()

        if not len(edge_hits):
            overlapping_edges = {} if get_time_interval else set()
        elif not get_time_interval:
            overlapping_edges = set()
            for i in edge_hits:
                overlapping_edges.update(self._expand_directed_pairs(int(i)))
        else:
            src_indices = self.edge_src_array[edge_hits]
            tgt_indices = self.edge_tgt_array[edge_hits]
            a_pos = self.vertex_positions[src_indices]
            b_pos = self.vertex_positions[tgt_indices]
            a_to_b = b_pos - a_pos
            K = a_pos.shape[0]

            # Endpoint spheres: vectorized gather of overlapping-vertex intervals.
            # Non-hit vertices stay at ±inf and do not contribute.
            if vtx_t_lo is not None:
                all_starts = np.minimum(vtx_t_lo[src_indices], vtx_t_lo[tgt_indices])
                all_ends = np.maximum(vtx_t_hi[src_indices], vtx_t_hi[tgt_indices])
            else:
                all_starts = np.full(K, np.inf)
                all_ends = np.full(K, -np.inf)

            # Finite cylinder contact: first-hit / leave times are the intersection of
            # (1) times when perpendicular distance to line(ab) ≤ r, and
            # (2) times when the projection of the agent onto ab lies in [0, 1].
            seg_len_sq = np.sum(a_to_b * a_to_b, axis=1) + 1e-12
            vdot = (a_to_b @ vel_vec) / seg_len_sq
            v_perp = vel_vec[None, :] - vdot[:, None] * a_to_b
            rel_pos_u = u_arr - a_pos
            pdot = np.sum(rel_pos_u * a_to_b, axis=1) / seg_len_sq
            pos_perp = rel_pos_u - pdot[:, None] * a_to_b

            A_c = np.sum(v_perp * v_perp, axis=1)
            B_c = 2.0 * np.sum(v_perp * pos_perp, axis=1)
            C_c = np.sum(pos_perp * pos_perp, axis=1) - r * r

            LARGE = 1e18
            moving_perp = A_c > 1e-12
            disc_c = B_c * B_c - 4.0 * A_c * C_c
            has_perp_roots = moving_perp & (disc_c >= 0.0)
            sqrt_disc_c = np.sqrt(np.maximum(0.0, disc_c))
            safe_A = np.where(moving_perp, A_c, 1.0)
            # Enter / exit the infinite cylinder (perp distance = r).
            t_perp_enter = np.where(has_perp_roots, (-B_c - sqrt_disc_c) / (2.0 * safe_A), LARGE)
            t_perp_exit = np.where(has_perp_roots, (-B_c + sqrt_disc_c) / (2.0 * safe_A), -LARGE)
            # Parallel motion already inside radius: stay inside for all time.
            stationary_inside = (~moving_perp) & (C_c <= 0.0)
            t_perp_enter = np.where(stationary_inside, -LARGE, t_perp_enter)
            t_perp_exit = np.where(stationary_inside, LARGE, t_perp_exit)

            # Enter / exit the finite segment via projection s(t) = pdot + t*vdot ∈ [0, 1].
            moving_proj = np.abs(vdot) > 1e-12
            safe_vdot = np.where(moving_proj, vdot, 1.0)
            t_at_a = -pdot / safe_vdot
            t_at_b = (1.0 - pdot) / safe_vdot
            t_proj_enter = np.minimum(t_at_a, t_at_b)
            t_proj_exit = np.maximum(t_at_a, t_at_b)
            proj_const_in = (~moving_proj) & (pdot >= 0.0) & (pdot <= 1.0)
            t_proj_enter = np.where(moving_proj, t_proj_enter, np.where(proj_const_in, -LARGE, LARGE))
            t_proj_exit = np.where(moving_proj, t_proj_exit, np.where(proj_const_in, LARGE, -LARGE))

            # First hit / leave the finite cylinder = intersection of the two intervals.
            t_first_hit = np.maximum(t_perp_enter, t_proj_enter)
            t_leave = np.minimum(t_perp_exit, t_proj_exit)
            has_cyl = t_first_hit < t_leave
            all_starts = np.where(has_cyl, np.minimum(all_starts, t_first_hit), all_starts)
            all_ends = np.where(has_cyl, np.maximum(all_ends, t_leave), all_ends)

            tau_start = np.clip(all_starts, 0.0, tdur)
            tau_end = np.clip(all_ends, 0.0, tdur)
            no_collision = (tau_start >= tau_end) | np.isinf(all_starts)
            overlapping_edges = {}
            for i in range(K):
                if no_collision[i]:
                    continue
                overlapping_edges.update(
                    self._expand_directed_intervals(
                        int(edge_hits[i]), float(tau_start[i]), float(tau_end[i]), tdur
                    )
                )

        if self.record_sweep:
            self.overlapping_interval_sweep[u, v, velocity, r] = (
                overlapping_vertices,
                overlapping_edges,
            )
        return overlapping_vertices, overlapping_edges
