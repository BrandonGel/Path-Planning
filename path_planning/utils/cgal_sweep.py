from CGAL.CGAL_Kernel import (
    Point_2,
    Point_3,
    Segment_2,
    Segment_3,
    Vector_2,
    Vector_3,
    squared_distance,
)
import numpy as np
from scipy.spatial import KDTree
from rtree import index


class CGAL_Sweep:
    """Spatial sweep queries against roadmap vertices/edges using CGAL primitives."""

    def __init__(self, record_sweep: bool = True, use_exact_collision_check: bool = True):
        self.reset()
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check

    def reset(self):
        self.Point_type = None
        self.Segment_type = None
        self.Zero_Vector = None
        self.vertices = []
        self.vertex_positions = None
        self.edges = []
        self.edge_indices = {}
        self.edge_directed_map = {}
        self.overlapping_sweep = {}
        self.overlapping_interval_sweep = {}
        # Undirected-canonical cache of raw spatial candidate hits, shared
        # across query directions and across the elements/interval methods.
        self.segment_spatial_cache = {}
        self.vertex_kdtree = None
        self.edge_aabbs = []
        self.edge_rtree = None

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
        if dim == 2:
            self.Point_type = Point_2
            self.Segment_type = Segment_2
            self.Zero_Vector = Vector_2(0.0, 0.0)
        elif dim == 3:
            self.Point_type = Point_3
            self.Segment_type = Segment_3
            self.Zero_Vector = Vector_3(0.0, 0.0, 0.0)
        else:
            raise ValueError(f"Unsupported point dimension: {dim}. Expected 2 or 3.")
        for vertex in vertices:
            p_pt = self.Point_type(*vertex)
            self.vertices.append(p_pt)

        # Build KDTree for vertex queries
        self.vertex_positions = np.array([list(v) for v in vertices])
        self.vertex_kdtree = KDTree(self.vertex_positions)

        # Deduplicate undirected edges: index geometry once, expand to every
        # directed key that appears in the input. Self-loops (i,i) once each.
        directed_by_undirected: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for e in edges:
            a, b = int(e[0]), int(e[1])
            if a == b:
                continue
            key = self._undirected_key(a, b)
            directed_by_undirected.setdefault(key, set()).add((a, b))

        n_verts = len(vertices)
        all_edges = list(directed_by_undirected.keys()) + [(i, i) for i in range(n_verts)]
        for edge in all_edges:
            src, tgt = edge
            a_pt = self.Point_type(*vertices[src])
            b_pt = self.Point_type(*vertices[tgt])
            self.edges.append(self.Segment_type(a_pt, b_pt))
            edge_idx = len(self.edges) - 1
            self.edge_indices[edge_idx] = (src, tgt)
            if src == tgt:
                self.edge_directed_map[edge_idx] = [(src, tgt)]
            else:
                self.edge_directed_map[edge_idx] = sorted(directed_by_undirected[(src, tgt)])

            # Store edge bounding box (will be expanded by query radius at query time)
            v1 = self.vertex_positions[src]
            v2 = self.vertex_positions[tgt]
            bbox_min = np.minimum(v1, v2)
            bbox_max = np.maximum(v1, v2)
            self.edge_aabbs.append((bbox_min, bbox_max, edge_idx))

        # Vectorized lookup arrays: avoid per-query list comprehensions over edge_indices.
        self.edge_src_array = np.array([e[0] for e in all_edges], dtype=np.int64)
        self.edge_tgt_array = np.array([e[1] for e in all_edges], dtype=np.int64)
        # Stack endpoint positions for vectorized distance/interval math.
        self.edge_src_positions = self.vertex_positions[self.edge_src_array]
        self.edge_tgt_positions = self.vertex_positions[self.edge_tgt_array]

        # Build R-tree for efficient spatial edge queries.
        p = index.Property()
        p.dimension = dim
        self.edge_rtree = index.Index(properties=p)

        for edge_idx, (bbox_min, bbox_max, _) in enumerate(self.edge_aabbs):
            if dim == 2:
                bbox = (bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1])
            else:
                bbox = (
                    bbox_min[0],
                    bbox_min[1],
                    bbox_min[2],
                    bbox_max[0],
                    bbox_max[1],
                    bbox_max[2],
                )
            self.edge_rtree.insert(edge_idx, bbox)

    def _query_vertices_on_segment(self, u_arr, v_arr, r):
        """Find vertices within distance r of segment u->v.

        Uses the tight enclosing sphere of the capsule (radius = seg_len/2 + r,
        centered at the segment midpoint) as a KDTree ball query, then filters
        with a vectorized numpy point-to-segment distance.
        """
        u_flat = u_arr.ravel()
        v_flat = v_arr.ravel()
        diff = v_flat - u_flat
        seg_half_len = 0.5 * float(np.linalg.norm(diff))
        center = 0.5 * (u_flat + v_flat)
        candidates = self.vertex_kdtree.query_ball_point(center, seg_half_len + r)
        if not candidates:
            return []
        cand = np.asarray(candidates, dtype=np.int64)
        pts = self.vertex_positions[cand]  # (N, d)
        uv_len_sq = float(diff @ diff)
        if uv_len_sq < 1e-20:
            d2 = np.sum((pts - u_flat) ** 2, axis=1)
        else:
            s = np.clip(((pts - u_flat) @ diff) / uv_len_sq, 0.0, 1.0)
            proj = u_flat + s[:, None] * diff
            d2 = np.sum((pts - proj) ** 2, axis=1)
        mask = d2 < r * r
        return cand[mask].tolist()

    def _build_query_bbox(self, u_arr, v_arr, r):
        """Build R-tree bbox tuple (min_x, min_y[, min_z], max_x, max_y[, max_z]) for segment expanded by r."""
        query_min = np.minimum(u_arr, v_arr).ravel() - r
        query_max = np.maximum(u_arr, v_arr).ravel() + r
        return (*query_min, *query_max)

    def _edges_within_r_of_segment(self, u_arr, v_arr, r, candidate_ids):
        """Vectorized minimum-distance check between segment u->v and a list of candidate edges.

        Returns the subset of candidate edge indices whose minimum distance to
        the query segment is < r. Uses the standard clamped closest-points-of-
        two-segments formulation, evaluated in numpy across all candidates at
        once.
        """
        if not candidate_ids:
            return []
        cand = np.asarray(candidate_ids, dtype=np.int64)
        u = u_arr.ravel()
        v = v_arr.ravel()
        a = self.edge_src_positions[cand]   # (N, d)
        b = self.edge_tgt_positions[cand]   # (N, d)
        d1 = (v - u)                        # (d,)
        d2 = b - a                          # (N, d)
        r_vec = u - a                       # (N, d)
        eps = 1e-12

        a_qq = float(d1 @ d1)                                # ||d1||^2 scalar
        e = np.einsum("ij,ij->i", d2, d2)                    # ||d2_i||^2 (N,)
        f = np.einsum("ij,ij->i", d2, r_vec)                 # d2 . r_vec   (N,)

        s_param = np.zeros(cand.shape[0])
        t_param = np.zeros(cand.shape[0])

        if a_qq <= eps:
            # Query segment is a point; closest point on each edge.
            t_param = np.where(e > eps, np.clip(f / np.where(e > eps, e, 1.0), 0.0, 1.0), 0.0)
        else:
            c_dot = r_vec @ d1                               # d1 . r_vec   (N,)
            b_dot = d2 @ d1                                  # d1 . d2_i    (N,)
            denom = a_qq * e - b_dot * b_dot                 # (N,)

            # Non-parallel case.
            non_parallel = denom > eps
            s_raw = np.zeros(cand.shape[0])
            s_raw[non_parallel] = np.clip(
                (b_dot[non_parallel] * f[non_parallel] - c_dot[non_parallel] * e[non_parallel])
                / denom[non_parallel],
                0.0,
                1.0,
            )
            # Parallel case: pick s=0 and recompute t from that.
            s_param = s_raw

            t_raw = (b_dot * s_param + f) / np.where(e > eps, e, 1.0)
            t_clipped = np.clip(t_raw, 0.0, 1.0)
            t_degenerate = e <= eps
            t_param = np.where(t_degenerate, 0.0, t_clipped)

            # If t got clipped, recompute s.
            recompute = (t_raw != t_clipped) & non_parallel
            if np.any(recompute):
                s_recalc = np.clip(
                    (b_dot[recompute] * t_param[recompute] - c_dot[recompute]) / a_qq,
                    0.0,
                    1.0,
                )
                s_param[recompute] = s_recalc

            # Parallel fallback: the single-projection formula misses overlaps,
            # shared-endpoint, and parallel-shifted cases. Use the canonical
            # 4-endpoint projection: project each segment's endpoints onto the
            # other segment (clamped to [0,1]) and take the minimum distance.
            parallel = ~non_parallel
            if np.any(parallel):
                a_p = a[parallel]
                b_p = b[parallel]
                d2_p = d2[parallel]
                e_p = e[parallel]
                safe_e = np.where(e_p > eps, e_p, 1.0)

                # Project u and v onto each parallel edge (a, b).
                tu = np.clip(np.einsum("ij,ij->i", (u - a_p), d2_p) / safe_e, 0.0, 1.0)
                tv = np.clip(np.einsum("ij,ij->i", (v - a_p), d2_p) / safe_e, 0.0, 1.0)
                pu = a_p + tu[:, None] * d2_p
                pv = a_p + tv[:, None] * d2_p
                d_u_sq = np.sum((u - pu) ** 2, axis=1)
                d_v_sq = np.sum((v - pv) ** 2, axis=1)

                # Project a and b onto the query segment (u, v).
                if a_qq > eps:
                    sa = np.clip(((a_p - u) @ d1) / a_qq, 0.0, 1.0)
                    sb = np.clip(((b_p - u) @ d1) / a_qq, 0.0, 1.0)
                else:
                    sa = np.zeros(a_p.shape[0])
                    sb = np.zeros(a_p.shape[0])
                pa = u + sa[:, None] * d1
                pb = u + sb[:, None] * d1
                d_a_sq = np.sum((a_p - pa) ** 2, axis=1)
                d_b_sq = np.sum((b_p - pb) ** 2, axis=1)

                # Pick the minimum distance and the corresponding (s, t).
                d_all = np.stack([d_u_sq, d_v_sq, d_a_sq, d_b_sq], axis=1)
                best = np.argmin(d_all, axis=1)
                # Build s/t for the best choice per row.
                # best=0: (s=0,    t=tu)
                # best=1: (s=1,    t=tv)
                # best=2: (s=sa,   t=0)
                # best=3: (s=sb,   t=1)
                s_par = np.where(
                    best == 0, 0.0,
                    np.where(best == 1, 1.0, np.where(best == 2, sa, sb)),
                )
                t_par = np.where(
                    best == 0, tu,
                    np.where(best == 1, tv, np.where(best == 2, 0.0, 1.0)),
                )
                s_param[parallel] = s_par
                t_param[parallel] = t_par

        # Closest points and distance.
        p1 = u + s_param[:, None] * d1      # (N, d)
        p2 = a + t_param[:, None] * d2      # (N, d)
        d2_arr = np.sum((p1 - p2) ** 2, axis=1)
        mask = d2_arr < r * r
        return cand[mask].tolist()

    def _build_point_query_bbox(self, point: tuple[float, ...], r: float):
        point_arr = np.asarray(point, dtype=float)
        query_min = point_arr - r
        query_max = point_arr + r
        if point_arr.shape[0] == 2:
            return (query_min[0], query_min[1], query_max[0], query_max[1])
        return (
            query_min[0],
            query_min[1],
            query_min[2],
            query_max[0],
            query_max[1],
            query_max[2],
        )

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
        cu = np.asarray(key[0], dtype=float)
        cv = np.asarray(key[1], dtype=float)
        vertex_hits = self._query_vertices_on_segment(cu, cv, r)
        candidate_edge_indices = list(
            self.edge_rtree.intersection(self._build_query_bbox(cu, cv, r))
        )
        edge_hits = self._edges_within_r_of_segment(cu, cv, r, candidate_edge_indices)
        result = (vertex_hits, edge_hits)
        if self.record_sweep:
            self.segment_spatial_cache[key] = result
        return result

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

        # Special case: point query (stationary agent)
        if u == v:
            u_pt = self.Point_type(*u)
            query_bbox = self._build_point_query_bbox(u, r)
            candidate_edge_indices = list(self.edge_rtree.intersection(query_bbox))
            for edge_idx in candidate_edge_indices:
                if squared_distance(u_pt, self.edges[edge_idx]) ** 0.5 < r:
                    overlapping_edges.update(self._expand_directed_pairs(int(edge_idx)))

            if self.record_sweep:
                self.overlapping_sweep[u, v, velocity, r] = overlapping_edges
            return overlapping_edges

        # Regular segment query (moving agent): shared symmetric spatial layer.
        vertex_hits, edge_hits = self._segment_spatial_hits(u, v, r)
        overlapping_vertices = set(vertex_hits)
        # Exact refinement on canonical undirected hits only, then expand.
        surviving = set(int(i) for i in edge_hits)
        if self.use_exact_collision_check and surviving:
            # Drop edges where the two moving agents (with relative velocity)
            # never come within r of each other within the duration tdur.
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

                u_to_v = v_arr - u_arr                # (d,)
                a_to_b = b_pos - a_pos                # (N, d)
                ro1 = a_pos - u_arr                   # (N, d)

                rel = a_to_b - u_to_v                 # (N, d)
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
                tmin = np.clip(
                    -vel_dot_ro1 / (vel_dot_vel + 1e-10), 0.0, tdur
                )
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

    def get_interval_from_quadratic_equation(self, r0: np.ndarray, vel: np.ndarray, r: float, tdur: float):
        """
        Solve ||r0 + t * vel||^2 = r^2 for t in [0, tdur].

        Supports:
        - r0: (N, d), vel: (d,)    -> shared velocity for all rows, scalar tdur or broadcastable
        - r0: (N, d), vel: (N, d)  -> per-row velocity, tdur scalar or (N,)
        """
        r0 = np.asarray(r0)
        vel = np.asarray(vel)

        if r0.ndim == 1:
            r0 = r0.reshape(1, -1)

        # Case 1: shared velocity vector for all rows.
        # Accept both (d,) and (1, d) to preserve previous behavior.
        if vel.ndim == 1 or (vel.ndim == 2 and vel.shape[0] == 1):
            v = vel.reshape(-1)                  # (d,)
            a = np.dot(v, v)                    # scalar

            b = 2.0 * (r0 @ v)                  # (N,)
            c = np.einsum('ij,ij->i', r0, r0) - r**2 + 1e-10  # (N,)

            disc = b**2 - 4.0 * a * c           # (N,)

            t1 = np.zeros_like(b, dtype=float)
            t2 = tdur*np.ones_like(b, dtype=float)
            tdur_arr = np.broadcast_to(tdur, b.shape).astype(float)

            if a > 0.0:
                valid = disc >= 0.0
                if np.any(valid):
                    sqrt_disc = np.sqrt(disc[valid])
                    t1_raw = (-b[valid] - sqrt_disc) / (2.0 * a)
                    t2_raw = (-b[valid] + sqrt_disc) / (2.0 * a) + 1e-9
                    t1[valid] = np.clip(t1_raw, 0.0, tdur_arr[valid])
                    t2[valid] = np.clip(t2_raw, 0.0, tdur_arr[valid])
            # For a == 0, we keep the default [0, tdur] interval.

        # Case 2: per-row velocity and duration
        elif vel.ndim == 2:
            if vel.shape != r0.shape:
                raise ValueError("For per-row velocities, r0 and vel must have the same shape.")

            a = np.einsum('ij,ij->i', vel, vel)  # (N,)
            b = 2.0 * np.einsum('ij,ij->i', vel, r0)
            c = np.einsum('ij,ij->i', r0, r0) - r**2 + 1e-10 

            disc = b**2 - 4.0 * a * c

            t1 = np.zeros_like(a, dtype=float)
            t2 = np.zeros_like(a, dtype=float)
            tdur_arr = np.broadcast_to(tdur, a.shape).astype(float)

            moving = a > 0.0
            valid = moving & (disc >= 0.0)
            if np.any(valid):
                sqrt_disc = np.sqrt(disc[valid])
                t1_raw = (-b[valid] - sqrt_disc) / (2.0 * a[valid])
                t2_raw = (-b[valid] + sqrt_disc) / (2.0 * a[valid]) + 1e-9
                t1[valid] = np.clip(t1_raw, 0.0, tdur_arr[valid])
                t2[valid] = np.clip(t2_raw, 0.0, tdur_arr[valid])

        else:
            raise ValueError("vel must be either a 1D or 2D array.")

        return t1, t2

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

        # Special case: point query (stationary agent)
        if u == v:
            indices = self.vertex_kdtree.query_ball_point(u, r - 1e-10)
            if get_time_interval:
                overlapping_vertices = {idx: (0.0, float("inf")) for idx in indices}
            else:
                overlapping_vertices = set(indices)

            u_pt = self.Point_type(*u)
            query_bbox = self._build_point_query_bbox(u, r)
            candidate_edge_indices = [
                edge_idx
                for edge_idx in self.edge_rtree.intersection(query_bbox)
                if squared_distance(u_pt, self.edges[edge_idx]) ** 0.5 < r
            ]
            if candidate_edge_indices:
                if get_time_interval:
                    overlapping_edges = {}
                    for i in candidate_edge_indices:
                        overlapping_edges.update(
                            self._expand_directed_intervals(int(i), 0.0, float("inf"), None)
                        )
                else:
                    overlapping_edges = set()
                    for i in candidate_edge_indices:
                        overlapping_edges.update(self._expand_directed_pairs(int(i)))
            else:
                overlapping_edges = {} if get_time_interval else set()

            if self.record_sweep:
                self.overlapping_interval_sweep[u, v, velocity, r] = (
                    overlapping_vertices,
                    overlapping_edges,
                )
            return overlapping_vertices, overlapping_edges

        # Regular segment query (moving agent): shared symmetric spatial layer.
        vertex_hits, edge_hits = self._segment_spatial_hits(u, v, r)

        # Compute the (shared) motion of the query agent once.
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
        if vertex_hits:
            if get_time_interval:
                vert_idx = np.asarray(vertex_hits, dtype=np.int64)
                r0 = u_arr - self.vertex_positions[vert_idx]
                t1, t2 = self.get_interval_from_quadratic_equation(r0, vel_vec, r, tdur)
                vtx_t_lo = np.full(self.vertex_positions.shape[0], np.inf)
                vtx_t_hi = np.full(self.vertex_positions.shape[0], -np.inf)
                vtx_t_lo[vert_idx] = t1
                vtx_t_hi[vert_idx] = t2
                overlapping_vertices = {
                    int(idx): (float(t1[i]), float(t2[i])) for i, idx in enumerate(vert_idx.tolist())
                }
            else:
                overlapping_vertices = set(vertex_hits)
        else:
            overlapping_vertices = {} if get_time_interval else set()

        if not edge_hits:
            overlapping_edges = {} if get_time_interval else set()
        elif not get_time_interval:
            overlapping_edges = set()
            for i in edge_hits:
                overlapping_edges.update(self._expand_directed_pairs(int(i)))
        else:
            cand = np.asarray(edge_hits, dtype=np.int64)
            src_indices = self.edge_src_array[cand]
            tgt_indices = self.edge_tgt_array[cand]
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
            # Handles the parallel-motion edge case (A_c ~ 0) where a pure
            # quadratic-root approach would otherwise miss interior contact.
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
                        int(cand[i]), float(tau_start[i]), float(tau_end[i]), tdur
                    )
                )

        if self.record_sweep:
            self.overlapping_interval_sweep[u, v, velocity, r] = (
                overlapping_vertices,
                overlapping_edges,
            )
        return overlapping_vertices, overlapping_edges