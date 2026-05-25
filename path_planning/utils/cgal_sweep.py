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
        self.overlapping_sweep = {}
        self.overlapping_interval_sweep = {}
        self.vertex_kdtree = None
        self.edge_aabbs = []
        self.edge_rtree = None

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

        # Precompute edge bounding boxes for filtering.
        for edge in edges:
            src, tgt = edge
            a_pt = self.Point_type(*vertices[src])
            b_pt = self.Point_type(*vertices[tgt])
            self.edges.append(self.Segment_type(a_pt, b_pt))
            edge_idx = len(self.edges) - 1
            self.edge_indices[edge_idx] = (src, tgt)

            # Store edge bounding box (will be expanded by query radius at query time)
            v1 = self.vertex_positions[src]
            v2 = self.vertex_positions[tgt]
            bbox_min = np.minimum(v1, v2)
            bbox_max = np.maximum(v1, v2)
            self.edge_aabbs.append((bbox_min, bbox_max, edge_idx))

        # Vectorized lookup arrays: avoid per-query list comprehensions over edge_indices.
        self.edge_src_array = np.array([e[0] for e in edges], dtype=np.int64)
        self.edge_tgt_array = np.array([e[1] for e in edges], dtype=np.int64)
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

    def overlapping_graph_elements_cgal(
        self, u: tuple[float, float], v: tuple[float, float], velocity: float = 0.0, r: float = 0.5
    ):
        if self.record_sweep and (u, v, velocity, r) in self.overlapping_sweep:
            return self.overlapping_sweep[u, v, velocity, r]

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
                    overlapping_edges.add(self.edge_indices[edge_idx])

            if self.record_sweep:
                self.overlapping_sweep[u, v, velocity, r] = overlapping_edges
            return overlapping_edges

        # Regular segment query (moving agent)
        overlapping_vertices = set(self._query_vertices_on_segment(u_arr, v_arr, r))

        # Edge overlap via R-tree spatial query + vectorized seg-seg distance.
        query_bbox = self._build_query_bbox(u_arr, v_arr, r)
        candidate_edge_indices = list(self.edge_rtree.intersection(query_bbox))
        hits = self._edges_within_r_of_segment(u_arr, v_arr, r, candidate_edge_indices)
        for edge_idx in hits:
            overlapping_edges.add(self.edge_indices[edge_idx])

        if self.use_exact_collision_check and overlapping_edges:
            # Drop edges where the two moving agents (with relative velocity)
            # never come within r of each other within the duration tdur.
            crossing_edges = [
                (src, tgt)
                for (src, tgt) in overlapping_edges
                if src not in overlapping_vertices or tgt not in overlapping_vertices
            ]
            if crossing_edges:
                crossing_src = np.array([e[0] for e in crossing_edges], dtype=np.int64)
                crossing_tgt = np.array([e[1] for e in crossing_edges], dtype=np.int64)
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
                for keep, e in zip(~miss, crossing_edges):
                    if not keep:
                        overlapping_edges.discard(e)

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
                cand = np.asarray(candidate_edge_indices, dtype=np.int64)
                src_indices = self.edge_src_array[cand]
                tgt_indices = self.edge_tgt_array[cand]
                pairs = list(zip(src_indices.tolist(), tgt_indices.tolist()))
                if get_time_interval:
                    overlapping_edges = {pair: (0.0, float("inf")) for pair in pairs}
                else:
                    overlapping_edges = set(pairs)
            else:
                overlapping_edges = {} if get_time_interval else set()

            if self.record_sweep:
                self.overlapping_interval_sweep[u, v, velocity, r] = (
                    overlapping_vertices,
                    overlapping_edges,
                )
            return overlapping_vertices, overlapping_edges

        # Regular segment query (moving agent)
        vertex_hits = self._query_vertices_on_segment(u_arr, v_arr, r)

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

        if vertex_hits:
            if get_time_interval:
                vert_idx = np.asarray(vertex_hits, dtype=np.int64)
                r0 = u_arr - self.vertex_positions[vert_idx]
                t1, t2 = self.get_interval_from_quadratic_equation(r0, vel_vec, r, tdur)
                overlapping_vertices = {
                    int(idx): (float(t1[i]), float(t2[i])) for i, idx in enumerate(vert_idx.tolist())
                }
            else:
                overlapping_vertices = set(vertex_hits)
        else:
            overlapping_vertices = {} if get_time_interval else set()

        # Edge overlap via R-tree + vectorized seg-seg distance.
        query_bbox = self._build_query_bbox(u_arr, v_arr, r)
        rtree_candidates = list(self.edge_rtree.intersection(query_bbox))
        edge_hits = self._edges_within_r_of_segment(u_arr, v_arr, r, rtree_candidates)

        if not edge_hits:
            overlapping_edges = {} if get_time_interval else set()
        elif not get_time_interval:
            cand = np.asarray(edge_hits, dtype=np.int64)
            src_indices = self.edge_src_array[cand]
            tgt_indices = self.edge_tgt_array[cand]
            overlapping_edges = set(zip(src_indices.tolist(), tgt_indices.tolist()))
        else:
            cand = np.asarray(edge_hits, dtype=np.int64)
            src_indices = self.edge_src_array[cand]
            tgt_indices = self.edge_tgt_array[cand]
            a_pos = self.vertex_positions[src_indices]
            b_pos = self.vertex_positions[tgt_indices]
            a_to_b = b_pos - a_pos
            K = a_pos.shape[0]

            all_starts = np.full(K, np.inf)
            all_ends = np.full(K, -np.inf)

            # Endpoint spheres (vertex a, vertex b).
            for endpoint in (a_pos, b_pos):
                rel_pos = u_arr - endpoint
                A = float(vel_vec @ vel_vec)
                B = 2.0 * (rel_pos @ vel_vec)
                C = np.sum(rel_pos * rel_pos, axis=1) - r * r
                disc = B * B - 4.0 * A * C
                mask = disc >= 0
                sqrt_disc = np.sqrt(np.maximum(0.0, disc))
                denom = 2.0 * A + 1e-12
                t1 = (-B - sqrt_disc) / denom
                t2 = (-B + sqrt_disc) / denom
                earlier = mask & (t1 < all_starts)
                all_starts[earlier] = t1[earlier]
                later = mask & (t2 > all_ends)
                all_ends[later] = t2[later]

            # Cylinder side: contact happens when perpendicular distance to
            # line(ab) is <= r AND the projection of the agent onto line(ab)
            # is within the segment [0, 1]. Compute each as a time interval
            # and take their intersection. This handles the parallel-motion
            # edge case (A_c ~ 0) where the quadratic-root approach would
            # otherwise miss the interior contact interval.
            seg_len_sq = np.sum(a_to_b * a_to_b, axis=1) + 1e-12
            vdot = (a_to_b @ vel_vec) / seg_len_sq                # (K,) rate of projection change
            v_perp = vel_vec[None, :] - vdot[:, None] * a_to_b
            rel_pos_u = u_arr - a_pos                              # (K, d)
            pdot = np.sum(rel_pos_u * a_to_b, axis=1) / seg_len_sq  # (K,) initial projection
            pos_perp = rel_pos_u - pdot[:, None] * a_to_b

            A_c = np.sum(v_perp * v_perp, axis=1)
            B_c = 2.0 * np.sum(v_perp * pos_perp, axis=1)
            C_c = np.sum(pos_perp * pos_perp, axis=1) - r * r

            LARGE = 1e18

            # (1) Interval where perpendicular distance to line(ab) is <= r.
            moving_perp = A_c > 1e-12
            disc_c = B_c * B_c - 4.0 * A_c * C_c
            has_perp_roots = moving_perp & (disc_c >= 0.0)
            sqrt_disc_c = np.sqrt(np.maximum(0.0, disc_c))
            safe_A = np.where(moving_perp, A_c, 1.0)
            t_perp_lo = np.where(has_perp_roots, (-B_c - sqrt_disc_c) / (2.0 * safe_A), LARGE)
            t_perp_hi = np.where(has_perp_roots, (-B_c + sqrt_disc_c) / (2.0 * safe_A), -LARGE)
            # Stationary perpendicular component (parallel motion) that's already inside r:
            stationary_inside = (~moving_perp) & (C_c <= 0.0)
            t_perp_lo = np.where(stationary_inside, -LARGE, t_perp_lo)
            t_perp_hi = np.where(stationary_inside, LARGE, t_perp_hi)

            # (2) Interval where the projection s(t) = pdot + t*vdot lies in [0, 1].
            moving_proj = np.abs(vdot) > 1e-12
            safe_vdot = np.where(moving_proj, vdot, 1.0)
            t_proj_a = -pdot / safe_vdot
            t_proj_b = (1.0 - pdot) / safe_vdot
            t_proj_lo_m = np.minimum(t_proj_a, t_proj_b)
            t_proj_hi_m = np.maximum(t_proj_a, t_proj_b)
            proj_const_in = (~moving_proj) & (pdot >= 0.0) & (pdot <= 1.0)
            t_proj_lo = np.where(moving_proj, t_proj_lo_m, np.where(proj_const_in, -LARGE, LARGE))
            t_proj_hi = np.where(moving_proj, t_proj_hi_m, np.where(proj_const_in, LARGE, -LARGE))

            # Intersection of the two intervals = cylinder contact interval.
            t_cyl_lo = np.maximum(t_perp_lo, t_proj_lo)
            t_cyl_hi = np.minimum(t_perp_hi, t_proj_hi)
            has_cyl = t_cyl_lo < t_cyl_hi

            earlier_cyl = has_cyl & (t_cyl_lo < all_starts)
            all_starts[earlier_cyl] = t_cyl_lo[earlier_cyl]
            later_cyl = has_cyl & (t_cyl_hi > all_ends)
            all_ends[later_cyl] = t_cyl_hi[later_cyl]

            tau_start = np.clip(all_starts, 0.0, tdur)
            tau_end = np.clip(all_ends, 0.0, tdur)
            no_collision = (tau_start >= tau_end) | np.isinf(all_starts)
            overlapping_edges = {
                (int(src_indices[i]), int(tgt_indices[i])): (float(tau_start[i]), float(tau_end[i]))
                for i in range(K)
                if not no_collision[i]
            }

        if self.record_sweep:
            self.overlapping_interval_sweep[u, v, velocity, r] = (
                overlapping_vertices,
                overlapping_edges,
            )
        return overlapping_vertices, overlapping_edges