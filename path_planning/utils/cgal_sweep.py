from CGAL.CGAL_Kernel import Point_3, Segment_3, Vector_3, squared_distance, Point_2, Segment_2, Vector_2
import numpy as np
from scipy.spatial import KDTree
from rtree import index

class CGAL_Sweep:
    def __init__(self,record_sweep: bool = True,use_exact_collision_check: bool = True):
        self.reset()
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check

    def reset(self):
        self.Zero_Point = None
        self.Point_type = None
        self.Segment_type = None
        self.vertices = []
        # Precomputed numpy representations
        self.vertex_positions = None
        self.edges = []
        self.edge_indices = {}
        self.overlapping_sweep = {}
        self.overlapping_interval_sweep = {}
        self.vertex_kdtree = None
        self.edge_aabbs = []
        self.vertex_adjacency = {}  # vertex_idx -> list of edge_indices connected to it
        self.edge_rtree = None  # R-tree for spatial edge queries

    def set_graph(self,vertices: list[tuple[float,float]], edges: list[tuple[int,int]], default_radius: float = 1.0):
        self.reset()
        assert len(vertices) > 0, "Vertices cannot be empty"
        assert len(edges) > 0, "Edges cannot be empty"
        dim = vertices[0].__len__()
        if dim == 2:
            self.Point_type = Point_2
            self.Segment_type = Segment_2
            self.Zero_Vector = Vector_2(0.0,0.0)
        elif dim == 3:
            self.Point_type = Point_3
            self.Segment_type = Segment_3
            self.Zero_Vector = Vector_3(0.0,0.0,0.0)
        for vertex in vertices:
            p_pt = self.Point_type(*vertex)
            self.vertices.append(p_pt)

        # Build KDTree for vertex queries
        self.vertex_positions = np.array([list(v) for v in vertices])
        self.vertex_kdtree = KDTree(self.vertex_positions)

        # Precompute edge bounding boxes for filtering
        # Use default_radius to expand AABBs (will be refined per query)
        for edge in edges:
            src,tgt = edge
            a_pt = self.Point_type(*vertices[src])
            b_pt = self.Point_type(*vertices[tgt])
            self.edges.append(self.Segment_type(a_pt, b_pt))
            edge_idx = len(self.edges) - 1
            self.edge_indices[edge_idx] = (src,tgt)

            # Store edge bounding box (will be expanded by query radius at query time)
            v1 = self.vertex_positions[src]
            v2 = self.vertex_positions[tgt]
            bbox_min = np.minimum(v1, v2)
            bbox_max = np.maximum(v1, v2)
            self.edge_aabbs.append((bbox_min, bbox_max, edge_idx))


        # Build R-tree for efficient spatial edge queries
        # R-tree stores bounding boxes and allows O(log E + k) queries instead of O(E)
        p = index.Property()
        p.dimension = dim
        self.edge_rtree = index.Index(properties=p)
        
        for edge_idx, (bbox_min, bbox_max, _) in enumerate(self.edge_aabbs):
            # R-tree expects (min_x, min_y, max_x, max_y) for 2D
            # or (min_x, min_y, min_z, max_x, max_y, max_z) for 3D
            if dim == 2:
                bbox = (bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1])
            else:  # 3D
                bbox = (bbox_min[0], bbox_min[1], bbox_min[2], 
                       bbox_max[0], bbox_max[1], bbox_max[2])
            
            # Insert edge with its bounding box into R-tree
            self.edge_rtree.insert(edge_idx, bbox)

    def _query_vertices_on_segment(self, u_arr, v_arr, r):
        """Sample along segment u->v and return list of vertex index lists per sample."""
        seg_length = np.linalg.norm(np.asarray(v_arr) - np.asarray(u_arr))
        num_samples = max(3, int(np.ceil(seg_length / (r * 0.5))) + 1)
        ts = np.linspace(0.0, 1.0, num_samples)[:, None]
        samples = u_arr + ts * (v_arr - u_arr)
        return self.vertex_kdtree.query_ball_point(samples, r - 1e-10)

    def _build_query_bbox(self, u_arr, v_arr, r):
        """Build R-tree bbox tuple (min_x, min_y[, min_z], max_x, max_y[, max_z]) for segment expanded by r."""
        query_min = np.minimum(u_arr, v_arr).ravel() - r
        query_max = np.maximum(u_arr, v_arr).ravel() + r
        return (*query_min, *query_max)

    def overlapping_graph_elements_cgal(self, u: tuple[float,float], v: tuple[float,float],velocity: float = 0.0, r: float = 0.5):
        if self.record_sweep and (u,v,velocity,r) in self.overlapping_sweep:
            overlapping_edges = self.overlapping_sweep[u,v,velocity,r]
            return overlapping_edges
            
        u_pt = self.Point_type(*u)
        v_pt = self.Point_type(*v)
        traversal_seg = self.Segment_type(u_pt, v_pt)
        
        u_arr = np.array(u)
        v_arr = np.array(v)

        overlapping_vertices = set()
        overlapping_edges = set()

        # Special case: point query (stationary agent)
        if u == v:    
            # Create query box: point expanded by radius
            query_min = u_arr - r
            query_max = u_arr + r
            
            dim = len(u)
            if dim == 2:
                query_bbox = (query_min[0], query_min[1], query_max[0], query_max[1])
            else:  # 3D
                query_bbox = (query_min[0], query_min[1], query_min[2],
                            query_max[0], query_max[1], query_max[2])
            
            # Query R-tree for edges whose bounding boxes overlap with query point
            candidate_edge_indices = list(self.edge_rtree.intersection(query_bbox))
            
            # Check exact distance for candidate edges
            overlapping_edges = set()
            for edge_idx in candidate_edge_indices:
                if squared_distance(u_pt, self.edges[edge_idx])**0.5 < r:
                    overlapping_edges.add(self.edge_indices[edge_idx])
            
            if self.record_sweep:
                self.overlapping_sweep[u,v,velocity,r] = overlapping_edges
            return overlapping_edges

        # Regular segment query (moving agent)
        indices_per_sample = self._query_vertices_on_segment(u_arr, v_arr, r)
        for idx_list in indices_per_sample:
            overlapping_vertices.update(idx_list)

        # --- Edge overlap using R-tree spatial query (OPTIMIZED) ---
        query_bbox = self._build_query_bbox(u_arr, v_arr, r)
        candidate_edge_indices = list(self.edge_rtree.intersection(query_bbox))
        for edge_idx in candidate_edge_indices:
            if squared_distance(traversal_seg, self.edges[edge_idx])**0.5 < r:
                overlapping_edges.add(self.edge_indices[edge_idx])

        if self.use_exact_collision_check:
            crossing_edges = set()
            for edge in overlapping_edges:
                src,tgt = edge
                if src not in overlapping_vertices:
                    crossing_edges.add(edge)
                if tgt not in overlapping_vertices:
                    crossing_edges.add(edge)

            remove_edges = set()
            
            for edge in crossing_edges:
                src,tgt = edge
                a_pt = self.vertices[src]
                b_pt = self.vertices[tgt]

                u_to_v = v_pt-u_pt
                a_to_b = b_pt-a_pt

                 # --- Edge 1 ---
                # a -> b and u -> v Z
                ro1 = a_pt-u_pt
                if velocity == 0.0:
                    vel = a_to_b-u_to_v
                    tdur = 1.0
                else:
                    dist = np.linalg.norm(a_to_b-u_to_v)
                    vel = velocity*(a_to_b-u_to_v)/dist if dist > 0.0 else np.zeros(dim)
                    tdur = np.linalg.norm(a_to_b-u_to_v)/velocity
                tmin = np.clip(-np.dot(ro1,vel)/(np.dot(vel,vel)+1e-10),0.0,tdur)
                vec = ro1 + vel*tmin
                if np.linalg.norm(vec) > r:
                    remove_edges.add(edge)

            overlapping_edges = overlapping_edges - remove_edges
        if self.record_sweep:
            self.overlapping_sweep[u,v,velocity,r] = overlapping_edges
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

        # Case 1: shared velocity vector for all rows
        if len(vel) == 1:
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

    def overlapping_interval_cgal(self, u: tuple[float,float], v: tuple[float,float],velocity: float = 0.0, r: float = 0.5,get_time_interval: bool = False):
        if self.record_sweep and (u,v,velocity,r) in self.overlapping_interval_sweep:
            overlapping_vertices,overlapping_edges = self.overlapping_interval_sweep[u,v,velocity,r]
            return overlapping_vertices,overlapping_edges
            
        u_pt = self.Point_type(*u)
        v_pt = self.Point_type(*v)
        traversal_seg = self.Segment_type(u_pt, v_pt)
        
        u_arr = np.array(u).reshape(1,-1)
        v_arr = np.array(v).reshape(1,-1)

        # Special case: point query (stationary agent)
        if u == v:
            indices = self.vertex_kdtree.query_ball_point(u, r-1e-10)
            overlapping_vertices = {index: (0, float('inf')) for index in indices}
            
            # Create query box: point expanded by radius
            query_min = u_arr - r
            query_max = u_arr + r
            
            dim = len(u)
            if dim == 2:
                query_bbox = (query_min[0,0], query_min[0,1], query_max[0,0], query_max[0,1])
            else:  # 3D
                query_bbox = (query_min[0,0], query_min[0,1], query_min[0,2],
                            query_max[0,0], query_max[0,1], query_max[0,2])
            
            # Query R-tree for edges whose bounding boxes overlap with query point
            candidate_edge_indices = list(self.edge_rtree.intersection(query_bbox))
            
            # Check exact distance for candidate edges
            overlapping_edges = {}
            for edge_idx in candidate_edge_indices:
                if squared_distance(u_pt, self.edges[edge_idx])**0.5 < r:
                    # overlapping_edges[self.edge_indices[edge_idx]] = (0, float('inf'),u,u)
                    overlapping_edges[self.edge_indices[edge_idx]] = (0, float('inf'))
            
            if self.record_sweep:
                self.overlapping_interval_sweep[u,v,velocity,r] = overlapping_vertices,overlapping_edges
            return overlapping_vertices,overlapping_edges

        # Regular segment query (moving agent)
        indices_per_sample = self._query_vertices_on_segment(u_arr, v_arr, r)
        overlapping_vertices_set = set()
        for idx_list in indices_per_sample:
            overlapping_vertices_set.update(idx_list)

        if overlapping_vertices_set:
            all_indices = list(overlapping_vertices_set)
            overlapping_vertices_position = self.vertex_positions[all_indices]
            r0 = u_arr - overlapping_vertices_position

            # Per-edge relative velocity and duration
            u_to_v = v_arr - u_arr 
            if velocity == 0.0:
                vel_vec = u_to_v            # (K, dim)
                tdur = 1
            else:
                dist = np.linalg.norm(u_to_v)     # (K,)
                vel_vec = velocity*u_to_v/dist
                tdur = dist / velocity

            t1, t2 = self.get_interval_from_quadratic_equation(r0, vel_vec, r, tdur)
            overlapping_vertices = {idx: (float(t1[i]), float(t2[i])) for i, idx in enumerate(all_indices)}
        else:
            overlapping_vertices = {}

        # --- Edge overlap using R-tree spatial query (OPTIMIZED) ---
        query_bbox = self._build_query_bbox(u_arr, v_arr, r)
        candidate_edge_indices = list(self.edge_rtree.intersection(query_bbox))
        candidate_edge_indices = [
            edge_idx for edge_idx in candidate_edge_indices
            if squared_distance(traversal_seg, self.edges[edge_idx])**0.5 < r
        ]

        if not candidate_edge_indices:
            overlapping_edges = {}
        elif not get_time_interval:
            # Only run expensive CGAL check on candidates
            overlapping_edges = {}
            for edge_idx in candidate_edge_indices:
                # overlapping_edges[self.edge_indices[edge_idx]] = (0, float('inf'),u,u)
                overlapping_edges[self.edge_indices[edge_idx]] = (0, float('inf'))
            crossing_edges = set()
            for edge in overlapping_edges:
                src,tgt = edge
                if src not in overlapping_vertices:
                    crossing_edges.add(edge)
                if tgt not in overlapping_vertices:
                    crossing_edges.add(edge)
            
            for edge in crossing_edges:
                src,tgt = edge
                a_pt = self.vertices[src]
                b_pt = self.vertices[tgt]

                u_to_v = v_pt-u_pt
                a_to_b = b_pt-a_pt

                 # --- Edge 1 ---
                # a -> b and u -> v Z
                ro1 = a_pt-u_pt
                if velocity == 0.0:
                    vel = a_to_b-u_to_v
                    tdur = 1.0
                else:
                    dist = np.linalg.norm(a_to_b-u_to_v)
                    vel = velocity*(a_to_b-u_to_v)/dist if dist > 0.0 else np.zeros(dim)
                    tdur = np.linalg.norm(a_to_b-u_to_v)/velocity
                tmin = np.clip(-np.dot(ro1,vel)/(np.dot(vel,vel)+1e-10),0.0,tdur)
                vec = ro1 + vel*tmin
                if np.linalg.norm(vec) > r:
                    overlapping_edges.pop(edge,None)
        else:
            overlapping_edges = {}
            candidate_edge_indices = np.array(candidate_edge_indices, dtype=int)
            src_indices = np.array([self.edge_indices[i][0] for i in candidate_edge_indices], dtype=int)
            tgt_indices = np.array([self.edge_indices[i][1] for i in candidate_edge_indices], dtype=int)

            # Positions and relative motion (float32 to reduce peak memory)
            K = len(candidate_edge_indices)
            a_pos = self.vertex_positions[src_indices].astype(np.float32, copy=False).reshape(K, -1)
            b_pos = self.vertex_positions[tgt_indices].astype(np.float32, copy=False).reshape(K, -1)
            u_arr_f = np.asarray(u_arr, dtype=np.float32)
            v_arr_f = np.asarray(v_arr, dtype=np.float32)
            u_to_v = v_arr_f - u_arr_f
            a_to_b = b_pos - a_pos

            if velocity == 0.0:
                vel_vec = u_to_v
                tdur = 1
            else:
                dist = np.linalg.norm(u_to_v)
                vel_vec = velocity * u_to_v / dist
                tdur = float(dist / velocity)

            K = a_pos.shape[0]
            # Initialize intervals with null values
            all_starts = np.full(K, np.inf, dtype=np.float32)
            all_ends = np.full(K, -np.inf, dtype=np.float32)

            # Track segment positions at start and end of collision
            start_seg_pos = np.full(K, np.nan, dtype=np.float32)
            end_seg_pos = np.full(K, np.nan, dtype=np.float32)

            def get_seg_proj(tau, a_to_b, seg_len_sq):
                # Position of moving point at time tau
                p_tau = u_arr_f + tau[:, np.newaxis] * vel_vec
                # Projection onto segment
                proj = np.sum((p_tau - a_pos) * a_to_b, axis=1) / seg_len_sq
                return np.clip(proj, 0, 1)
                
            # 1. Endpoints a and b (Spheres)
            for i, endpoint in enumerate([a_pos, b_pos]):
                rel_pos = u_arr_f - endpoint
                A = np.sum(vel_vec**2, axis=1)
                B = 2 * np.sum(vel_vec * rel_pos, axis=1)
                C = np.sum(rel_pos**2, axis=1) - r**2
                
                disc = B**2 - 4*A*C
                mask = disc >= 0
                sqrt_disc = np.sqrt(np.maximum(0, disc))
                t1 = (-B - sqrt_disc) / (2 * A + 1e-9)
                t2 = (-B + sqrt_disc) / (2 * A + 1e-9)
                
                # Update starts
                is_earlier = mask & (t1 < all_starts)
                all_starts[is_earlier] = t1[is_earlier]
                start_seg_pos[is_earlier] = float(i) # 0 for a, 1 for b
                
                # Update ends
                is_later = mask & (t2 > all_ends)
                all_ends[is_later] = t2[is_later]
                end_seg_pos[is_later] = float(i)

            # 2. Cylinder Body
            a_to_b = b_pos - a_pos
            seg_len_sq = np.sum(a_to_b**2, axis=1) + 1e-9
            
            def get_perp(vec, axis_vec, axis_len_sq):
                dot = np.sum(vec * axis_vec, axis=1) / axis_len_sq
                return vec - dot[:, np.newaxis] * axis_vec

            v_perp = get_perp(vel_vec, a_to_b, seg_len_sq)
            pos_perp = get_perp(u_arr_f - a_pos, a_to_b, seg_len_sq)

            A_c = np.sum(v_perp**2, axis=1)
            B_c = 2 * np.sum(v_perp * pos_perp, axis=1)
            C_c = np.sum(pos_perp**2, axis=1) - r**2

            disc_c = B_c**2 - 4*A_c*C_c
            mask_c = disc_c >= 0
            sqrt_disc_c = np.sqrt(np.maximum(0, disc_c))
            t1_c = (-B_c - sqrt_disc_c) / (2 * A_c + 1e-9)
            t2_c = (-B_c + sqrt_disc_c) / (2 * A_c + 1e-9)

            # Check projection for cylinder entries/exits
            for t_val, is_start in [(t1_c, True), (t2_c, False)]:
                proj = get_seg_proj(t_val, a_to_b, seg_len_sq)
                valid_cyl = mask_c & (proj >= 0) & (proj <= 1)
                
                if is_start:
                    is_earlier = valid_cyl & (t_val < all_starts)
                    all_starts[is_earlier] = t_val[is_earlier]
                    start_seg_pos[is_earlier] = proj[is_earlier]
                else:
                    is_later = valid_cyl & (t_val > all_ends)
                    all_ends[is_later] = t_val[is_later]
                    end_seg_pos[is_later] = proj[is_later]

            # 3. Final Constraints
            # Clip to the travel period [0, tdur]
            tau_start = np.clip(all_starts, 0, tdur)
            tau_end = np.clip(all_ends, 0, tdur)

            # Re-calculate segment positions for clipped bounds
            start_seg_pos = np.where(all_starts < 0, get_seg_proj(tau_start, a_to_b, seg_len_sq), start_seg_pos)
            end_seg_pos = np.where(all_ends > tdur, get_seg_proj(tau_end, a_to_b, seg_len_sq), end_seg_pos)

            no_collision = (tau_start >= tau_end) | (all_starts == np.inf)
            tau_start[no_collision], tau_end[no_collision] = np.nan, np.nan
            start_seg_pos[no_collision], end_seg_pos[no_collision] = np.nan, np.nan

            L = np.linalg.norm(a_to_b, axis=1)
            lower_s = start_seg_pos*L
            upper_s = end_seg_pos*L
            lower_ta = tau_start - lower_s/velocity if velocity != 0.0 else lower_s/L
            lower_tb = tau_end - upper_s/velocity if velocity != 0.0 else upper_s/L
            upper_ta = tdur - lower_s/velocity if velocity != 0.0 else lower_s/L
            upper_tb = tdur - upper_s/velocity if velocity != 0.0 else upper_s/L
            lower_t = np.minimum(lower_ta, lower_tb)
            upper_t = np.maximum(upper_ta, upper_tb)
            overlapping_edges = {(src_indices[ii], tgt_indices[ii]): (float(lower_t[ii]), float(upper_t[ii])) for ii in range(len(candidate_edge_indices)) if not no_collision[ii]}

        # Caller expects overlapping_vertices as dict (vertex_index -> interval)
        if self.record_sweep:
            self.overlapping_interval_sweep[u,v,velocity,r] = overlapping_vertices,overlapping_edges
        return overlapping_vertices,overlapping_edges