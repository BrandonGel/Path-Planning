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
        # --- Vertex overlap using KDTree ---
        seg_length = np.linalg.norm(v_arr - u_arr)
        num_samples = max(3, int(np.ceil(seg_length / (r * 0.5))) + 1)
        
        # Sample along segment and query KDTree in a single batched call
        ts = np.linspace(0.0, 1.0, num_samples)[:, None]  # (num_samples, 1)
        samples = u_arr + ts * (v_arr - u_arr)            # (num_samples, dim)
        indices_per_sample = self.vertex_kdtree.query_ball_point(samples, r - 1e-10)
        for idx_list in indices_per_sample:
            overlapping_vertices.update(idx_list)

        # --- Edge overlap using R-tree spatial query (OPTIMIZED) ---
        # Compute AABB for query segment (expanded by radius)
        query_min = np.minimum(u_arr, v_arr) - r
        query_max = np.maximum(u_arr, v_arr) + r
        
        # Query R-tree: only returns edges whose bounding boxes overlap with query box
        # This is O(log E + k) instead of O(E) where k is number of overlapping edges
        dim = len(u)
        if dim == 2:
            query_bbox = (query_min[0], query_min[1], query_max[0], query_max[1])
        else:  # 3D
            query_bbox = (query_min[0], query_min[1], query_min[2],
                         query_max[0], query_max[1], query_max[2])
        
        # R-tree intersection query - only returns candidate edges!
        candidate_edge_indices = list(self.edge_rtree.intersection(query_bbox))
        
        # Only run expensive CGAL check on candidates
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
        if vel.ndim == 1:
            v = vel.reshape(-1)                  # (d,)
            a = np.dot(v, v)                    # scalar

            b = 2.0 * (r0 @ v)                  # (N,)
            c = np.einsum('ij,ij->i', r0, r0) - r**2 + 1e-10  # (N,)

            disc = b**2 - 4.0 * a * c           # (N,)

            t1 = np.zeros_like(b, dtype=float)
            t2 = np.zeros_like(b, dtype=float)
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

    def overlapping_interval_cgal(self, u: tuple[float,float], v: tuple[float,float],velocity: float = 0.0, r: float = 0.5):
        if self.record_sweep and (u,v,velocity,r) in self.overlapping_interval_sweep:
            overlapping_vertices,overlapping_edges = self.overlapping_interval_sweep[u,v,velocity,r]
            return overlapping_vertices,overlapping_edges
            
        u_pt = self.Point_type(*u)
        v_pt = self.Point_type(*v)
        traversal_seg = self.Segment_type(u_pt, v_pt)
        
        u_arr = np.array(u)
        v_arr = np.array(v)

        # Special case: point query (stationary agent)
        if u == v:
            indices = self.vertex_kdtree.query_ball_point(u_arr, r-1e-10)
            overlapping_vertices = {index: (0, float('inf')) for index in indices}
            
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
            overlapping_edges = {}
            for edge_idx in candidate_edge_indices:
                if squared_distance(u_pt, self.edges[edge_idx])**0.5 < r:
                    overlapping_edges[self.edge_indices[edge_idx]] = (0, float('inf'))
            
            if self.record_sweep:
                self.overlapping_interval_sweep[u,v,velocity,r] = overlapping_vertices,overlapping_edges
            return overlapping_vertices,overlapping_edges

        # Regular segment query (moving agent)
        # --- Vertex overlap using KDTree ---
        seg_length = np.linalg.norm(v_arr - u_arr)
        num_samples = max(3, int(np.ceil(seg_length / (r * 0.5))) + 1)
        
        # Sample along segment and query KDTree in a single batched call
        ts = np.linspace(0.0, 1.0, num_samples)[:, None]  # (num_samples, 1)
        samples = u_arr + ts * (v_arr - u_arr)            # (num_samples, dim)
        indices_per_sample = self.vertex_kdtree.query_ball_point(samples, r - 1e-10)

        overlapping_vertices_set = set()
        for idx_list in indices_per_sample:
            overlapping_vertices_set.update(idx_list)

        if overlapping_vertices_set:
            all_indices = list(overlapping_vertices_set)
            overlapping_vertices_position = self.vertex_positions[all_indices]
            r0 = u_arr - overlapping_vertices_position
            vel_vec = v_arr - u_arr
            if velocity > 0.0:
                vel_norm = np.linalg.norm(vel_vec)
                vel_vec = velocity * vel_vec / vel_norm if vel_norm > 0.0 else np.zeros_like(vel_vec)
                tdur = vel_norm / velocity 
            else:
                tdur = 1.0
            t1, t2 = self.get_interval_from_quadratic_equation(r0, vel_vec, r, tdur)
            overlapping_vertices = {idx: (float(t1[i]), float(t2[i])) for i, idx in enumerate(all_indices)}
        else:
            overlapping_vertices = {}

        # --- Edge overlap using R-tree spatial query (OPTIMIZED) ---
        # Compute AABB for query segment (expanded by radius)
        query_min = np.minimum(u_arr, v_arr) - r
        query_max = np.maximum(u_arr, v_arr) + r
        
        # Query R-tree: only returns edges whose bounding boxes overlap with query box
        # This is O(log E + k) instead of O(E) where k is number of overlapping edges
        dim = len(u)
        if dim == 2:
            query_bbox = (query_min[0], query_min[1], query_max[0], query_max[1])
        else:  # 3D
            query_bbox = (query_min[0], query_min[1], query_min[2],
                         query_max[0], query_max[1], query_max[2])
        
        # R-tree intersection query - only returns candidate edges!
        candidate_edge_indices = list(self.edge_rtree.intersection(query_bbox))
        
        # Only run expensive CGAL check on candidates
        candidate_edge_indices = [
            edge_idx for edge_idx in candidate_edge_indices
            if squared_distance(traversal_seg, self.edges[edge_idx])**0.5 < r
        ]

        if not candidate_edge_indices:
            overlapping_edges = {}
        else:
            candidate_edge_indices = np.array(candidate_edge_indices, dtype=int)
            # Map to vertex indices
            src_indices = np.array([self.edge_indices[i][0] for i in candidate_edge_indices], dtype=int)
            tgt_indices = np.array([self.edge_indices[i][1] for i in candidate_edge_indices], dtype=int)

            # Positions and relative motion
            a_pos = self.vertex_positions[src_indices]  # (K, dim)
            b_pos = self.vertex_positions[tgt_indices]  # (K, dim)
            u_to_v = v_arr - u_arr                      # (dim,)
            a_to_b = b_pos - a_pos                      # (K, dim)
            ro1 = a_pos - u_arr                         # (K, dim)

            # Per-edge relative velocity and duration
            if velocity == 0.0:
                vel_edges = a_to_b - u_to_v             # (K, dim)
                tdur_edges = np.ones(len(candidate_edge_indices), dtype=float)
            else:
                diff = a_to_b - u_to_v                  # (K, dim)
                dist = np.linalg.norm(diff, axis=1)     # (K,)
                vel_edges = np.zeros_like(diff)
                nonzero = dist > 0.0
                vel_edges[nonzero] = (velocity * diff[nonzero] /
                                       dist[nonzero, None])
                tdur_edges = np.zeros_like(dist)
                tdur_edges[nonzero] = dist[nonzero] / velocity

            # Determine crossing edges: any endpoint not in overlapping_vertices
            overlapping_vertex_indices = set(overlapping_vertices.keys())
            is_crossing = np.array([
                (src not in overlapping_vertex_indices) or
                (tgt not in overlapping_vertex_indices)
                for src, tgt in zip(src_indices, tgt_indices)
            ], dtype=bool)

            vel_norm_sq = np.sum(vel_edges * vel_edges, axis=1)  # (K,)

            # Crossing edges: remove if min distance along motion exceeds r
            keep_mask = np.ones(len(candidate_edge_indices), dtype=bool)
            crossing_idx = np.where(is_crossing & (vel_norm_sq > 0.0))[0]
            if crossing_idx.size > 0:
                ro1_cross = ro1[crossing_idx]
                vel_cross = vel_edges[crossing_idx]
                tdur_cross = tdur_edges[crossing_idx]
                vel_norm_sq_cross = vel_norm_sq[crossing_idx]

                tmin = np.clip(
                    -np.sum(ro1_cross * vel_cross, axis=1) /
                    (vel_norm_sq_cross + 1e-10),
                    0.0,
                    tdur_cross
                )
                vec = ro1_cross + vel_cross * tmin[:, None]
                vec_norm = np.linalg.norm(vec, axis=1)
                # Remove edges where closest approach is outside radius
                keep_mask[crossing_idx[vec_norm > r]] = False

            # Quadratic intervals for kept edges, using shared helper
            kept_idx = np.where(keep_mask)[0]
            overlapping_edges = {}
            if kept_idx.size > 0:
                ro1_kept = ro1[kept_idx]
                vel_kept = vel_edges[kept_idx]
                tdur_kept = tdur_edges[kept_idx]

                t1, t2 = self.get_interval_from_quadratic_equation(ro1_kept, vel_kept, r, tdur_kept)

                # Build final mapping using original (src, tgt) pairs
                for local_i, global_i in enumerate(kept_idx):
                    edge_idx = candidate_edge_indices[global_i]
                    edge_key = self.edge_indices[edge_idx]
                    lower_t1 = t1[local_i]
                    upper_t2 = t2[local_i]  
                    if lower_t1 == upper_t2:
                        if  np.linalg.norm(ro1_kept[local_i]) >= r:
                            continue
                        lower_t1 = 0.0
                        upper_t2 = tdur_kept[local_i]
                    overlapping_edges[edge_key] = (float(lower_t1), float(upper_t2))


        # Caller expects overlapping_vertices as dict (vertex_index -> interval)
        if self.record_sweep:
            self.overlapping_interval_sweep[u,v,velocity,r] = overlapping_vertices,overlapping_edges
        return overlapping_vertices,overlapping_edges