from CGAL.CGAL_Kernel import Point_3, Segment_3, squared_distance, Point_2, Segment_2
import numpy as np
from scipy.spatial import KDTree
from rtree import index

class CGAL_Sweep:
    def __init__(self,record_sweep: bool = True,use_exact_collision_check: bool = True):
        self.reset()
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check

    def reset(self):
        self.Point_type = None
        self.Segment_type = None
        self.vertices = []
        self.edges = []
        self.edge_indices = {}
        self.overlapping_sweep = {}
        self.vertex_kdtree = None
        self.vertex_positions = None
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
        elif dim == 3:
            self.Point_type = Point_3
            self.Segment_type = Segment_3

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
            v1 = np.array(vertices[src])
            v2 = np.array(vertices[tgt])
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
            # Use R-tree to find ALL edges within radius (not just connected ones)
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
                    overlapping_edges.add(self.edge_indices[edge_idx][::-1])
            
            if self.record_sweep:
                self.overlapping_sweep[u,v,velocity,r] = overlapping_edges
            return overlapping_edges

        # Regular segment query (moving agent)
        # --- Vertex overlap using KDTree ---
        seg_length = np.linalg.norm(v_arr - u_arr)
        num_samples = max(3, int(np.ceil(seg_length / (r * 0.5))) + 1)
        
        # Sample along segment and query KDTree
        for t in np.linspace(0, 1, num_samples):
            sample_pt = u_arr + t * (v_arr - u_arr)
            indices = self.vertex_kdtree.query_ball_point(sample_pt, r-1e-10)
            overlapping_vertices.update(indices)

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
                overlapping_edges.add(self.edge_indices[edge_idx][::-1])

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
