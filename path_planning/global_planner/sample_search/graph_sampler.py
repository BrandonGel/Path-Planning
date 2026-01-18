
from typing import Union, List, Tuple, Dict, Any, Iterable
from python_motion_planning.common import  Node
from typing import List
import numpy as np
from scipy.spatial import KDTree, Delaunay
from python_motion_planning.path_planner import BasePathPlanner
from scipy.spatial.distance import cdist

class GraphSampler(BasePathPlanner):
    def __init__(self,*args,num_sample,num_neighbors = 13.0, min_edge_len = 0.0, max_edge_len = 30.0,**kwargs):
        super().__init__(*args, **kwargs)

        # Check if start and goal are lists, non-empty, and not None
        assert self.start is not None, "Start must not be None"
        assert self.goal is not None, "Goal must not be None"
        assert isinstance(self.start, list), "Start must be a list"
        assert isinstance(self.goal, list), "Goal must be a list"
        assert len(self.start) > 0, "Start list must not be empty"
        assert len(self.goal) > 0, "Goal list must not be empty"
        assert self.start[0] is not None, "First element of start must not be None"
        assert self.goal[0] is not None, "First element of goal must not be None"
        assert isinstance(self.start[0], tuple), "First element of start must be a tuple"
        assert isinstance(self.goal[0], tuple), "First element of goal must be a tuple"

        self.num_samples = num_sample
        self.num_total_nodes = num_sample + len(self.start) + len(self.goal)
        self.num_neighbors = num_neighbors
        self.min_edge_length = min_edge_len
        self.max_edge_length = max_edge_len
        self.node_index_list = {}
        self.road_map = []

    def __str__(self) -> str:
        return "Graph Sampler"
    
    def in_collision_dda(self, p1: Tuple[float, ...], p2: Tuple[float, ...]) -> bool:
        """
        Check if the line of sight between two continuous (world) points is in collision
        using DDA (Digital Differential Analyzer) grid traversal to check all tiles the line crosses.
        Optimized version with early termination on first collision.
        
        Args:
            p1: Start point in continuous (world) coordinates
            p2: End point in continuous (world) coordinates
        
        Returns:
            in_collision: True if any tile along the line is in collision, False otherwise
        """
        dim = self.map_.dim
        if len(p1) != dim or len(p2) != dim:
            raise ValueError(f"Points must have dimension {dim}")
        
        # Convert to grid coordinates using map's point_float_to_int
        p1_grid = np.array(self.map_.point_float_to_int(p1), dtype=int)
        p2_grid = np.array(self.map_.point_float_to_int(p2), dtype=int)
        current_tile = p1_grid.copy()
        
        # Early exit: check start and end tiles first
        if not self.map_.is_expandable(tuple(p1_grid)) or not self.map_.is_expandable(tuple(p2_grid)):
            return True
        
        # Early exit if start and end are in the same tile
        if np.array_equal(p1_grid, p2_grid):
            return False
        
        # Convert to numpy arrays for DDA calculations (world coordinates)
        p1_arr = np.array(p1, dtype=float)
        p2_arr = np.array(p2, dtype=float)
        
        # Calculate deltas
        delta = p2_arr - p1_arr
        
        # Calculate step direction
        step = np.sign(delta).astype(int)
        step[delta == 0] = 0
        
        # Calculate offsets to next grid boundary (vectorized)
        # Need to account for how the map converts coordinates
        # For DDA, we need the offset in world space to the next grid cell
        offset = np.where(delta > 0, 
                         np.ceil(p1_arr) - p1_arr,
                         p1_arr - np.floor(p1_arr))
        
        # Calculate tMax and tDelta (vectorized)
        abs_delta = np.abs(delta)
        non_zero_mask = abs_delta > 1e-10
        
        tMax = np.full(dim, float('inf'), dtype=float)
        tDelta = np.full(dim, float('inf'), dtype=float)
        
        tMax[non_zero_mask] = offset[non_zero_mask] / abs_delta[non_zero_mask]
        tDelta[non_zero_mask] = 1.0 / abs_delta[non_zero_mask]
        
        # Calculate distance (Manhattan distance in grid space)
        dist = int(np.sum(np.abs(p2_grid - p1_grid)))
        
        # Track seen tiles to avoid checking duplicates
        seen = {tuple(current_tile)}
        
        # Traverse with early termination
        for _ in range(dist):
            # Find dimension with minimum tMax
            valid_dims = np.isfinite(tMax)
            if not np.any(valid_dims):
                break
            
            valid_tMax = np.where(valid_dims, tMax, np.inf)
            min_dim = np.argmin(valid_tMax)
            
            # Step in that dimension
            current_tile[min_dim] += step[min_dim]
            
            # Update tMax
            if tDelta[min_dim] != float('inf'):
                tMax[min_dim] += tDelta[min_dim]
            
            # Check collision immediately (early termination)
            tile_tuple = tuple(current_tile)
            if tile_tuple not in seen:
                seen.add(tile_tuple)
                if not self.map_.is_expandable(tile_tuple):
                    return True  # Found collision, terminate early
        
        return False  # No collision found

    def generateRandomNodes(self):
        num_nodes = 0
        bounds = np.array(self.map_.bounds)
        nodes = []

        while num_nodes < self.num_samples+1:
            normalized_points = np.random.random((self.num_samples,self.map_.dim))
            points = normalized_points * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
            for point in points:
                node = Node(tuple(point),None,0,0)
                if self.map_.is_expandable(self.map_.point_float_to_int(node.current)):
                    nodes.append(node)
                    self.node_index_list[node] = len(nodes) - 1
                    num_nodes += 1
                if num_nodes == self.num_samples:
                    break
        for start in self.start:
            node = Node(tuple(start),None,0,0)
            nodes.append(node)
            self.node_index_list[node] = len(nodes) - 1
        for goal in self.goal:
            node = Node(tuple(goal),None,0,0)
            nodes.append(node)
            self.node_index_list[node] = len(nodes) - 1
        return nodes

    def generate_roadmap(self, samples: List[Node]):
        road_map = []
        points = np.array([samp.current for samp in samples])
        sample_kd_tree = KDTree(points)
        for i, node_s in zip(range(len(samples)), samples):
            s_pos = node_s.current
            dists, indexes = sample_kd_tree.query(s_pos, k=self.num_total_nodes)
            edge_id = []

            for ii in range(1, len(indexes)):
                n_pos = samples[indexes[ii]].current

                if not self.in_collision_dda(s_pos, n_pos)  \
                    and self.get_cost(s_pos,n_pos) >= self.min_edge_length and self.get_cost(s_pos,n_pos) <= self.max_edge_length:
                    edge_id.append(indexes[ii])

                if self.num_neighbors > 0 and len(edge_id) >= self.num_neighbors:
                    break

            road_map.append(edge_id)
        
        return road_map

    def generate_planar_map(self, samples: List[Node]):
        road_map = [[] for _ in range(len(samples))]
        edge_list = {}
        points = np.array([samp.current for samp in samples])
        tri = Delaunay(points)

        # Get the edges of the Delaunay triangulation
        edges = [tuple(edge) for edge in np.concatenate([tri.simplices[:,[0,1]],tri.simplices[:,[1,2]],tri.simplices[:,[2,0]]],axis=0)]
        selected_edges = []
        for edge in edges:
            if edge in edge_list:
                continue
            edge_list[edge] = 1
            edge_list[edge[::-1]] = 1
            s_pos = points[edge[0]]    
            n_pos = points[edge[1]]
            if not self.in_collision_dda(s_pos, n_pos) \
                and  self.get_cost(s_pos,n_pos) >= self.min_edge_length and  self.get_cost(s_pos,n_pos) <= self.max_edge_length:
                selected_edges.append(edge)

        for edge in selected_edges:
            road_map[edge[0]].append(edge[1])
            road_map[edge[1]].append(edge[0])
        return road_map

    def plan(self):
        pass