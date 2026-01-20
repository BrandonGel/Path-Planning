
from typing import Union, List, Tuple, Dict, Any, Iterable
from python_motion_planning.common import  Node as base_node
from typing import List
import numpy as np
from scipy.spatial import KDTree, Delaunay
from python_motion_planning.path_planner import BasePathPlanner
from python_motion_planning.common.env.map.grid import Grid
from scipy.spatial.distance import cdist
from itertools import product

class Node(base_node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = None
        self.neighbors = []

    def add_neighbor(self, neighbor: 'Node'):
        self.neighbors.append(neighbor)

    def get_neighbor(self, index: int) -> 'Node':
        return self.neighbors[index]

class GraphSampler(Grid):
    def __init__(self,*args,start,goal,sample_num=0,num_neighbors = 13.0, min_edge_len = 0.0, max_edge_len = 30.0,**kwargs):
        super().__init__(*args, **kwargs)

        # Check if start and goal are lists, non-empty, and not None
        assert start is not None, "Start must not be None"
        assert goal is not None, "Goal must not be None"
        assert isinstance(start, list), "Start must be a list"
        assert isinstance(goal, list), "Goal must be a list"
        self.start = start
        self.goal = goal
        self.sample_num = sample_num
        self.num_total_nodes = sample_num + len(self.start) + len(self.goal)
        self.num_neighbors = num_neighbors
        self.min_edge_length = min_edge_len
        self.max_edge_length = max_edge_len
        self.cost_matrix = None
        self.node_index_list = {}
        self.start_nodes = {}
        self.goal_nodes = {}
        self.nodes = []

    def __str__(self) -> str:
        return "Graph Sampler"

    def set_parameters(self, sample_num, num_neighbors, min_edge_len, max_edge_len):
        self.sample_num = sample_num
        self.num_neighbors = num_neighbors
        self.min_edge_length = min_edge_len
        self.max_edge_length = max_edge_len

    def set_start(self, start):
        self.start = start
        for start in self.start:
            node = Node(tuple(start),None,0,0)
            if node in self.node_index_list:
                continue
            self.nodes.append(node)
            self.node_index_list[node] = len(self.nodes) - 1
            self.start_nodes[node] = len(self.nodes) - 1
        

    def set_goal(self, goal):
        self.goal = goal
        for goal in self.goal:
            node = Node(tuple(goal),None,0,0)
            if node in self.node_index_list:
                continue
            self.nodes.append(node)
            self.node_index_list[node] = len(self.nodes) - 1
            self.goal_nodes[node] = len(self.nodes) - 1
    
    def get_cost(self, p1_node: tuple, p2_node: tuple) -> float:
        """
        Get the cost between two nodes. (default: distance defined in the map)

        Args:
            p1: Start node.
            p2: Goal node.
        """
        if self.cost_matrix is None:
            
            return self.get_distance(p1_node.current, p2_node.current)
        else:
            p1_index = self.node_index_list[p1_node]
            p2_index = self.node_index_list[p2_node]
            return self.cost_matrix[p1_index, p2_index]

    def get_neighbors(self, node: Node) -> List[Node]:
        """
        Get the neighbors of a node.
        """
        return node.neighbors

    def line_of_sight(self, p1: Tuple[float, ...], p2: Tuple[float, ...]) -> bool:
        """
        Check if the line of sight between two points is in collision.
        
        Args:
            p1: Start point of the line.
            p2: End point of the line.
        """
        dim = self.dim
        if len(p1) != dim or len(p2) != dim:
            raise ValueError(f"Points must have dimension {dim}")
        
        # Convert to grid coordinates using map's point_float_to_int
        p1_grid = np.array(self.world_to_map(p1,discrete=True), dtype=int)
        p2_grid = np.array(self.world_to_map(p2,discrete=True), dtype=int)
        current_tile = p1_grid.copy()
        
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
                if not self.is_expandable(tile_tuple):
                    break # Found collision, terminate early
        seen_tiles = [tuple(int(x) for x in tile) for tile in seen]
        return seen_tiles

    def in_collision(self, p1: Tuple[float, ...], p2 : Tuple[float, ...] = None) -> bool:
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
        
        # Check if start point has the correct dimension
        dim = self.dim
        if len(p1) != dim:
            raise ValueError(f"Start point must have dimension {dim}")

        # Convert to grid coordinates using map's point_float_to_int for start point
        p1_grid = np.array(self.world_to_map(p1,discrete=True), dtype=int)
        # Early exit: check start tile is expandable
        if not self.is_expandable(tuple(p1_grid)):
            return True
        if p2 is None:
            return False
        
        # Check if end point has the correct dimension
        if len(p2) != dim:
            raise ValueError(f"End point must have dimension {dim}")
        
        # Convert to grid coordinates using map's point_float_to_int for end point
        p2_grid = np.array(self.world_to_map(p2,discrete=True), dtype=int)
        # Early exit: check end tile is expandable
        if not self.is_expandable(tuple(p2_grid)):
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
        current_tile = p1_grid.copy()
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
                if not self.is_expandable(tile_tuple):
                    return True  # Found collision, terminate early
        
        return False  # No collision found

    def generateRandomNodes(self, generate_grid_nodes = False):
        num_nodes = 0
        bounds = np.array(self.bounds)
        nodes = []

        while num_nodes < self.sample_num:
            normalized_points = np.random.random((self.sample_num,self.dim))
            points = normalized_points * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
            for point in points:
                node = Node(tuple(point),None,0,0)
                if self.is_expandable(self.world_to_map(node.current,discrete=True)):
                    nodes.append(node)
                    self.node_index_list[node] = len(nodes) - 1
                    num_nodes += 1
                if num_nodes == self.sample_num:
                    break

        if generate_grid_nodes:
            # Iterate through all grid points in the mesh
            # Get grid shape (number of cells in each dimension)
            if hasattr(self, 'shape'):
                grid_shape = self.shape
            else:
                # Fallback: calculate shape from bounds and resolution
                bounds = np.array(self.bounds)
                resolution = getattr(self, 'resolution', 1.0)
                grid_shape = tuple(int((bounds[d][1] - bounds[d][0]) / resolution) for d in range(self.dim))
            
            # Generate all grid coordinate combinations using itertools.product
            grid_ranges = [range(grid_shape[d]) for d in range(self.dim)]
            
            # Iterate through all combinations of grid coordinates
            for grid_coords in product(*grid_ranges):
                # Convert to tuple for indexing
                grid_coords_tuple = tuple(grid_coords)
                
                # Check if this grid cell is expandable (not in collision)
                if self.is_expandable(grid_coords_tuple):
                    # Adjust the world coordinates to avoid precision and rounding issues
                    world_coords = tuple(i - 1e-10 for i in self.map_to_world(grid_coords_tuple))
                    node = Node(world_coords, None, 0, 0)
                    nodes.append(node)
                    self.node_index_list[node] = len(nodes) - 1
        
        for start in self.start:
            node = Node(tuple(start),None,0,0)
            nodes.append(node)
            self.node_index_list[node] = len(nodes) - 1
            self.start_nodes[node] = len(nodes) - 1
        for goal in self.goal:
            node = Node(tuple(goal),None,0,0)
            nodes.append(node)
            self.node_index_list[node] = len(nodes) - 1
            self.goal_nodes[node] = len(nodes) - 1
        
        # Update total node count after all nodes are added
        self.num_total_nodes = len(nodes)
        self.cost_matrix = cdist(np.array([node.current for node in nodes]), np.array([node.current for node in nodes]), metric='euclidean')
        self.nodes = nodes
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
                node_n = samples[indexes[ii]]
                n_pos = samples[indexes[ii]].current

                if self.get_cost(node_s,node_n) < self.min_edge_length:
                    continue

                if self.get_cost(node_s,node_n) > self.max_edge_length:
                    break

                if not self.in_collision(s_pos, n_pos):
                    edge_id.append(indexes[ii])

                if self.num_neighbors > 0 and len(edge_id) >= self.num_neighbors:
                    break

            road_map.append(edge_id)
        
        return road_map

    def generate_planar_map(self, samples: List[Node]):
        planar_map = [[] for _ in range(len(samples))]
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
            node_s = samples[edge[0]]
            node_n = samples[edge[1]]
            s_pos = points[edge[0]]    
            n_pos = points[edge[1]]
            if not self.in_collision(s_pos, n_pos) \
                and  self.get_cost(node_s,node_n) >= self.min_edge_length and  self.get_cost(node_s,node_n) <= self.max_edge_length:
                selected_edges.append(edge)

        for edge in selected_edges:
            planar_map[edge[0]].append(edge[1])
            planar_map[edge[1]].append(edge[0])
        return planar_map

    def set_neighbors(self, roadmap: List[List[int]]):
        for i, edges in enumerate(roadmap):
            for j in edges:
                self.nodes[i].add_neighbor(self.nodes[j])

    def plan(self):
        pass