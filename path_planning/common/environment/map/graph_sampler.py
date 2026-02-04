from typing import List, Tuple
import numpy as np
from scipy.spatial import KDTree, Delaunay
from path_planning.common.environment.node import Node
from python_motion_planning.common.env.map.grid import Grid
from scipy.spatial.distance import cdist
from itertools import product
from python_motion_planning.common import TYPES
from path_planning.utils.cgal_sweep import CGAL_Sweep
import networkx as nx

class GraphSampler(Grid):
    def __init__(self,*args,start,goal,sample_num=0,num_neighbors = 13.0, min_edge_len = 0.0, max_edge_len = 30.0,use_discrete_space=True,use_constraint_sweep=True,record_sweep=True,use_exact_collision_check=True,**kwargs):
        super().__init__(*args, **kwargs)

        # Check if start and goal are lists, non-empty, and not None
        assert start is not None, "Start must not be None"
        assert goal is not None, "Goal must not be None"
        assert isinstance(start, list), "Start must be a list"
        assert isinstance(goal, list), "Goal must be a list"
        self.start = [tuple(float(pt) for pt in  s) for s in start]
        self.goal = [tuple(float(pt) for pt in g) for g in goal]
        self.sample_num = sample_num
        self.num_total_nodes = sample_num + len(self.start) + len(self.goal)
        self.num_neighbors = num_neighbors
        self.min_edge_length = min_edge_len
        self.max_edge_length = max_edge_len
        self.cost_matrix = None
        self.start_to_all_edges_dict = {}
        self.goal_to_all_edges_dict = {}
        self.node_index_list = {}
        self.start_nodes_index = {}
        self.goal_nodes_index = {}
        self.grid_nodes_index = {}
        self.nodes = []
        self.obstacle_nodes = []
        self.track_with_link = False
        self.road_map = []
        self.use_discrete_space = use_discrete_space
        self.edges = []
        self.edge_weights = []
        self.edge_indices = {}
        self.use_constraint_sweep = use_constraint_sweep
        self.constraint_sweep = CGAL_Sweep(record_sweep=record_sweep,use_exact_collision_check=use_exact_collision_check)
        self.sample_kd_tree = None

    def __str__(self) -> str:
        return "Graph Sampler"

    def set_parameters(self, sample_num, num_neighbors, min_edge_len, max_edge_len):
        self.sample_num = sample_num
        self.num_neighbors = num_neighbors
        self.min_edge_length = min_edge_len
        self.max_edge_length = max_edge_len

    def set_start(self, start):
        if self.use_discrete_space:
            start_pixel = start
        else:
            start_pixel = [self.world_to_map(s,discrete=True) for s in start]
        self.start = [tuple(float(pt) for pt in  s) for s in start]
        for s in start_pixel:
            self.type_map[tuple(s)] = TYPES.START
        

    def set_goal(self, goal):
        if self.use_discrete_space:
            goal_pixel = goal
        else:
            goal_pixel =  [self.world_to_map(g,discrete=True) for g in goal]
        self.goal = [tuple(float(pt) for pt in g) for g in goal]
        for g in goal_pixel:
            self.type_map[tuple(g)] = TYPES.GOAL
    
    def get_start_nodes(self) -> List[Node]:
        return [self.nodes[i] for i in self.start_nodes_index.values()]
    
    def get_goal_nodes(self) -> List[Node]:
        return [self.nodes[i] for i in self.goal_nodes_index.values()]
    
    def get_grid_nodes(self) -> List[Node]:
        return [self.nodes[i] for i in self.grid_nodes_index.values()]

    def get_random_nodes(self) -> List[Node]:
        if len(self.nodes) >= self.sample_num:
            return self.nodes[:self.sample_num]
        print("Warning: number of nodes is less than sample num")
        return self.nodes

    def get_nodes(self) -> List[Node]:
        return self.nodes
    
    def get_node_index(self, node: Node) -> int:
        return self.node_index_list[node]

    def calculate_edges(self,roadmap: List[List[int]]):
        edges = set()
        edge_indices = {}
        for ii in range(len(roadmap)):
            for jj in roadmap[ii]:
                if (ii,jj) not in edges:
                    edges.add((ii,jj))
                    edge_indices[len(edges)-1] = (ii,jj)
        return list(edges), edge_indices
    
    def set_obstacle_map(self, obstacles: np.ndarray):
        obstacles = np.array(obstacles)
        if len(obstacles) == 0:
            return
        if len(obstacles[0]) == 2:
            self.type_map[obstacles[:,0], obstacles[:,1]] = TYPES.OBSTACLE 
        elif len(obstacles[0]) == 3:
            self.type_map[obstacles[:,0], obstacles[:,1], obstacles[:,2]] = TYPES.OBSTACLE 
        else:
            raise ValueError(f"Unsupported dimensions: {len(obstacles.shape)}")
        for obstacle in obstacles:
            node = Node(tuple(obstacle),None,0,0)
            self.obstacle_nodes.append(node)

    def get_obstacle_map(self) -> np.ndarray:
        return self.type_map.data == TYPES.OBSTACLE

    def get_obstacle_nodes(self) -> List[Node]:
        return self.obstacle_nodes

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

    def get_neighbors(self, node: Node, **kwargs) -> List[Node]:
        """
        Get the neighbors of a node.
        """
        if self.road_map is None or node not in self.node_index_list:
            return []
        neighbors =  [self.nodes[i] for i in self.road_map[self.node_index_list[node]]]
        if self.track_with_link:
            neighbors = [node.link(neighbor) for neighbor in neighbors]
        return neighbors


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

    def in_collision_point(self, point: Tuple[float, ...]) -> bool:
        """
        Check if the point is in collision.
        """
        return not self.is_expandable(self.world_to_map(point,discrete=True))

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

        
        # Check if end point has the correct dimension
        if  len(p1) != dim and len(p2) != dim:
            raise ValueError(f"End point must have dimension {dim}")
        
        # Convert to grid coordinates using map's point_float_to_int
        p1_grid = np.array(self.world_to_map(p1,discrete=True), dtype=int)
        p2_grid = np.array(self.world_to_map(p2,discrete=True), dtype=int)
        
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
        if np.all(np.abs(tMax) < 1e-10):
            tMax = abs_delta*1e-10
        
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

            tied_dims = np.where(np.abs(tMax - tMax[min_dim]) < 1e-10)[0]
            if len(tied_dims) > 1:
                diff = p2_grid - current_tile
                min_dim = np.argmax(np.abs(diff))
            
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

            # tied_dims = np.where(np.abs(tMax - tMax[min_dim]) < 1e-10)[0]

            # # Step in ALL tied dimensions
            # for min_dim in tied_dims:
            #     # Step in that dimension
            #     new_tile = current_tile.copy()
            #     new_tile[min_dim] += step[min_dim]

            #     # Update tMax
            #     if tDelta[min_dim] != float('inf'):
            #         tMax[min_dim] += tDelta[min_dim]
                
            #     # Check collision immediately (early termination)
            #     tile_tuple = tuple(new_tile)
            #     if tile_tuple not in seen:
            #         seen.add(tile_tuple)
            #         if not self.is_expandable(tile_tuple):
            #             return True  # Found collision, terminate early

            #  # Step in ALL tied dimensions
            # for min_dim in tied_dims:
            #     # Step in that dimension
            #     current_tile[min_dim] += step[min_dim]
                
            # # Check collision immediately (early termination)
            # tile_tuple = tuple(current_tile)
            # if tile_tuple not in seen:
            #     seen.add(tile_tuple)
            #     if not self.is_expandable(tile_tuple):
            #         return True  # Found collision, terminate early
            if tile_tuple == tuple(p2_grid):
                return False
        
        return False  # No collision found

    def generateRandomNodes(self, generate_grid_nodes = False):
        num_nodes = 0
        bounds = np.array(self.bounds)
        nodes = []

        while num_nodes < self.sample_num:
            normalized_points = np.random.random((self.sample_num,self.dim))
            points = normalized_points * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
            pixels = [self.world_to_map(point,discrete=True) for point in points]
            for ii in range(self.sample_num):
                if self.use_discrete_space:
                    current = tuple(self.world_to_map(points[ii])) # Convert to discrete space but not into int
                else:
                    current = tuple(points[ii])
                node = Node(current,None,0,0)
                if self.is_expandable(tuple(pixels[ii])):
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
                    if self.use_discrete_space:
                        current = tuple(float(i) for i in grid_coords_tuple)
                    else:
                        current =tuple(float(i) for i in self.map_to_world(grid_coords_tuple))
                    node = Node(current, None, 0, 0)
                    nodes.append(node)
                    self.node_index_list[node] = len(nodes) - 1
                    self.grid_nodes_index[node] = len(nodes)-1
        
        for start in self.start:
            node = Node(tuple(start),None,0,0)
            if node in self.node_index_list:
                self.start_nodes_index[node] = self.node_index_list[node]
                continue
            nodes.append(node)
            self.node_index_list[node] = len(nodes) - 1
            self.start_nodes_index[node] = len(nodes) - 1
        for goal in self.goal:
            node = Node(tuple(goal),None,0,0)
            if node in self.node_index_list:
                self.goal_nodes_index[node] = self.node_index_list[node]
                continue
            nodes.append(node)
            self.node_index_list[node] = len(nodes) - 1
            self.goal_nodes_index[node] = len(nodes) - 1
        

        for start in self.start:
            start_node = Node(tuple(start),None,0,0)
            start_idx = self.node_index_list[start_node]
            other_costs = [self.get_distance(start_node.current, nodes[i].current) for i in range(len(nodes))]
            self.start_to_all_edges_dict[start_idx] = list(zip(range(len(nodes)), other_costs))
        for goal in self.goal:
            goal_node = Node(tuple(goal),None,0,0)
            goal_idx = self.node_index_list[goal_node]
            other_costs = [self.get_distance(goal_node.current, nodes[i].current) for i in range(len(nodes))]
            self.goal_to_all_edges_dict[goal_idx] = list(zip(range(len(nodes)), other_costs))

        # Update total node count after all nodes are added
        self.num_total_nodes = len(nodes)
        self.cost_matrix = cdist(np.array([node.current for node in nodes]), np.array([node.current for node in nodes]), metric='euclidean')
        self.nodes = nodes
        return nodes

    def generate_roadmap(self, samples: List[Node]):
        road_map = []
        points = np.array([samp.current for samp in samples])
        sample_kd_tree = KDTree(points)
        self.sample_kd_tree = sample_kd_tree
        for i, node_s in zip(range(len(samples)), samples):
            s_pos = node_s.current
            indexes = sample_kd_tree.query_ball_point(s_pos, self.max_edge_length)
            edge_id = []

            for ii in range(1, len(indexes)):
                node_n = samples[indexes[ii]]
                n_pos = samples[indexes[ii]].current

                if self.get_cost(node_s,node_n) < self.min_edge_length:
                    continue

                if self.get_cost(node_s,node_n) > self.max_edge_length:
                    break

                if not self.in_collision(s_pos, n_pos) and not self.in_collision(n_pos, s_pos):
                    edge_id.append(indexes[ii])

                if self.num_neighbors > 0 and len(edge_id) >= self.num_neighbors:
                    break

            road_map.append(edge_id)
        self.road_map = road_map
        self.edges, self.edge_indices = self.calculate_edges(road_map)
        self.edge_weights = [self.get_cost(self.nodes[i], self.nodes[j]) for i, j in self.edges]
        return road_map

    def generate_planar_map(self, samples: List[Node]):
        planar_map = [[] for ii in range(len(samples))]
        edge_list = {}
        points = np.array([samp.current for samp in samples])
        tri = Delaunay(points)
        self.sample_kd_tree = KDTree(points)

        # Get the edges of the Delaunay triangulation
        edges = [tuple(edge) for edge in np.concatenate([tri.simplices[:,[0,1]],tri.simplices[:,[1,2]],tri.simplices[:,[2,0]]],axis=0).tolist()]
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
            if not self.in_collision(s_pos, n_pos) and not self.in_collision(n_pos, s_pos) \
                and  self.get_cost(node_s,node_n) >= self.min_edge_length and  self.get_cost(node_s,node_n) <= self.max_edge_length:
                selected_edges.append(edge)

        for edge in selected_edges:
            planar_map[edge[0]].append(edge[1])
            planar_map[edge[1]].append(edge[0])

        self.road_map = planar_map
        self.edges, self.edge_indices = self.calculate_edges(planar_map)
        self.edge_weights = [self.get_cost(self.nodes[i], self.nodes[j]) for i, j in self.edges]
        return planar_map

    def set_constraint_sweep(self):
        self.constraint_sweep.set_graph([node.current for node in self.nodes],self.edges)

    def get_constraint_sweep(self, p1: tuple[float,float], p2: tuple[float,float],v: float = 0.0, r: float = 0.5):
        if not self.use_constraint_sweep:
            return set()
        overlapping_edges = self.constraint_sweep.overlapping_graph_elements_cgal(p1, p2,v, r)
        edges_locations = set((self.nodes[edge_idx[0]].current, self.nodes[edge_idx[1]].current) for edge_idx in overlapping_edges)
        return edges_locations

    def get_constraint_segment(self, p1a: Tuple[float, ...],p1b: Tuple[float, ...],p2a: Tuple[float, ...],p2b: Tuple[float, ...],v: float = 0.0,r: float = 0.5) -> bool:
        p1a = np.array(p1a)
        p2a = np.array(p2a)
        p1b = np.array(p1b)
        p2b = np.array(p2b)
        if v == 0.0:
            v1 = p1b-p1a
            v2 = p2b-p2a
            t_dur = 1.0
        else:
            dist1 = np.linalg.norm(p1b-p1a)
            dist2 = np.linalg.norm(p2b-p2a)
            t_dur1 = dist1/v
            t_dur2 = dist2/v
            t_dur = min(t_dur1, t_dur2)
            v1 = (p1b-p1a)/t_dur1 if dist1 > 0.0 else np.zeros(self.dim)
            v2 = (p2b-p2a)/t_dur2 if dist2 > 0.0 else np.zeros(self.dim)
        r = 2*r
        r_vec = p2a-p1a
        vel = v2-v1
        tmin = np.clip(-np.dot(vel,r_vec)/(np.dot(vel,vel)+1e-10),0.0,t_dur)
        r_min_vec = r_vec + vel*tmin
        if np.linalg.norm(r_min_vec) < r:
            return True
        return False

    def point_float_to_int(self, point: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        Convert a point from float to integer coordinates.

        Args:
            point: a point in float coordinates
        
        Returns:
            point: a point in integer coordinates
        """
        point_int = []
        for d in range(self.dim):
            point_int.append(max(0, min(self.shape[d] - 1, int(round(point[d]+1e-10)))))
        point_int = tuple(point_int)
        return point_int

    def read_from_numpy(self,position:np.ndarray,edge_index:np.ndarray,edge_attr:np.ndarray, use_roadmap:bool = True):
        self.clear_data()
        position = position.tolist()
        node_start_idx = len(position) - len(self.start) - len(self.goal)
        node_goal_idx = len(position) - len(self.goal)
        nodes= []
        for ii in range(len(position)):
            nodes.append(Node(current=tuple(position[ii])))
            self.node_index_list[nodes[ii]] = len(nodes) - 1

            if ii >= node_start_idx and ii < node_goal_idx:
                self.start_nodes_index[nodes[ii]] = len(nodes) - 1
            elif ii >= node_goal_idx:
                self.goal_nodes_index[nodes[ii]] = len(nodes) - 1
        
        if use_roadmap: 
            edge_index = edge_index.tolist()
            edge_attr = edge_attr.tolist()
            self.edges = [tuple(e) for e in edge_index]
            self.edge_indices = {i: (e[0], e[1]) for i, e in enumerate(self.edges)}
            self.edge_weights = edge_attr
            self.road_map= [[] for ii in range(len(position))]
            for i, j in self.edges:
                self.road_map[i].append(j)

            for start in self.start:
                start_node = Node(tuple(start),None,0,0)
                start_idx = self.node_index_list[start_node]
                other_costs = [self.get_distance(start_node.current, nodes[i].current) for i in range(len(nodes))]
                self.start_to_all_edges_dict[start_idx] = list(zip(range(len(nodes)), other_costs))
            for goal in self.goal:
                goal_node = Node(tuple(goal),None,0,0)
                goal_idx = self.node_index_list[goal_node]
                other_costs = [self.get_distance(goal_node.current, nodes[i].current) for i in range(len(nodes))]
                self.goal_to_all_edges_dict[goal_idx] = list(zip(range(len(nodes)), other_costs))

        # Update total node count after all nodes are added
        self.num_total_nodes = len(nodes)
        self.cost_matrix = cdist(np.array([node.current for node in nodes]), np.array([node.current for node in nodes]), metric='euclidean')
        self.nodes = nodes


    def clear_data(self):
        self.start_to_all_edges_dict = {}
        self.goal_to_all_edges_dict = {}
        self.node_index_list = {}
        self.start_nodes_index = {}
        self.goal_nodes_index = {}
        self.grid_nodes_index = {}
        self.nodes = []
        self.obstacle_nodes = []
        self.road_map = []
        self.edges = []
        self.edge_weights = []
        self.edge_indices = {}

    def plan(self):
        pass