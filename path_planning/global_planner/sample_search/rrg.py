import math
import random
from typing import Union, Dict, List, Tuple, Any, Optional

import numpy as np
import faiss
import heapq

from python_motion_planning.common import BaseMap, Node, TYPES, Grid
from python_motion_planning.path_planner import BasePathPlanner
from scipy.spatial import KDTree

class RRG(BasePathPlanner):
    """
    Class for RRG (Rapidly-exploring Random Graph) path planner.
    RRG maintains a graph structure where nodes can have multiple connections,
    making it suitable for multi-query path planning.

    Args:
        *args: see the parent class.
        max_dist: Maximum connection distance between nodes.
        sample_num: Maximum number of samples to generate.
        goal_sample_rate: Probability of sampling the goal directly.
        discrete: Whether to use discrete or continuous space.
        use_faiss: Whether to use Faiss to accelerate the search.
        *kwargs: see the parent class.

    References:
        [1] Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms
            for optimal motion planning. The International Journal of Robotics Research.

    Examples:

    """

    def __init__(
        self,
        *args,
        min_dist: float = 0.0,
        max_dist: float = 5.0,
        sample_num: int = 100000,
        goal_sample_rate: float = 0.1,
        discrete: bool = False,
        use_faiss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.sample_num = sample_num
        self.goal_sample_rate = goal_sample_rate
        self.discrete = discrete
        self.use_faiss = use_faiss
        self.road_map = []  # Graph structure: list of adjacency lists
        self.node_index_list = {}
        self.nodes = []  # List containing all nodes

    def __str__(self) -> str:
        return "RRG"

    def plan(self) -> Union[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        RRG path planning algorithm implementation.
        Builds a rapidly-exploring random graph where nodes can have multiple connections.

        Returns:
            path: A list containing the path waypoints
            path_info: A dictionary containing path information
        """
        # Initialize graph structure
        road_map = []  # Graph structure: list of adjacency lists
        start_nodes = []  # List to hold all start nodes
        goal_nodes = []  # List to hold all goal nodex

        # Add start node to the graph and necessary lists
        for start_node_coord in self.start:
            start_node = Node(start_node_coord, None, 0, 0)
            start_nodes.append(start_node)
            self.nodes.append(start_node)
            self.node_index_list[start_node] = len(self.nodes) - 1
            road_map.append([])  # Initialize empty adjacency list for this node

        # Add goal node to graph if not already present
        for goal_node_coord in self.goal:
            goal_node = Node(goal_node_coord, None, 0, 0)
            goal_nodes.append(goal_node)
            if tuple(goal_node.current) not in self.nodes:
                self.nodes.append(goal_node)
                self.node_index_list[goal_node] = len(self.nodes) - 1
                road_map.append([])

        # Initialize FAISS index for efficient nearest neighbor search
        faiss_index = faiss.IndexFlatL2(self.dim)
        faiss_nodes = []
        for node in self.nodes:
            self._faiss_add_node(node, faiss_index, faiss_nodes)

        # Main RRG sampling loop
        for _ in range(self.sample_num):
            # Generate random sample node
            node_rand = self._generate_random_node()

            # Skip if node already exists
            if node_rand.current in self.nodes:
                continue

            # Find nearest node in graph
            node_near = self._get_nearest_node(
                self.nodes, node_rand, faiss_index, faiss_nodes
            )

            # Create new node towards random sample
            node_new = self._steer(node_near, node_rand)
            if node_new is None:
                continue

            # Check if edge from nearest to new node is collision-free
            if self.map_.in_collision(
                self.map_.point_float_to_int(node_new.current),
                self.map_.point_float_to_int(node_near.current),
            ):
                continue

            # Add new node to graph
            self.nodes.append(node_new)
            self.node_index_list[node_new] = len(self.nodes) - 1
            road_map.append([])  # Initialize empty adjacency list for this node

            if self.use_faiss:
                self._faiss_add_node(node_new, faiss_index, faiss_nodes)


        # Connect new node to all nearby nodes if collision-free
        for ii in range(len(self.nodes)):
             # RRG key step: Find ALL nearby nodes within max_dist radius
            node_new = self.nodes[ii]
            nearby_nodes = self._get_nearby_nodes(
                self.nodes, node_new, faiss_index, faiss_nodes
            )

            # Connect new node to all nearby nodes if collision-free
            for nearby_node in nearby_nodes:
                # Check if edge is collision-free
                # if not self.map_.in_collision(
                #     self.map_.point_float_to_int(node_new.current),
                #     self.map_.point_float_to_int(nearby_node.current),
                # ):
                    # Add bidirectional edge in graph
                    node_new_idx = self.node_index_list[node_new]
                    nearby_idx = self.node_index_list[nearby_node]
                    road_map[node_new_idx].append(nearby_idx)
                    road_map[nearby_idx].append(node_new_idx)

        # Store graph structure
        self.road_map = road_map


        # Use Dijkstra to find shortest path between each start and goal in the graph
        paths = []  # list to hold all of the paths between the start and goal nodes
        path_info = [{}]
        i = 0  # index of which start/goal we're on
        for start_node in start_nodes:
            path, path_info = self._dijkstra_planning(
                self.road_map,
                self.nodes,
                start_node,
                goal_nodes[i],
            )
            paths.append(path)
            i += 1

        return paths, path_info

    def _generate_random_node(self) -> Node:
        """
        Generate a random node within map bounds as integer grid point.

        Returns:
            node: Generated random node on grid
        """
        # Sample goal directly with specified probability
        if random.random() < self.goal_sample_rate:
            goal_ind = random.choice(range(len(self.goal)))
            return Node(self.goal[goal_ind], None, 0, 0)

        point = []
        # Generate random integer point within grid bounds
        for d in range(self.dim):
            d_min, d_max = -0.5, self.map_.shape[d] - 0.5
            point.append(random.uniform(d_min, d_max))
        point = tuple(point)

        if self.discrete:
            point = self.map_.point_float_to_int(point)

        return Node(point, None, 0, 0)

    def _get_nearest_node(
        self,
        nodes: Dict[Tuple[float, ...], Node],
        node_rand: Node,
        index=None,
        faiss_nodes=None,
    ) -> Node:
        """
        Find the nearest node in the graph to a random sample.

        Args:
            nodes: Current graph of nodes
            node_rand: Random sample node
            index: FAISS index (required when `use_faiss`=True)
            faiss_nodes: List of nodes in FAISS index (required when `use_faiss`=True)

        Returns:
            node: Nearest node in the graph
        """
        # knn search using faiss
        if self.use_faiss and index is not None and faiss_nodes is not None:
            query = np.array(node_rand.current, dtype=np.float32).reshape(1, -1)
            _, indices = index.search(query, 1)
            return faiss_nodes[indices[0][0]]

        # brute force search
        min_dist = float("inf")
        nearest_node = None

        dist = [self.get_cost(node.current, node_rand.current) for node in nodes]
        min_dist = min(dist)
        nearest_node = nodes[dist.index(min_dist)]
        return nearest_node

    def _get_nearby_nodes(
        self,
        nodes: List[Node],
        node_new: Node,
        index=None,
        faiss_nodes=None,
    ) -> List[Tuple[Tuple[float, ...], int]]:
        """
        Find all nodes within max_dist radius of the new node.
        This is the key difference between RRT and RRG - RRG connects to ALL nearby nodes.

        Args:
            nodes: Current graph of nodes
            node_list: List of node positions in order (for index mapping)
            node_new: New node to find neighbors for
            node_new_idx: Index of new node in node_list
            index: FAISS index (required when `use_faiss`=True)
            faiss_nodes: List of nodes in FAISS index (required when `use_faiss`=True)

        Returns:
            nearby_nodes: List of tuples (node_position, node_index) within max_dist
        """
        nearby_nodes = []

        if self.use_faiss and index is not None and faiss_nodes is not None:
            # Use FAISS for radius search
            query = np.array(node_new.current, dtype=np.float32).reshape(1, -1)
            # Search for all nodes within max_dist^2 (L2 distance squared)
            min_dist_sq = self.min_dist**2
            max_dist_sq = self.max_dist**2
            distances, indices = index.search(
                query, min(index.ntotal, len(faiss_nodes))
            )

            for dist_sq, faiss_idx in zip(distances[0][1:], indices[0][1:]):
                if min_dist_sq <=dist_sq <= max_dist_sq and faiss_idx < len(faiss_nodes):
                    nearby_node = faiss_nodes[faiss_idx]
                    nearby_nodes.append(nearby_node)
                else:
                    break
        else:
            # Brute force search for all nodes within radius
            points = np.array([node.current for node in nodes])
            sample_kd_tree = KDTree(points)
            node_new_pos = node_new.current
            _, indexes = sample_kd_tree.query(node_new_pos, k=len(nodes))
            for ii in range(1, len(indexes)):
                node_pos = nodes[indexes[ii]].current
                dist = self.get_cost(node_new_pos, node_pos)
                if self.min_dist <= dist <= self.max_dist:
                    nearby_nodes.append(nodes[indexes[ii]])
                else:
                    break
        return nearby_nodes

    def _steer(self, node_near: Node, node_rand: Node) -> Union[Node, None]:
        """
        Steer from nearest node towards random sample.

        Args:
            node_near: Nearest node in tree
            node_rand: Random sample node

        Returns:
            node: New node in direction of random sample
        """
        # Calculate differences for each dimension
        diffs = [node_rand.current[i] - node_near.current[i] for i in range(self.dim)]

        # Calculate Euclidean distance in n-dimensional space
        dist = math.sqrt(sum(diff**2 for diff in diffs))

        # Handle case where nodes are coincident
        if math.isclose(dist, 0):
            return None

        # If within max distance, use the random node directly
        if dist <= self.max_dist:
            return node_rand

        # Otherwise scale to maximum distance
        scale = self.max_dist / dist
        new_point = [node_near.current[i] + scale * diffs[i] for i in range(self.dim)]
        new_point = tuple(new_point)

        if self.discrete:
            new_point = self.map_.point_float_to_int(new_point)

        return Node(new_point, None, 0, 0)

    def _faiss_add_node(self, node: Node, index, nodes):
        """
        Add a node to the FAISS index.

        Args:
            node: Node to add
            index: FAISS index
            nodes: List of nodes in FAISS index
        """
        vec = np.array(node.current, dtype=np.float32).reshape(1, -1)
        index.add(vec)
        nodes.append(node)

    def _dijkstra_planning(
        self,
        road_map: List[List[int]],
        nodes: List[Node],
        start_node: Node,
        goal_node: Node,
    ) -> Optional[List[int]]:
        """
        Run Dijkstra on an RRG roadmap.

        Returns:
            List of node indices representing the shortest path,
            or None if no path exists.
        """
        OPEN = []
        heapq.heappush(OPEN, start_node)
        CLOSED = dict()

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node.current == goal_node.current:
                CLOSED[node.current] = node
                path, length, cost = self.extract_path(
                    CLOSED, start_node.current, goal_node.current
                )

                return path, {
                    "success": True,
                    "start": self.start,
                    "goal": self.goal,
                    "length": length,
                    "cost": cost,
                    "expand": CLOSED,
                }

            for node_index in road_map[self.node_index_list[node]]:
                node_n = nodes[node_index]
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue

                # For Dijkstra, we only update g-value (no heuristic)
                node_n.g = node.g + self.get_cost(node.current, node_n.current)
                node_n.parent = node.current

                # goal found
                if node_n.current == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break

                # update OPEN list with node sorted by g-value
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node

        self.failed_info[1]["expand"] = CLOSED
        return self.failed_info
