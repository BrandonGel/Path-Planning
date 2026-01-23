import math
import random
from typing import Union, Dict, List, Tuple, Any, Optional

import numpy as np
import faiss
import heapq

from python_motion_planning.common import BaseMap, Node, TYPES, Grid
from python_motion_planning.path_planner import BasePathPlanner


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
        max_dist: float = 5.0,
        sample_num: int = 100000,
        goal_sample_rate: float = 0.1,
        discrete: bool = False,
        use_faiss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_dist = max_dist
        self.sample_num = sample_num
        self.goal_sample_rate = goal_sample_rate
        self.discrete = discrete
        self.use_faiss = use_faiss
        self.road_map = []  # Graph structure: list of adjacency lists

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
        # nodes: dictionary mapping node position to Node object
        # road_map: adjacency list representation of the graph
        nodes = {}
        node_list = []  # List to maintain order and index mapping
        road_map = []  # Graph structure: list of adjacency lists

        # Add start node to the graph
        start_node = Node(self.start, None, 0, 0)
        nodes[tuple(start_node.current)] = start_node
        node_list.append(start_node.current)
        road_map.append([])  # Initialize empty adjacency list

        # Add goal node to graph if not already present
        goal_node = Node(self.goal, None, 0, 0)
        if tuple(goal_node.current) not in nodes:
            nodes[tuple(goal_node.current)] = goal_node
            node_list.append(goal_node.current)
            road_map.append([])

        # Initialize FAISS index for efficient nearest neighbor search
        # if self.use_faiss: Commented this out since we seem to always use FAISS
        faiss_index = faiss.IndexFlatL2(self.dim)
        faiss_nodes = []
        for node_pos in node_list:
            # print("Node pos (right before added by FAISS):", node_pos)
            self._faiss_add_node(nodes[tuple(node_pos)], faiss_index, faiss_nodes)

        # Main RRG sampling loop
        for _ in range(self.sample_num):
            # Generate random sample node
            node_rand = self._generate_random_node()

            # Skip if node already exists
            if node_rand.current in nodes:
                continue

            # Find nearest node in graph
            # print("Nodes (before call to get nearest node):", nodes)
            # print("Node rand:", node_rand)
            # print("Faiss index:", faiss_index)
            # print("Faiss nodes:", faiss_nodes)
            node_near = self._get_nearest_node(
                nodes, node_rand, faiss_index, faiss_nodes
            )

            # Create new node towards random sample
            node_new = self._steer(node_near, node_rand)
            if node_new is None:
                continue

            # print("Node new:", node_new.current[0])
            # print("Node near:", node_near.current[0])
            # Check if edge from nearest to new node is collision-free
            if self.map_.in_collision(
                self.map_.point_float_to_int(node_new.current[0]),
                self.map_.point_float_to_int(node_near.current[0]),
            ):
                continue

            # print("ABOUT TO ADD A NEW NODE")
            # print("New node:", node_new)
            # print("new_node.current:", node_new.current[0])
            # Add new node to graph
            node_new_idx = len(node_list)
            nodes[node_new.current[0]] = node_new
            node_list.append(node_new.current)
            road_map.append([])  # Initialize empty adjacency list

            # What is faiss, what is it here for, what does it do?
            if self.use_faiss:
                self._faiss_add_node(node_new, faiss_index, faiss_nodes)

            # RRG key step: Find ALL nearby nodes within max_dist radius
            nearby_nodes = self._get_nearby_nodes(
                nodes, node_list, node_new, node_new_idx, faiss_index, faiss_nodes
            )

            # Connect new node to all nearby nodes if collision-free
            for nearby_pos, nearby_idx in nearby_nodes:
                if nearby_pos == node_new.current:
                    continue

                # print("Node new:", node_new.current[0])
                # print("Nearby pos:", nearby_pos)
                # Check if edge is collision-free
                if not self.map_.in_collision(
                    self.map_.point_float_to_int(node_new.current[0]),
                    self.map_.point_float_to_int(nearby_pos),
                ):
                    # Add bidirectional edge in graph
                    road_map[node_new_idx].append(nearby_idx)
                    road_map[nearby_idx].append(node_new_idx)

            # Need to read through the code from here down and see what is necessary and what is not

        # Store graph structure
        self.road_map = road_map
        self.failed_info[1]["expand"] = nodes
        self.failed_info[1]["road_map"] = road_map
        self.failed_info[1]["node_list"] = node_list

        # print("Start node:", node_list.index(start_node.current))
        # print("Goal Node:", node_list.index(goal_node.current))
        # print("Node list:", node_list)
        # Use Dijkstra to find shortest path in the constructed graph and return
        path, path_info = self._dijkstra_planning(
            self.road_map,
            node_list,
            node_list.index(start_node.current),
            node_list.index(goal_node.current),
        )
        return path, path_info

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

        # print("Nodes (in get nearest node):", nodes)
        # print("Nodes values:", nodes.values())

        for node in nodes.values():
            # print("Node current:", node.current[0])
            # print("Node random:", node_rand.current)
            dist = math.sqrt(
                (node.current[0][0] - node_rand.current[0]) ** 2
                + (node.current[0][1] - node_rand.current[1]) ** 2
            )
            # print("Distance:", dist)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def _get_nearby_nodes(
        self,
        nodes: Dict[Tuple[float, ...], Node],
        node_list: List[Tuple[float, ...]],
        node_new: Node,
        node_new_idx: int,
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
            max_dist_sq = self.max_dist**2
            distances, indices = index.search(
                query, min(index.ntotal, len(faiss_nodes))
            )

            for dist_sq, faiss_idx in zip(distances[0], indices[0]):
                if dist_sq <= max_dist_sq and faiss_idx < len(faiss_nodes):
                    nearby_node = faiss_nodes[faiss_idx]
                    # Find index in node_list
                    try:
                        nearby_idx = node_list.index(nearby_node.current)
                        if nearby_idx != node_new_idx:
                            nearby_nodes.append((nearby_node.current, nearby_idx))
                    except ValueError:
                        continue
        else:
            # Brute force search for all nodes within radius
            for idx, node_pos in enumerate(node_list):
                if idx == node_new_idx or node_pos == node_new.current:
                    continue
                # print("Node new:", node_new.current[0])
                # print("Node pos:", node_pos[0])
                dist = self.get_cost(node_new.current[0], node_pos[0])
                if dist <= self.max_dist:
                    nearby_nodes.append((node_pos[0], idx))

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
        # print("Node random:", node_rand.current)
        # print("Node near:", node_near.current[0])
        diffs = [
            node_rand.current[i] - node_near.current[0][i] for i in range(self.dim)
        ]

        # Calculate Euclidean distance in n-dimensional space
        dist = math.sqrt(sum(diff**2 for diff in diffs))

        # Handle case where nodes are coincident
        if math.isclose(dist, 0):
            # print("Returned none")
            return None

        # If within max distance, use the random node directly
        if dist <= self.max_dist:
            # print("Returned random node directly")
            # print("Node_rand:", node_rand.current)
            new_node_rand = Node([node_rand.current], None, 0, 0)
            # print("New random node:", new_node_rand)
            # return node_rand
            return new_node_rand

        # Otherwise scale to maximum distance
        scale = self.max_dist / dist
        new_point = [
            node_near.current[0][i] + scale * diffs[i] for i in range(self.dim)
        ]
        new_point = tuple(new_point)

        if self.discrete:
            new_point = self.map_.point_float_to_int(new_point)

        # print("New point:", new_point)
        return Node([new_point], None, 0, 0)

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
        node_list: List[Tuple[float, ...]],
        start_idx: int,
        goal_idx: int,
    ) -> Optional[List[int]]:
        """
        Run Dijkstra on an RRG roadmap.

        Returns:
            List of node indices representing the shortest path,
            or None if no path exists.
        """
        n = len(node_list)

        dist = [math.inf] * n
        parent = [-1] * n
        visited = [False] * n

        dist[start_idx] = 0.0
        pq = [(0.0, start_idx)]  # (distance, node_index)

        while pq:
            curr_dist, u = heapq.heappop(pq)

            if visited[u]:
                continue
            visited[u] = True

            # Early exit if we reached the goal
            if u == goal_idx:
                break

            for v in road_map[u]:
                # print(node_list[u])
                # print(node_list[v])
                # Edge cost = Euclidean distance
                cost = math.dist(node_list[u][0], node_list[v][0])
                alt = curr_dist + cost

                if alt < dist[v]:
                    dist[v] = alt
                    parent[v] = u
                    heapq.heappush(pq, (alt, v))

        # No path found
        if parent[goal_idx] == -1 and start_idx != goal_idx:
            return None

        # Reconstruct path (goal → start)
        path = []
        curr = goal_idx
        while curr != -1:
            path.append(curr)
            curr = parent[curr]

        path.reverse()

        length = dist[goal_idx]

        return path, {
            "success": True,
            "start": self.start,
            "goal": self.goal,
            "length": length,
            "cost": cost,
        }
