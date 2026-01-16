import random
from typing import List

import numpy as np
from scipy.spatial import KDTree
from python_motion_planning.common import  Node
from python_motion_planning.path_planner import BasePathPlanner
import heapq

    
class PRM(BasePathPlanner):

    def __init__(self,*args, num_sample = 3000, num_neighbors = 13.0, min_edge_len = 0.0, max_edge_len = 30.0,**kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_samples = num_sample
        self.num_neighbors = num_neighbors
        self.min_edge_length = min_edge_len
        self.max_edge_length = max_edge_len
        self.sample_list = []
        self.node_index_list = {}
        

    def __str__(self) -> str:
        return "Probabilistic Road Maps (PRM)"
    
    def plan(self):
        self.sample_list = self.generateRandomNodes()
        self.road_map = self.generate_roadmap(samples=self.sample_list)
        path, path_info = self.dijkstra_planning(self.road_map, self.sample_list)
        return path, path_info


    def run(self):
        cost, path, _ = self.plan()
        self.plot.animation(path, str(self), cost, self.sample_list)

    def generateRandomNode(self) -> Node:
        """
        Generate a random node to extend exploring tree.

        Returns:
            node (Node): a random node based on sampling
        """

        point = []
        # Generate random integer point within grid bounds
        for d in range(self.dim):
            d_min, d_max =self.map_.bounds[d][0] -0.5, self.map_.bounds[d][1] - 0.5
            point.append(random.uniform(d_min, d_max))
        point = tuple(point)
        
        return Node(point, None, 0, 0)
    
    def generateRandomNodes(self) -> List[Node]:
        """
        Generate random nodes in the environment.

        Returns:
            List[Node]: List of random nodes
        """
        nodes = []
        while len(nodes) != self.num_samples+1:
            node = self.generateRandomNode()
            if self.map_.is_expandable(self.map_.point_float_to_int(node.current)):
                nodes.append(node)
                self.node_index_list[node] = len(nodes) - 1
        nodes.append(Node(self.start, None, 0, 0))
        self.node_index_list[nodes[-1]] = len(nodes) - 1
        nodes.append(Node(self.goal, None, 0, 0))
        self.node_index_list[nodes[-1]] = len(nodes) - 1
        return nodes
    
    def generate_roadmap(self, samples: List[Node]):
        road_map = []
        sample_kd_tree = KDTree(np.array([samp.current for samp in samples]))

        for i, node_s in zip(range(len(samples)), samples):
            s_pos = node_s.current
            dists, indexes = sample_kd_tree.query(s_pos, k=self.num_samples)
            edge_id = []

            for ii in range(1, len(indexes)):
                n_pos = samples[indexes[ii]].current
                if not self.map_.in_collision(self.map_.point_float_to_int(s_pos), self.map_.point_float_to_int(n_pos)) \
                    and self.get_cost(s_pos,n_pos) >= self.min_edge_length and self.get_cost(s_pos,n_pos) <= self.max_edge_length:
                    edge_id.append(indexes[ii])

                if len(edge_id) >= self.num_neighbors:
                    break

            road_map.append(edge_id)
        
        return road_map

    def dijkstra_planning(self, road_map, samples:List[Node]):
        OPEN = []
        # For Dijkstra, we only use g-value (no heuristic h-value)
        start_node = Node(self.start, None, 0, 0)
        heapq.heappush(OPEN, start_node)
        CLOSED = dict()
        
        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node.current == self.goal:
                CLOSED[node.current] = node
                path, length, cost = self.extract_path(CLOSED)
                return path, {
                    "success": True, 
                    "start": self.start, 
                    "goal": self.goal, 
                    "length": length, 
                    "cost": cost, 
                    "expand": CLOSED
                }
            
            for node_index in road_map[self.node_index_list[node]]: 
                node_n = samples[node_index]
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

