import math
import numpy as np
from python_motion_planning.global_planner.sample_search.sample_search import SampleSearcher
from scipy.spatial import KDTree
from typing import List
from python_motion_planning.utils import Env, Node

class Node2:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," +\
               str(self.cost) + "," + str(self.parent_index)
    
class PRM(SampleSearcher):

    def __init__(self, start: tuple, goal: tuple, env: Env, num_sample = 3000, num_neighbors = 13.0, max_edge_len = 30.0) -> None:
        super().__init__(start, goal, env)

        self.num_samples = num_sample
        self.num_neighbors = num_neighbors
        self.max_edge_length = max_edge_len
        self.sample_list = []
        

    def __str__(self) -> str:
        return "Probabilistic Road Maps (PRM)"
    
    def plan(self):
        self.sample_list = self.generateRandomNodes()
        self.road_map = self.generate_roadmap(samples=self.sample_list)
        rx, ry,expand = self.dijkstra_planning(self.road_map, self.sample_list)
        path = [(x,y) for (x,y) in zip(rx,ry)]
        cost = self.calculatePathCost(path)

        return cost, path, None


    def run(self):
        cost, path, _ = self.plan()
        self.plot.animation(path, str(self), cost, self.sample_list)

    def generateRandomNode(self) -> Node:
        """
        Generate a random node to extend exploring tree.

        Returns:
            node (Node): a random node based on sampling
        """
        
        current = (np.random.uniform(self.delta, self.env.x_range - self.delta),
                np.random.uniform(self.delta, self.env.y_range - self.delta))
        return Node(current, None, 0, 0)
    
    def generateRandomNodes(self) -> List[Node]:
        """
        Generate random nodes in the environment.

        Returns:
            List[Node]: List of random nodes
        """
        nodes = []
        while len(nodes) != self.num_samples+1:
            node = self.generateRandomNode()
            if not self.isInsideObs(node):
                nodes.append(node)
        nodes.append(self.start)
        nodes.append(self.goal)
        return nodes
    
    def calculatePathCost(self, path) -> float:
        """
        Calculate the total cost of the path.

        Parameters:
            path (List[Node]): List of nodes representing the path.

        Returns:
            float: Total cost of the path.
        """
        cost = 0.0
        for i in range(1, len(path)):
            cost += ((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)**0.5  # Assuming dist is a method to calculate distance between nodes
        return cost
    
    def generate_roadmap(self, samples: List[Node]):
        road_map = []
        sx, sy = [i.x for i in samples], [i.y for i in samples]
        sample_kd_tree = KDTree(np.vstack((sx, sy)).T)

        for (i, ix, iy) in zip(range(len(samples)), sx, sy):
            dists, indexes = sample_kd_tree.query([ix, iy], k=self.num_samples)
            edge_id = []

            for ii in range(1, len(indexes)):
                nx = sx[indexes[ii]]
                ny = sy[indexes[ii]]

                node_ixiy = Node((ix,iy))
                node_nxny = Node((nx,ny))
                if not self.isCollision(node_ixiy, node_nxny) and ((ix - nx)**2 + (iy-ny)**2)**0.5 <= self.max_edge_length:
                    edge_id.append(indexes[ii])

                if len(edge_id) >= self.num_neighbors:
                    break

            road_map.append(edge_id)
        
        return road_map

    def dijkstra_planning(self, road_map, samples:List[Node]):
        sx = self.start.x
        sy = self.start.y
        gx = self.goal.x
        gy = self.goal.y

        sample_x = [i.x for i in samples]
        sample_y = [i.y for i in samples]

        start_node = Node2(sx, sy, 0.0, -1)
        goal_node = Node2(gx, gy, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[len(road_map)-2] = start_node

        path_found = True

        while True:
            if not open_set:
                path_found = False
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            if c_id == (len(road_map) - 1):
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]
            # Add it to the closed set
            closed_set[c_id] = current
            # expand search grid based on motion model
            for i in range(len(road_map[c_id])):
                n_id = road_map[c_id][i]
                dx = sample_x[n_id] - current.x
                dy = sample_y[n_id] - current.y
                d = math.hypot(dx, dy)
                node = Node2(sample_x[n_id], sample_y[n_id],
                            current.cost + d, c_id)

                if n_id in closed_set:
                    continue
                # Otherwise if it is already in the open set
                if n_id in open_set:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id].cost = node.cost
                        open_set[n_id].parent_index = c_id
                else:
                    open_set[n_id] = node

        if path_found is False:
            return [], [], []

        # generate final course
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index

        return rx, ry,closed_set


    


    