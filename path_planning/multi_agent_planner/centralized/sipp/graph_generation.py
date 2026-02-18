import argparse
import yaml
from bisect import bisect
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node
from typing import Any, List
from collections import deque

class State(object):
    def __init__(self, position=(-1,-1), t=0, interval=(0,float('inf'))):
        self.position = tuple(position)
        self.time = t
        self.interval = interval # safe interval (start_time, end_time)

class SippNode(object):
    def __init__(self):
        self.interval_list = [(0, float('inf'))]
        self.f = float('inf')
        self.g = float('inf')
        self.parent_state = State()

    # Split the safety interval with the agent depature time, and agent arrival time
    def split_interval(self, t1,t2,t_buffer=1):
        """
        Function to generate safe-intervals
        """
        for interval in self.interval_list:
            if t2 == float('inf'):
                if t1<=interval[0]:
                    self.interval_list.remove(interval)
                elif t1>interval[1]:
                    continue
                else:
                    self.interval_list.remove(interval)
                    self.interval_list.append((interval[0], t1-t_buffer))
            else:
                if t1 == interval[0]:
                    self.interval_list.remove(interval)
                    if t2 <= interval[1]:
                        self.interval_list.append((t2, interval[1]))
                elif t1 == interval[1]:
                    self.interval_list.remove(interval)
                    if t1-t_buffer >= interval[0]:
                        self.interval_list.append((interval[0],t1-t_buffer))
                elif bisect(interval,t1) == 1:
                    self.interval_list.remove(interval)
                    self.interval_list.append((interval[0], t1-t_buffer))
                    self.interval_list.append((t2, interval[1]))
            self.interval_list.sort()

    def is_in_safe_interval(self, t):
        for interval in self.interval_list:
            if t >= interval[0] and t <= interval[1]:
                return True
        return False

class SippEdge(SippNode):
    def __init__(self):
        super().__init__()

class SippGraph(object):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True):
        self.graph_map = graph_map 
        self.dyn_obstacles = dynamic_obstacles
        self.sipp_graph = {}
        if radius > 0:
            self.graph_map.set_constraint_sweep()
        self.radius = radius
        self.velocity = velocity
        self.use_constraint_sweep = use_constraint_sweep
        self._constraint_sweep_cache = {}  # (p1, p2, r) -> (nodes, edges, start_nodes)
        self._constraint_segment_cache = {}  # (p1a, p1b, p2a, p2b, v1, v2, r1, r2) -> bool
        self.init_graph()
        self.init_intervals()

    def init_graph(self):
        for node in self.graph_map.nodes:
            node_sipp_dict = {node.current:SippNode()}
            self.sipp_graph.update(node_sipp_dict)

        # Initialize SIPP edges keyed by endpoint positions (p1, p2),
        # to match the keys returned by GraphSampler.get_constraint_sweep.
        for edge in self.graph_map.edges:
            src_idx, tgt_idx = edge
            src_pos = self.graph_map.nodes[src_idx].current
            tgt_pos = self.graph_map.nodes[tgt_idx].current
            edge_key = (src_pos, tgt_pos)
            edge_sipp_dict = {edge_key: SippEdge()}
            self.sipp_graph.update(edge_sipp_dict)

    def get_cost(self, position1, position2):
        cost = 1
        if self.velocity > 0:
            cost = self.graph_map.get_cost(Node(tuple(position1)), Node(tuple(position2)))
            cost = cost / self.velocity
        return cost

    def init_intervals(self):
        if not self.dyn_obstacles or len(self.dyn_obstacles) == 0: return
        for schedule in self.dyn_obstacles.values():
            # for location in schedule:
            for i in range(len(schedule)):
                location = schedule[i]
                position = (location["x"],location["y"])
                t = max(0,location["t"])
                
                last_t = i == len(schedule)-1

                if self.radius > 0:
                    if last_t:
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, self.radius)
                    else:
                        next_location = schedule[i+1]
                        next_position = (next_location["x"],next_location["y"])
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, next_position,self.velocity, self.radius)
                    for vertex_pos, vertex_interval in overlapping_vertices.items():
                        t_start,t_end = vertex_interval
                        t1 = t+t_start
                        t2 = t+t_end 
                        self.sipp_graph[vertex_pos].split_interval(t1, t2,0)
                    for edge_pos, edge_interval in overlapping_edges.items():
                        t_start,t_end = edge_interval
                        t1 = t+t_start
                        t2 = t+t_end 
                        self.sipp_graph[edge_pos].split_interval(t1, t2,0)
                else:
                    t1 = t
                    t2 = t1 + 1 if not last_t else float('inf')
                    self.sipp_graph[position].split_interval(t1, t2,1)

        # Update the intervals of the SIPP graph based on the agent's plan (treated as dynamic obstacles)
   
    def update_intervals(self,plans: List[List[State]] | List[State] | State):
        if not plans or len(plans) == 0: return
        for plan in plans:
            # for location in schedule:
            for i in range(len(plan)):
                location = plan[i]
                position = location.position
                t = location.time
                last_t = i == len(plan)-1

                if self.radius > 0:
                    if last_t:
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, self.radius)
                    else:
                        next_location = plan[i+1]
                        next_position = next_location.position
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, next_position,self.velocity, self.radius)
                    for vertex_pos, vertex_interval in overlapping_vertices.items():
                        t_start,t_end = vertex_interval
                        t1 = t+t_start
                        t2 = t+t_end 
                        self.sipp_graph[vertex_pos].split_interval(t1, t2,0)
                    for edge_pos, edge_interval in overlapping_edges.items():
                        t_start,t_end = edge_interval
                        t1 = t+t_start
                        t2 = t+t_end 
                        self.sipp_graph[edge_pos].split_interval(t1, t2,0)
                else:
                    t1 = t
                    t2 = t1 + 1 if not last_t else float('inf')
                    self.sipp_graph[position].split_interval(t1, t2,1)

    def is_valid_position(self, position):
        return not self.graph_map.in_collision_point(position)

    def get_valid_neighbours(self, position):
        neighbors = []
        node = Node(tuple[Any, ...](position))
        nodes = self.graph_map.get_neighbors(node)

        # Move action
        for node in nodes:
            if self.is_valid_position(node.current):
                neighbors.append(node.current)
        return neighbors

    def _get_constraint_sweep_cached(self, p1, p2,v, r):
        """Cached wrapper for get_constraint_sweep to avoid duplicate queries."""
        key = (p1, p2, v, r)
        if key not in self._constraint_sweep_cache:
            self._constraint_sweep_cache[key] = self.graph_map.get_constraint_sweep(p1, p2,v, r, use_interval=True)
        return self._constraint_sweep_cache[key]

    def clear_sipp_graph_values(self,no_clear_interval: bool = False):
        for node in self.sipp_graph.values():
            node.g = float('inf')
            node.f = float('inf')
            node.parent_state = State()
            if not no_clear_interval:
                node.interval_list = [(0, float('inf'))]
        if not no_clear_interval:
            for edge in self.sipp_graph.values():
                edge.interval_list = [(0, float('inf'))]

    