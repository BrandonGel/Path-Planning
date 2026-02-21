import argparse
import yaml
from bisect import bisect
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node
from typing import Any, List
import numpy as np
from collections import deque

class State(object):
    def __init__(self, position=(-1,-1), t=0, interval=(0,float('inf'))):
        self.position = tuple(position)
        self.time = t
        self.interval = interval # safe interval (start_time, end_time)

class SippNode(object):
    def __init__(self):
        self.interval_list = [(0, float('inf'))]

    # Split the safety interval with the agent depature time, and agent arrival time
    def split_interval(self, t1,t2,t_buffer=1):
        """
        Function to generate safe-intervals
        """
        for interval in list(self.interval_list):
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
        self.overlapping_collision_list = []
        self.overlapping_being_used = []

    def is_in_safe_interval(self, t1, t2):
        for interval in self.interval_list:
            if t1 >= interval[0] and t2 <= interval[1]:
                return True
        return False

class SippGraph(object):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True):
        self.graph_map = graph_map 
        self.dyn_obstacles = {}
        self.sipp_graph = {}
        if radius > 0:
            self.graph_map.set_constraint_sweep()
        self.radius = radius
        self.velocity = velocity
        self.use_constraint_sweep = use_constraint_sweep
        self._constraint_sweep_cache = {}  # (p1, p2, r) -> (nodes, edges, start_nodes)
        self._constraint_segment_cache = {}  # (p1a, p1b, p2a, p2b, v1, v2, r1, r2) -> bool
        self.init_graph()
        self.init_intervals(dynamic_obstacles)

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
            self.sipp_graph[(src_pos, tgt_pos)] = SippEdge()
            self.sipp_graph[(tgt_pos, src_pos)] = SippEdge()



    def init_intervals(self,dyn_obstacles:dict = {}):
        if not dyn_obstacles or len(dyn_obstacles) == 0: return
        for dyn_name, schedule in dyn_obstacles.items():
            self.dyn_obstacles[dyn_name] = np.array([State(position=(location["x"],location["y"]), t=location["t"]) for location in schedule])
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
   
    def update_intervals(self,plans: List[List[State]] | List[State] | State,dyn_names: List[str]):
        if not plans or len(plans) == 0: return
        for plan,dyn_name in zip(plans,dyn_names):
            self.dyn_obstacles[dyn_name] = np.array([State(position=s.position, t=s.time) for s in plan])
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
                        # print("Vertex: ", vertex_pos, t1,t2)
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
    