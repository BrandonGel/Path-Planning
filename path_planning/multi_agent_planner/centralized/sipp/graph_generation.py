import argparse
import yaml
from bisect import bisect
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node
from typing import Any, List, Tuple
import numpy as np
from collections import deque

class State(object):
    def __init__(self, position=(-1,-1), t=0, interval=(0,float('inf'))):
        self.position = tuple(position)
        self.time = t
        self.interval = interval # safe interval (start_time, end_time)
    
    def __eq__(self, other):
        return self.position == other.position and self.time == other.time and self.interval == other.interval
    def __hash__(self):
        return hash((self.position, self.time, self.interval))
    def __str__(self):
        return str((self.position, self.time, self.interval))
    def __repr__(self):
        return str((self.position, self.time, self.interval))
    def is_equal_location(self, other):
        return self.position == other.position

class SippNode(object):
    def __init__(self):
        self.interval_list = [(0, float('inf'))]

    # Split the safety interval with the agent depature time, and agent arrival time
    def split_interval(self, t1, t2, t_buffer=1e-10):
        """
        Function to generate safe-intervals
        """
        interval_list = []
        # Apply buffer to the blocked window to ensure safety margins
        b_start = t1 - t_buffer
        b_end = t2 + t_buffer

        for s_start, s_end in self.interval_list:
            # Case 1: Blocked window is entirely after this interval
            if b_start >= s_end:
                interval_list.append((s_start, s_end))
                
            # Case 2: Blocked window is entirely before this interval
            elif b_end <= s_start:
                interval_list.append((s_start, s_end))
                
            # Case 3: Overlap occurs
            else:
                # Check for a left remnant
                if b_start > s_start:
                    interval_list.append((s_start, b_start))
                
                # Check for a right remnant
                if b_end < s_end:
                    interval_list.append((b_end, s_end))
        self.interval_list = sorted(interval_list)

    def is_in_safe_interval(self, depart_t, arrive_t= None):
        if arrive_t is None:
            arrive_t = depart_t
        lo, hi = 0, len(self.interval_list) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = self.interval_list[mid]
            if start <= depart_t and arrive_t <= end:
                return True
            elif depart_t < start:
                hi = mid - 1
            else:
                lo = mid + 1
        return False

    def copy(self):
        """Return a new SippNode with the same safe intervals."""
        n = SippNode()
        n.interval_list = list(self.interval_list)
        return n

    def merge_safe_intervals(self, other):
        """
        Set self.interval_list to the intersection of self's and other's safe
        intervals (so that a time is safe only if it is safe in both).
        """
        a = self.interval_list
        b = other.interval_list
        result = []
        i, j = 0, 0
        while i < len(a) and j < len(b):
            a_lo, a_hi = a[i]
            b_lo, b_hi = b[j]
            lo = max(a_lo, b_lo)
            hi = min(a_hi, b_hi)
            if lo <= hi:
                result.append((lo, hi))
            if a_hi <= b_hi:
                i += 1
            else:
                j += 1
        self.interval_list = result


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
        self._valid_neighbours_cache = {}

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
            self.sipp_graph[(src_pos, tgt_pos)] = SippNode()
            self.sipp_graph[(tgt_pos, src_pos)] = SippNode()

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
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, 2*self.radius)
                        next_t = float('inf')
                        for vertex_pos, vertex_interval in overlapping_vertices.items():
                            self.sipp_graph[vertex_pos].split_interval(t, next_t)
                        for edge_pos, edge_interval in overlapping_edges.items():
                            self.sipp_graph[edge_pos].split_interval(t, next_t)
                        continue
                    else:
                        next_location = schedule[i + 1]
                        next_position = (next_location["x"], next_location["y"])
                        overlapping_vertices, overlapping_edges = self._get_constraint_sweep_cached(position, next_position, self.velocity, 2 * self.radius)
                        next_t = next_location["t"]
                    for vertex_pos, vertex_interval in overlapping_vertices.items():
                        t_start,t_end = vertex_interval
                        t1 = t+t_start
                        t2 = t+t_end 
                        self.sipp_graph[vertex_pos].split_interval(t1, t2)
                    for edge_pos, edge_interval in overlapping_edges.items():
                        t_start,t_end = edge_interval
                        t1 = max(0,t+t_start)
                        t2 = max(0,t+t_end)
                        self.sipp_graph[edge_pos].split_interval(t1, t2)
                else:
                    t1 = t
                    t2 = t1 + 1 if not last_t else float('inf')
                    self.sipp_graph[position].split_interval(t1, t2,1)

        # Update the intervals of the SIPP graph based on the agent's plan (treated as dynamic obstacles)
   
    def update_intervals(self,plans: List[List[State]] | List[State] | State,action_costs: List[List[Tuple[float,float]]] | List[Tuple[float,float]] | Tuple[float,float],dyn_names: List[str]):
        if not plans or len(plans) == 0: return
        for plan,action_cost,dyn_name in zip(plans,action_costs,dyn_names):
            dyn_plan = []
            for i in range(len(plan)):
                dyn_plan.append(State(position=plan[i].position, t=plan[i].time))
                if action_cost[i][0] > 0:
                    dyn_plan.append(State(position=plan[i].position, t=plan[i].time + action_cost[i][0]))
            self.dyn_obstacles[dyn_name] = dyn_plan
            # for location in schedule:
            for i in range(len(plan)):
                location = plan[i]
                position = location.position
                t = location.time
                last_t = i == len(plan)-1

                if self.radius > 0:
                    # Last time step
                    if last_t:
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, 2*self.radius)
                        next_t = float('inf')
                        for vertex_pos, vertex_interval in overlapping_vertices.items():
                            t1,t2 = vertex_interval
                            self.sipp_graph[vertex_pos].split_interval(t, next_t)
                        for edge_pos, edge_interval in overlapping_edges.items():
                            self.sipp_graph[edge_pos].split_interval(t, next_t)
                        continue
                    
                    # Intermediate time step between two locations
                    next_location = plan[i+1]
                    next_position = next_location.position
                    next_t = next_location.time
                    wait_time, move_time = action_cost[i]
                    if wait_time:
                        t0 = t+wait_time
                        overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, position,self.velocity, 2*self.radius)
                        for vertex_pos, vertex_interval in overlapping_vertices.items():
                            self.sipp_graph[vertex_pos].split_interval(t, t0)
                        for edge_pos, edge_interval in overlapping_edges.items():
                            self.sipp_graph[edge_pos].split_interval(t, t0,0)
                    else:
                        t0 = t
                    overlapping_vertices,overlapping_edges = self._get_constraint_sweep_cached(position, next_position,self.velocity, 2*self.radius)
                    for vertex_pos, vertex_interval in overlapping_vertices.items():
                        t_start,t_end = vertex_interval
                        t1 = t0+t_start
                        t2 = t0+t_end 
                        self.sipp_graph[vertex_pos].split_interval(t1, t2)
                    for edge_pos, edge_interval in overlapping_edges.items():
                        t_start,t_end = edge_interval
                        t1 = max(0,t0+t_start)
                        t2 = max(0,t0+t_end)                        
                        if t1 == 0 and t2 == 0:
                            continue
                        self.sipp_graph[edge_pos].split_interval(t1, t2)
                    
                else:
                    t1 = t
                    t2 = t1 + 1 if not last_t else float('inf')
                    self.sipp_graph[position].split_interval(t1, t2,1)

    def is_valid_position(self, position):
        return not self.graph_map.in_collision_point(position)

    def get_valid_neighbours(self, position):
        if position in self._valid_neighbours_cache:
            return self._valid_neighbours_cache[position]
        neighbors = []
        node = Node(tuple(position))
        nodes = self.graph_map.get_neighbors(node)

        # Move action
        for node in nodes:
            if self.is_valid_position(node.current):
                neighbors.append(node.current)
        self._valid_neighbours_cache[position] = neighbors
        return neighbors

    def _get_constraint_sweep_cached(self, p1, p2,v, r):
        """Cached wrapper for get_constraint_sweep to avoid duplicate queries."""
        key = (p1, p2, v, r)
        if key not in self._constraint_sweep_cache:
            self._constraint_sweep_cache[key] = self.graph_map.get_constraint_sweep(p1, p2,v, r, use_interval=True,get_time_interval=True)
        return self._constraint_sweep_cache[key]
    