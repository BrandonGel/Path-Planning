import argparse
import yaml
from math import fabs
import heapq
import random
from bisect import bisect
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node
from typing import Any, List
import numpy as np
from collections import deque
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner as SippBaseEnvironment
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import State
from path_planning.multi_agent_planner.centralized.ccbs.sipp import SIPP 
from itertools import count,combinations
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippNode
from copy import deepcopy

# Shared singleton for unconstrained vertex/edge (avoids allocation in hot path)
_UNCONSTRAINED_NODE = SippNode()


class Conflict(object):
    WAIT = 1 # Wait-vertex conflict
    MOVE = 2 # Wait-edge conflict
    def __init__(self):
        self.time_interval_1 = None
        self.time_interval_2 = None
        self.type1 = -1
        self.type2 = -1
        self.agent_1 = ''
        self.agent_2 = ''
        self.location_1a = None
        self.location_1b = None
        self.location_2a = None
        self.location_2b = None
        self.travel_time_1 = 0
        self.travel_time_2 = 0

    def __str__(self):
        return '('+ str(ConflictType[self.type1]) + ', ' + str(ConflictType[self.type2]) + ', ' + str(self.time_interval_1) + ', ' + str(self.time_interval_2) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1a) + ', ' + str(self.location_1b) + ', ' + str(self.location_2a) + ', ' + str(self.location_2b) + ')'

ConflictType = {
    Conflict.WAIT: "WAIT",
    Conflict.MOVE: "MOVE"
}

# Similar to VertexConstraint in CBS
class WaitConstraint(object):
    def __init__(self, interval:tuple, position:tuple, t_buffer:float = 1):
        self.interval = interval
        self.position = position
        self.t_buffer = t_buffer
    def __eq__(self, other):
        return self.interval == other.interval and self.position == other.position
    def __hash__(self):
        return hash(str(self.position)+str(self.interval))
    def __str__(self):
        return '(' + str(self.interval) + ', '+ str(self.position) + ')'

# Similar to EdgeConstraint in CBS
class MoveConstraint(object):
    def __init__(self, interval:tuple, position_1:tuple, position_2:tuple, travel_time:float, t_buffer:float = 1):
        self.interval = interval
        self.position_1 = position_1
        self.position_2 = position_2
        self.travel_time = travel_time
        self.t_buffer = t_buffer
    def __eq__(self, other):
        return self.interval == other.interval and self.position_1 == other.position_1 \
            and self.position_2 == other.position_2
    def __hash__(self):
        return hash(str(self.position_1) + str(self.position_2) + str(self.interval))
    def __str__(self):
        return '(' + str(self.interval) + ', '+ str(self.position_1) +', '+ str(self.position_2) + ')'

class Constraints(object):
    def __init__(self):
        self.wait_constraints = {}
        self.move_constraints = {}

    def add_constraint(self, other):
        """Merge other's constraints into self (intersection of safe intervals)."""
        for wait_key, wait_sipp_node in other.wait_constraints.items():
            if wait_key not in self.wait_constraints:
                self.wait_constraints[wait_key] = SippNode()
            self.wait_constraints[wait_key].merge_safe_intervals(wait_sipp_node)
        for move_key, move_sipp_node in other.move_constraints.items():
            if move_key not in self.move_constraints:
                self.move_constraints[move_key] = SippNode()
            self.move_constraints[move_key].merge_safe_intervals(move_sipp_node)

    def add_wait_constraint(self, other):
        if other.position not in self.wait_constraints:
            self.wait_constraints[other.position] = SippNode()
        self.wait_constraints[other.position].split_interval(other.interval[0], other.interval[1],  other.t_buffer)
    
    def add_move_constraint(self, other):
        if (other.position_1, other.position_2) not in self.move_constraints:
            self.move_constraints[(other.position_1, other.position_2)] = SippNode()
        self.move_constraints[(other.position_1, other.position_2)].split_interval(other.interval[0], other.interval[1], other.t_buffer)

    def is_in_safe_interval(self,key,depart_t, arrive_t= None):
        if key in self.wait_constraints:
            return self.wait_constraints[key].is_in_safe_interval(depart_t, arrive_t)
        if key in self.move_constraints:
            return self.move_constraints[key].is_in_safe_interval(depart_t, arrive_t)
        return True

    def __getitem__(self, key:tuple):
        if key in self.wait_constraints:
            return self.wait_constraints[key]
        if key in self.move_constraints:
            return self.move_constraints[key]
        return _UNCONSTRAINED_NODE

    def __str__(self):
        return "WC: " + str([str(wc) for wc in self.wait_constraints])  + \
            " MC: " + str([str(mc) for mc in self.move_constraints])

class Environment(SippBaseEnvironment):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},agents:list = [],sipp_max_iterations=-1,radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True,heuristic_type : str = 'manhattan',time_limit: float | None = None, max_iterations: int | None = None,verbose: bool = False):
        self.agents = agents
        self.agent_dict = {}
        self.make_agent_dict()
        SippBaseEnvironment.__init__(self, graph_map, dynamic_obstacles,agents,radius,velocity,use_constraint_sweep,heuristic_type,time_limit,max_iterations,verbose)
        assert radius > 0, "Radius must be greater than 0 for CCBS"
        assert velocity > 0, "Velocity must be greater than 0 for CCBS"
        self.sipp = SIPP(self, max_iterations=sipp_max_iterations,verbose=verbose)
        self.collision_radius = (2 * radius) ** 2
        self.constraints = Constraints()
        self.constraint_dict = {agent["name"]: Constraints() for agent in agents}
        self.t_buffer = 1 if radius == 0 else 1e-10
        self.t_epsilon = 1e-10
        self.time_limit = time_limit if time_limit is not None and time_limit > 0 else float('inf')
        self.max_iterations = max_iterations if max_iterations is not None and max_iterations > 0 else float('inf')
        self.verbose = verbose

    def get_initial_state(self,position):
        node =  self.constraints[position]
        return State(position,0, node.interval_list[0])

    def get_earliest_no_collision_arrival_time_body(self,start_t, vertex_interval, edge_interval, start_pos, neighbour,m_time):
        arrive_t = max(start_t, vertex_interval[0],edge_interval[0] + m_time) 
        depart_t = arrive_t - m_time
        if arrive_t > vertex_interval[1]:
            return None

        if not self.constraints.is_in_safe_interval(start_pos,depart_t):
            return None
        if not self.constraints.is_in_safe_interval(neighbour,arrive_t):
            return None
        if not self.constraints.is_in_safe_interval((start_pos,neighbour),depart_t):
            return None
        return arrive_t

    def get_successors(self, state):
        successors = []
        costs = []
        time_taken = []
        neighbour_list = self.get_valid_neighbours(state.position)
        for neighbour_pos in neighbour_list:
            m_time = self.get_mtime(state.position, neighbour_pos)
            start_pos = state.position
            depart_t = state.time 
            start_t = state.time + m_time  # Earliest possible arrival time
            end_t = state.interval[1] + m_time #Latest possible arrival time
            
            for i in self.constraints[neighbour_pos].interval_list:
                # If the interval is outside the possible arrival time, skip the interval
                if i[0] > end_t or i[1] < start_t:
                    continue

                early_depart_t = state.time
                late_depart_t = state.interval[1]
                for i_edge in self.constraints[(start_pos,neighbour_pos)].interval_list:
                    if i_edge[0] > late_depart_t or i_edge[1] < early_depart_t:
                        continue
                    # Skip if edge (departure) interval does not overlap arrival window shifted to departure time
                    if i_edge[1] < i[0] - m_time or i_edge[0] > i[1] - m_time:
                        continue

                    t = self.get_earliest_no_collision_arrival_time_body(start_t, i,i_edge, start_pos, neighbour_pos,m_time)
                    if t is None: # Any collision, skip the interval
                        continue
                    
                    #Get total cost & wait cost for the successor
                    cost = t - state.time 
                    w_cost = cost - m_time

                    # Create the successor state
                    s = State(neighbour_pos, t, i)
                    successors.append(s)
                    costs.append(cost)
                    time_taken.append((w_cost, m_time))
                
        return successors, costs, time_taken

    def check_collision(self, position_1a,velocity1, position_2a,velocity2,tdur):
        pos1a2a = position_2a - position_1a
        vel12 = velocity2 - velocity1
        a = np.dot(vel12, vel12)
        b = 2*np.dot(pos1a2a, vel12)
        c = np.dot(pos1a2a, pos1a2a) - (self.radius*2)**2 + 1e-10
        d = b**2 - 4*a*c

        if a == 0:
            if c < 0:
                return 0,tdur,tdur
            else:
                return False
        if d < 0:
            return False
        t1 = (-b + np.sqrt(d)) / (2*a)
        t2 = (-b - np.sqrt(d)) / (2*a)
        tmin = -b/(2*a)
        
        if t1 < 0 and t2 < 0:
            return False
        if t1 > tdur and t2 > tdur:
            return False
        t1,t2 = min(t1,t2), max(t1,t2)
        t1 = np.clip(t1, 0, tdur)
        t2 = np.clip(t2, 0, tdur)
        tmin = np.clip(tmin, 0, tdur)
        return t1, t2, tmin
        
    def get_conflicts(self, solution,solution_action_cost,get_first_conflict: bool = True):
        result = Conflict()
        conflicts = []

        for agent in solution.keys():
            plan = solution[agent]
            action_cost = solution_action_cost[agent]
            for i in range(len(plan) - 1):
                position_1 = plan[i].position
                position_2 = plan[i+1].position
                action = action_cost[i]
                wait_time, _ = action
                if wait_time > 0:
                    self._get_constraint_sweep_cached(position_1, position_1,self.velocity, 2*self.radius)
                self._get_constraint_sweep_cached(position_1, position_2,self.velocity, 2*self.radius)

        # Precompute per-agent arrays once (avoid recomputing for each pair)
        agent_data = {}
        for agent in solution:
            plan = solution[agent]
            ac = solution_action_cost[agent]
            pos_arr = np.array([p.position for p in plan])
            ac_arr = np.array(ac)
            times = [p.time for p in plan]
            n = len(plan)
            if n >= 2:
                move_costs = ac_arr[:-1, 1].reshape(-1, 1)
                denom = np.where(move_costs != 0, move_costs, 1.0)
                vel_arr = np.zeros((n - 1, pos_arr.shape[1]), dtype=np.float64)
                np.divide(np.diff(pos_arr, axis=0), denom, out=vel_arr, where=move_costs != 0)
            else:
                vel_arr = np.zeros((0, pos_arr.shape[1]), dtype=np.float64)
            agent_data[agent] = {"pos": pos_arr, "vel": vel_arr, "times": times, "ac": list(ac), "plan": plan}

        for agent_1, agent_2 in combinations(solution.keys(), 2):
            d1 = agent_data[agent_1]
            d2 = agent_data[agent_2]
            plan_1 = d1["plan"]
            plan_2 = d2["plan"]
            result = Conflict()
            action_cost_1 = d1["ac"]
            action_cost_2 = d2["ac"]

            times_1 = d1["times"]
            times_2 = d2["times"]
            all_t = sorted(set(times_1 + times_2))
            if len(all_t) < 2:
                continue

            t1_now_idx = 0
            t2_now_idx = 0
            t1_next_idx = 0
            t2_next_idx = 0
            v1_arr = d1["pos"]
            v2_arr = d2["pos"]
            vel_1_arr = d1["vel"]
            vel_2_arr = d2["vel"]
            for k in range(len(all_t) - 1):
                # Global time interval; skip numerically tiny slices
                t_start, t_end = all_t[k], all_t[k + 1]
                duration = t_end - t_start
                
                # Update the time indices for the two agents
                if times_1[t1_next_idx] ==  t_start:
                    t1_now_idx = t1_next_idx
                    t1_next_idx = min(t1_next_idx + 1, len(times_1) - 1)
                if times_2[t2_next_idx] == t_start:
                    t2_now_idx = t2_next_idx
                    t2_next_idx = min(t2_next_idx + 1, len(times_2) - 1)

                v1_now = plan_1[t1_now_idx].position
                v2_now = plan_2[t2_now_idx].position
                v1_next = plan_1[t1_next_idx].position
                v2_next = plan_2[t2_next_idx].position
                e1 = (v1_now, v1_next)
                e2 = (v2_now, v2_next)

                if duration <= self.t_epsilon :
                    continue

                # Get the wait and move costs for the two agents
                wait_cost_1, move_cost_1 = action_cost_1[t1_now_idx]
                wait_cost_2, move_cost_2 = action_cost_2[t2_now_idx]

                # Get the global start and moving times for the two agents
                t1_start = times_1[t1_now_idx]
                t1_moving_start = t1_start + wait_cost_1
                t2_start = times_2[t2_now_idx]
                t2_moving_start = t2_start + wait_cost_2

                # Get the overlapping vertices and edges for the two agents
                overlapping_vertices_1_wait,overlapping_edges_1_wait = {},{}
                if wait_cost_1 > 0:
                    overlapping_vertices_1_wait,overlapping_edges_1_wait = self._get_constraint_sweep_cached(v1_now, v1_now,self.velocity, 2*self.radius)
                overlapping_vertices_1_move,overlapping_edges_1_move = self._get_constraint_sweep_cached(v1_now, v1_next,self.velocity, 2*self.radius)
                overlapping_vertices_1 = [overlapping_vertices_1_wait,overlapping_vertices_1_move]
                overlapping_edges_1 = [overlapping_edges_1_wait,overlapping_edges_1_move]

                overlapping_vertices_2_wait,overlapping_edges_2_wait = {},{}
                if wait_cost_2 > 0:
                    overlapping_vertices_2_wait,overlapping_edges_2_wait = self._get_constraint_sweep_cached(v2_now, v2_now,self.velocity, 2*self.radius)
                overlapping_vertices_2_move,overlapping_edges_2_move = self._get_constraint_sweep_cached(v2_now, v2_next,self.velocity, 2*self.radius)   
                overlapping_vertices_2 = [overlapping_vertices_2_wait,overlapping_vertices_2_move]
                overlapping_edges_2 = [overlapping_edges_2_wait,overlapping_edges_2_move]

                # Get the time intervals for agent 1
                t1_interval = []
                if t_start < t1_moving_start:
                    if t_end >= t1_moving_start:
                        t1_interval = [(t1_start, t1_moving_start),(t1_moving_start, t_end)]
                    else:
                        t1_interval = [(t1_start, t1_moving_start),(t1_moving_start, t1_moving_start)]
                else:
                    t1_interval = [(t_start, t_start),(t_start, t_end)]

                # Get the time intervals for agent 2
                t2_interval = []
                if t_start < t2_moving_start:
                    if t_end >= t2_moving_start:
                        t2_interval = [(t2_start, t2_moving_start),(t2_moving_start, t_end)]
                    else:
                        t2_interval = [(t2_start, t2_moving_start),(t2_moving_start, t2_moving_start)]
                else:
                    t2_interval = [(t_start, t_start),(t_start, t_end)]

                pass
                for it1_idx, t1_int in enumerate(t1_interval):
                    for it2_idx, t2_int in enumerate(t2_interval):
                        if t1_int[0] == t1_int[1] or t2_int[0] == t2_int[1]:
                            continue
                        overlapping_vertices_1it = overlapping_vertices_1[it1_idx]
                        overlapping_vertices_2it = overlapping_vertices_2[it2_idx]
                        overlapping_edges_1it = overlapping_edges_1[it1_idx]
                        overlapping_edges_2it = overlapping_edges_2[it2_idx]
                    
                        if it1_idx == 0 and it2_idx == 0:
                            if v2_now not in overlapping_vertices_1it or v1_now not in overlapping_vertices_2it:
                                continue

                            collision_interval1 = (t_start, t_end)
                            collision_interval2 = (t_start, t_end)

                            result.type1 = Conflict.WAIT
                            result.type2 = Conflict.WAIT
                            result.agent_1 = agent_1
                            result.agent_2 = agent_2
                overlapping_vertices_2 = [overlapping_vertices_2_wait,overlapping_vertices_2_move]
                overlapping_edges_2 = [overlapping_edges_2_wait,overlapping_edges_2_move]

                # Get the time intervals for agent 1
                t1_interval = []
                if t_start < t1_moving_start:
                    if t_end >= t1_moving_start:
                        t1_interval = [(t1_start, t1_moving_start),(t1_moving_start, t_end)]
                    else:
                        t1_interval = [(t1_start, t1_moving_start),(t1_moving_start, t1_moving_start)]
                else:
                    t1_interval = [(t_start, t_start),(t_start, t_end)]

                # Get the time intervals for agent 2
                t2_interval = []
                if t_start < t2_moving_start:
                    if t_end >= t2_moving_start:
                        t2_interval = [(t2_start, t2_moving_start),(t2_moving_start, t_end)]
                    else:
                        t2_interval = [(t2_start, t2_moving_start),(t2_moving_start, t2_moving_start)]
                else:
                    t2_interval = [(t_start, t_start),(t_start, t_end)]

                pass
                for it1_idx, t1_int in enumerate(t1_interval):
                    for it2_idx, t2_int in enumerate(t2_interval):
                        if t1_int[0] == t1_int[1] or t2_int[0] == t2_int[1]:
                            continue
                        overlapping_vertices_1it = overlapping_vertices_1[it1_idx]
                        overlapping_vertices_2it = overlapping_vertices_2[it2_idx]
                        overlapping_edges_1it = overlapping_edges_1[it1_idx]
                        overlapping_edges_2it = overlapping_edges_2[it2_idx]
                    
                        if it1_idx == 0 and it2_idx == 0:
                            if v2_now not in overlapping_vertices_1it or v1_now not in overlapping_vertices_2it:
                                continue

                            collision_interval1 = (t_start, t_end)
                            collision_interval2 = (t_start, t_end)

                            result.type1 = Conflict.WAIT
                            result.type2 = Conflict.WAIT
                            result.agent_1 = agent_1
                            result.agent_2 = agent_2
                            result.location_1a = v1_now
                            result.location_1b = v1_now
                            result.location_2a = v2_now
                            result.location_2b = v2_now
                            result.time_interval_1 = collision_interval1
                            result.time_interval_2 = collision_interval2
                            result.travel_time_1 = 0
                            result.travel_time_2 = 0
                            conflicts.append(result)
                            if get_first_conflict:
                                return conflicts
                            
                        elif it1_idx == 0 and it2_idx == 1:
                            if e2 not in overlapping_edges_1it or v1_now not in overlapping_vertices_2it:
                                continue
                            t_wait1, t_depart1 = t1_int[0], t1_int[1]
                            t_depart2, t_arrive2 = t2_int[0], t2_int[1]

                            v1_now_arr = v1_arr[t1_now_idx]
                            
                            v2_now_arr = v2_arr[t2_now_idx]
                            vel2_now = vel_2_arr[t2_now_idx]
                            v2_delta_t = t_start - t_depart2
                            v2_current = v2_now_arr + vel2_now * v2_delta_t

                            tdur = t_end - t_start
                            collision_time= self.check_collision(v1_now_arr,0, v2_current,vel2_now,tdur)
                            if not collision_time:
                                continue
                            t1,t2,min_collision_time = collision_time
                            collision_interval1 = (t_start+t1, t_start+t2)
                            collision_interval2 = (t_depart2, t_start+min_collision_time)
                            
                            result.type1 = Conflict.WAIT
                            result.type2 = Conflict.MOVE
                            result.agent_1 = agent_1
                            result.agent_2 = agent_2
                            result.location_1a = v1_now
                            result.location_1b = v1_now
                            result.location_2a = v2_now
                            result.location_2b = v2_next
                            result.time_interval_1 = collision_interval1
                            result.time_interval_2 = collision_interval2
                            result.travel_time_1 = 0
                            result.travel_time_2 = t_arrive2 - t_depart2
                            conflicts.append(result)
                            if get_first_conflict:
                                return conflicts
                        elif it1_idx == 1 and it2_idx == 0:
                            if v2_now not in overlapping_vertices_1it or e1 not in overlapping_edges_2it:
                                continue
                            t_depart1, t_arrive1 = t1_int[0], t1_int[1]
                            t_wait2, t_depart2 = t2_int[0], t2_int[1]

                            v1_now_arr = v1_arr[t1_now_idx]
                            vel1_now = vel_1_arr[t1_now_idx]
                            v1_delta_t = t_start - t_depart1
                            v1_current = v1_now_arr + vel1_now * v1_delta_t
                            
                            v2_now_arr = v2_arr[t2_now_idx]

                            tdur = t_end - t_start
                            collision_time = self.check_collision(v1_current,vel1_now, v2_now_arr,0,tdur)
                            if not collision_time:
                                continue
                            t1,t2,min_collision_time = collision_time
                            collision_interval1 = (t_depart1, t_start+min_collision_time)
                            collision_interval2 = (t_start+t1, t_start+t2)

                            result.type1 = Conflict.MOVE
                            result.type2 = Conflict.WAIT
                            result.agent_1 = agent_1
                            result.agent_2 = agent_2
                            result.location_1a = v1_now
                            result.location_1b = v1_next
                            result.location_2a = v2_now
                            result.location_2b = v2_now
                            result.time_interval_1 = collision_interval1
                            result.time_interval_2 = collision_interval2
                            result.travel_time_1 = t_arrive1 - t_depart1
                            result.travel_time_2 = 0
                            conflicts.append(result)
                            if get_first_conflict:
                                return conflicts
                        elif it1_idx == 1 and it2_idx == 1:
                            if e2 not in overlapping_edges_1it or e1 not in overlapping_edges_2it:
                                continue

                            t_depart1, t_arrive1 = t1_int[0], t1_int[1]
                            t_depart2, t_arrive2 = t2_int[0], t2_int[1]
                            
                            v1_now_arr = v1_arr[t1_now_idx]
                            vel1_now = vel_1_arr[t1_now_idx]
                            v1_delta_t = t_start - t_depart1
                            v1_current = v1_now_arr + vel1_now * v1_delta_t
                            
                            v2_now_arr = v2_arr[t2_now_idx]
                            vel2_now = vel_2_arr[t2_now_idx]
                            v2_delta_t = t_start - t_depart2
                            v2_current = v2_now_arr + vel2_now * v2_delta_t
                            tdur = t_end - t_start
                            collision_time = self.check_collision(v1_current,vel1_now, v2_current,vel2_now,tdur)
                            if not collision_time:
                                continue
                            t1,t2,min_collision_time = collision_time
                            collision_interval1 = (t_depart1, t_start+min_collision_time)
                            collision_interval2 = (t_depart2, t_start+min_collision_time)
                            result.type1 = Conflict.MOVE
                            result.type2 = Conflict.MOVE
                            result.agent_1 = agent_1
                            result.agent_2 = agent_2
                            result.location_1a = v1_now
                            result.location_1b = v1_next
                            result.location_2a = v2_now
                            result.location_2b = v2_next
                            result.time_interval_1 = collision_interval1
                            result.time_interval_2 = collision_interval2
                            result.travel_time_1 = t_arrive1 - t_depart1
                            result.travel_time_2 = t_arrive2 - t_depart2
                            conflicts.append(result)
                            if get_first_conflict:
                                return conflicts

        return conflicts

    def check_collision_interval(self, time_interval, interval):
        # No collision if time interval is completely outside the interval
        if time_interval[1] < interval[0] or time_interval[0] > interval[1]:
            return False
        
        # Yes collision if time interval is completely inside the interval
        collision_start_time = max(time_interval[0], interval[0])
        collision_end_time = min(time_interval[1], interval[1])
        return (collision_start_time, collision_end_time)

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}

        # Agent 1 constraints
        if conflict.type1 == Conflict.WAIT:
            w_constraint = WaitConstraint(conflict.time_interval_1, conflict.location_1a, self.t_buffer)
            constraint = Constraints()
            constraint.add_wait_constraint(w_constraint)
            constraint_dict[conflict.agent_1] = constraint
        elif conflict.type1 == Conflict.MOVE:
            m_constraint = MoveConstraint(conflict.time_interval_1, conflict.location_1a, conflict.location_1b, conflict.travel_time_1, self.t_buffer)
            constraint = Constraints()
            constraint.add_move_constraint(m_constraint)
            constraint_dict[conflict.agent_1] = constraint

        # Agent 2 constraints
        if conflict.type2 == Conflict.WAIT:
            w_constraint = WaitConstraint(conflict.time_interval_2, conflict.location_2a, self.t_buffer)
            constraint = Constraints()
            constraint.add_wait_constraint(w_constraint)
            constraint_dict[conflict.agent_2] = constraint
        elif conflict.type2 == Conflict.MOVE:
            m_constraint = MoveConstraint(conflict.time_interval_2, conflict.location_2a, conflict.location_2b, conflict.travel_time_2, self.t_buffer)
            constraint = Constraints()
            constraint.add_move_constraint(m_constraint)
            constraint_dict[conflict.agent_2] = constraint

        return constraint_dict   

    def is_at_goal(self, state: State, goal_state: State):
        return state.is_equal_location(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(agent['start'])
            goal_state = State(agent['goal'])
            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    def compute_solution(self, affected_agent=None, base_solution=None, base_action_cost=None, base_cost=None,find_num_conflicts=False):
        if base_solution is not None and affected_agent is not None:
            solution = dict(base_solution)
            solution_action_cost = dict(base_action_cost)
            solution_cost = dict(base_cost)
            agents_to_plan = [affected_agent]
        else:
            solution = {}
            solution_action_cost = {}
            solution_cost = {}
            agents_to_plan = list(self.agent_dict.keys())

        for agent in agents_to_plan:
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            if find_num_conflicts:
                local_solution, local_action_cost, local_cost = self.sipp.search(agent,solution=solution,solution_action_cost=solution_action_cost)
            else:
                local_solution, local_action_cost, local_cost = self.sipp.search(agent)
            if not local_solution:
                return {}, {}, {}
            solution[agent] = local_solution
            solution_action_cost[agent] = local_action_cost
            solution_cost[agent] = local_cost
        return solution, solution_action_cost, solution_cost

    def _get_constraint_sweep_cached(self, p1, p2,v, r):
        """Cached wrapper for get_constraint_sweep to avoid duplicate queries."""
        key = (p1, p2, v, r)
        if key not in self._constraint_sweep_cache:
            self._constraint_sweep_cache[key] = self.graph_map.get_constraint_sweep(p1, p2,v, r, use_interval=True,get_time_interval=False)
        return self._constraint_sweep_cache[key]
    