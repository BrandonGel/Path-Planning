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
        # if (other.position_1, other.position_2) not in self.move_constraints:
        #     self.move_constraints[(other.position_1, other.position_2)] = SippNode()
        # self.move_constraints[(other.position_1, other.position_2)].split_interval(other.interval[0], other.interval[1], other.t_buffer)

        if other.position_2 not in self.wait_constraints:
            self.wait_constraints[other.position_2] = SippNode()
        self.wait_constraints[other.position_2].split_interval(other.interval[1], other.interval[1]+other.travel_time, other.t_buffer)

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
        return SippNode()

    def __str__(self):
        return "WC: " + str([str(wc) for wc in self.wait_constraints])  + \
            " MC: " + str([str(mc) for mc in self.move_constraints])

class Environment(SippBaseEnvironment):
    def __init__(self, graph_map: GraphSampler,dynamic_obstacles:dict = {},agents:list = [],sipp_max_iterations=-1,radius:float = 0.0,velocity:float = 0.0,use_constraint_sweep:bool = True,verbose:bool = False):
        self.agents = agents
        self.agent_dict = {}
        self.make_agent_dict()
        SippBaseEnvironment.__init__(self, graph_map, dynamic_obstacles,agents,radius,velocity,use_constraint_sweep)
        self.sipp = SIPP(self, max_iterations=sipp_max_iterations,verbose=verbose)
        self.collision_radius = (2 * radius) ** 2
        self.constraints = Constraints()
        self.constraint_dict = {agent["name"]: Constraints() for agent in agents}
        self.t_buffer = 1 if radius == 0 else 0


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
            
            for i in self.constraints[neighbour_pos].interval_list   :
                # If the interval is outside the possible arrival time, skip the interval
                if i[0] > end_t or i[1] < start_t:
                    continue

                # Get the earliest no collision arrival time
                t = self.get_earliest_no_collision_arrival_time(depart_t, start_t, i, start_pos, neighbour_pos)
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

    def get_earliest_no_collision_arrival_time(self,depart_t, start_t, interval, start_pos, neighbour):
        arrive_t = max(start_t, interval[0]) 
        if arrive_t >= interval[0] and arrive_t <= interval[1]:
            # Departure time for edge traversal (agent may have waited at start_pos)
            if not self.constraints.is_in_safe_interval(start_pos,depart_t,arrive_t):
                return None

            if not self.constraints.is_in_safe_interval((start_pos,neighbour),depart_t,arrive_t):
                return None
            return arrive_t
        return None
    
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
                
        for agent_1, agent_2 in combinations(solution.keys(), 2):
            plan_1 = solution[agent_1]
            plan_2 = solution[agent_2]
            result = Conflict()
            action_cost_1 = solution_action_cost[agent_1]
            action_cost_2 = solution_action_cost[agent_2]

            times_1 = [p.time for p in plan_1]
            times_2 = [p.time for p in plan_2]
            all_t = sorted(set(times_1 + times_2))
            if len(all_t) < 2:
                continue
            
            # Iterate over all the relevant time steps and check for conflicts
            t1_next_idx = 0
            t2_next_idx = 0
            for k in range(len(all_t) - 1):
                # Global time interval
                t_start, t_end = all_t[k], all_t[k + 1]
                duration = t_end - t_start
                if duration <= 0:
                    continue
                
                # Update the time indices for the two agents
                if times_1[t1_next_idx] ==  t_start:
                    t1_next_idx = min(t1_next_idx + 1, len(times_1) - 1)
                if times_2[t2_next_idx] == t_start:
                    t2_next_idx = min(t2_next_idx + 1, len(times_2) - 1)

                t1_now_idx = max(0, t1_next_idx-1)
                t2_now_idx = max(0, t2_next_idx-1)
                v1_now = plan_1[t1_now_idx].position
                v2_now = plan_2[t2_now_idx].position
                v1_next = plan_1[t1_next_idx].position
                v2_next = plan_2[t2_next_idx].position
                e1 = (v1_now, v1_next)
                e2 = (v2_now, v2_next)

                # Get the wait and move costs for the two agents
                wait_cost_1, _ = action_cost_1[t1_now_idx]
                wait_cost_2, _ = action_cost_2[t2_now_idx]

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
            
                for it1_idx, t1_int in enumerate(t1_interval):
                    for it2_idx, t2_int in enumerate(t2_interval):
                        if t1_int[0] == t1_int[1] or t2_int[0] == t2_int[1]:
                            continue
                        overlapping_vertices_1it = overlapping_vertices_1[it1_idx]
                        overlapping_vertices_2it = overlapping_vertices_2[it2_idx]
                        overlapping_edges_1it = overlapping_edges_1[it1_idx]
                        overlapping_edges_2it = overlapping_edges_2[it2_idx]
                    
                        if it1_idx == 0 and it2_idx == 0:
                            interval_1 = overlapping_vertices_1it.get(v2_now)
                            interval_2 = overlapping_vertices_2it.get(v1_now)
                            if interval_1 is not None and interval_2 is not None:
                                t_stationary1 = t1_int[0]
                                t_stationary2 = t2_int[0]

                                # Get the time intervals for agent 1 and agent 2 w.r.t. each other
                                t_stationary1_wrt_2 = t_stationary1 - t_stationary2
                                t_stationary2_wrt_1 = t_stationary2 - t_stationary1
                                t_int1_wrt2 = (t_stationary1_wrt_2, t_stationary1_wrt_2)
                                t_int2_wrt1 = (t_stationary2_wrt_1, t_stationary2_wrt_1)

                                # Check for collision conflict
                                vertex_collision_interval_1 = self.check_collision_interval(t_int1_wrt2, interval_1)
                                vertex_collision_interval_2 = self.check_collision_interval(t_int2_wrt1, interval_2)
                                edge_conflict = vertex_collision_interval_1 and vertex_collision_interval_2
                                if edge_conflict:
                                    result.type1 = Conflict.WAIT
                                    result.type2 = Conflict.WAIT
                                    result.agent_1 = agent_1
                                    result.agent_2 = agent_2
                                    result.location_1a = v1_now
                                    result.location_1b = v1_now
                                    result.location_2a = v2_now
                                    result.location_2b = v2_now
                                    result.time_interval_1 = (t_stationary1, t_stationary1+vertex_collision_interval_1[1])
                                    result.time_interval_2 = (t_stationary2, t_stationary2+vertex_collision_interval_2[1])
                                    result.travel_time_1 = 0
                                    result.travel_time_2 = 0
                                    if get_first_conflict:
                                        return result
                                    conflicts.append(result)
                        elif it1_idx == 0 and it2_idx == 1:
                            interval_1 = overlapping_edges_1it.get(e2)
                            interval_2 = overlapping_vertices_2it.get(v1_now)
                            if interval_1 is not None and interval_2 is not None:
                                t_stationary1 = t1_int[0]
                                t_depart2, t_arrive2 = t2_int[0], t2_int[1]

                                # Get the time intervals for agent 1 and agent 2 w.r.t. each other
                                t_stationary1_wrt_2 = t_stationary1 - t_depart2
                                t_depart2_wrt_1 = t_depart2 - t_stationary1
                                t_arrive2_wrt_1 = t_arrive2 - t_stationary1
                                t_int1_wrt2 = (t_stationary1_wrt_2, t_stationary1_wrt_2)
                                t_int2_wrt1 = (t_depart2_wrt_1, t_arrive2_wrt_1)
                                
                                # Check for collision conflict
                                vertex_collision_interval_1 = self.check_collision_interval(t_int1_wrt2, interval_1)
                                edge_collision_interval_2 = self.check_collision_interval(t_int2_wrt1, interval_2)

                                edge_conflict = vertex_collision_interval_1 and edge_collision_interval_2
                                if edge_conflict:
                                    result.type1 = Conflict.WAIT
                                    result.type2 = Conflict.MOVE
                                    result.agent_1 = agent_1
                                    result.agent_2 = agent_2
                                    result.location_1a = v1_now
                                    result.location_1b = v1_now
                                    result.location_2a = v2_now
                                    result.location_2b = v2_next
                                    result.time_interval_1 = (t_stationary1,  t_depart2+edge_collision_interval_2[1])
                                    result.time_interval_2 = (t_depart2, t_depart2+edge_collision_interval_2[1])
                                    result.travel_time_1 = 0
                                    result.travel_time_2 = t_arrive2 - t_depart2
                                    if get_first_conflict:
                                        return result
                                    conflicts.append(result)
                        elif it1_idx == 1 and it2_idx == 0:
                            interval_1 = overlapping_vertices_1it.get(v2_now)
                            interval_2 = overlapping_edges_2it.get(e1)
                            if interval_1 is not None and interval_2 is not None:
                                t_depart1, t_arrive1 = t1_int[0], t1_int[1]
                                t_stationary2 = t2_int[0]

                                # Get the time intervals for agent 1 and agent 2 w.r.t. each other
                                t_depart1_wrt_2 = t_depart1 - t_stationary2
                                t_arrive1_wrt_2 = t_arrive1 - t_stationary2
                                t_stationary2_wrt_1 = t_stationary2 - t_depart1
                                t_int1_wrt2 = (t_depart1_wrt_2, t_arrive1_wrt_2)
                                t_int2_wrt1 = (t_stationary2_wrt_1, t_stationary2_wrt_1)

                                # Check for collision conflict                    
                                edge_collision_interval_1 = self.check_collision_interval(t_int1_wrt2, interval_1)
                                vertex_collision_interval_2 = self.check_collision_interval(t_int2_wrt1, interval_2)

                                edge_conflict = edge_collision_interval_1 and vertex_collision_interval_2
                                if edge_conflict:
                                    result.type1 = Conflict.MOVE
                                    result.type2 = Conflict.WAIT
                                    result.agent_1 = agent_1
                                    result.agent_2 = agent_2
                                    result.location_1a = v1_now
                                    result.location_1b = v1_next
                                    result.location_2a = v2_now
                                    result.location_2b = v2_now
                                    result.time_interval_1 = (t_depart1,  t_stationary2+edge_collision_interval_1[1])
                                    result.time_interval_2 = (t_stationary2, t_stationary2+edge_collision_interval_1[1])
                                    result.travel_time_1 = t_arrive1 - t_depart1
                                    result.travel_time_2 = 0
                                    if get_first_conflict:
                                        return result
                                    conflicts.append(result)
                        elif it1_idx == 1 and it2_idx == 1:
                            interval_1 = overlapping_edges_1it.get(e2)
                            interval_2 = overlapping_edges_2it.get(e1)
                            if interval_1 is not None and interval_2 is not None:
                                t_depart1, t_arrive1 = t1_int[0], t1_int[1]
                                t_depart2, t_arrive2 = t2_int[0], t2_int[1]

                                # Get the time intervals for agent 1 and agent 2 w.r.t. each other
                                t_depart1_wrt_2 = t_depart1 - t_depart2
                                t_arrive1_wrt_2 = t_arrive1 - t_depart2
                                t_depart2_wrt_1 = t_depart2 - t_depart1
                                t_arrive2_wrt_1 = t_arrive2 - t_depart1
                                t_int1_wrt2 = (t_depart1_wrt_2, t_arrive1_wrt_2)
                                t_int2_wrt1 = (t_depart2_wrt_1, t_arrive2_wrt_1)
                                
                                # Check for collision conflict
                                edge_collision_interval_1 = self.check_collision_interval(t_int1_wrt2, interval_1)
                                edge_collision_interval_2 = self.check_collision_interval(t_int2_wrt1, interval_2)
                                edge_conflict = edge_collision_interval_1 and edge_collision_interval_2
                                if edge_conflict:
                                    result.type1 = Conflict.MOVE
                                    result.type2 = Conflict.MOVE
                                    result.agent_1 = agent_1
                                    result.agent_2 = agent_2
                                    result.location_1a = v1_now
                                    result.location_1b = v1_next
                                    result.location_2a = v2_now
                                    result.location_2b = v2_next
                                    result.time_interval_1 = (t_depart1, t_depart1+edge_collision_interval_1[1])
                                    result.time_interval_2 = (t_depart2, t_depart2+edge_collision_interval_2[1])
                                    result.travel_time_1 = t_arrive1 - t_depart1
                                    result.travel_time_2 = t_arrive2 - t_depart2
                                    if get_first_conflict:
                                        return result
                                    conflicts.append(result)
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

    def compute_solution(self):
        # Clear caches for fresh solution attempt
        solution = {}
        solution_action_cost = {}
        solution_cost = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            local_solution, local_action_cost, local_cost = self.sipp.search(agent)
            if not local_solution:
                return False, False, float('inf')
            solution.update({agent:local_solution})
            solution_action_cost.update({agent:local_action_cost})
            solution_cost[agent] = local_cost
        return solution, solution_action_cost, solution_cost

    def _get_constraint_sweep_cached(self, p1, p2,v, r):
        """Cached wrapper for get_constraint_sweep to avoid duplicate queries."""
        key = (p1, p2, v, r)
        if key not in self._constraint_sweep_cache:
            self._constraint_sweep_cache[key] = self.graph_map.get_constraint_sweep(p1, p2,v, r, use_interval=True,get_time_interval=True)
        return self._constraint_sweep_cache[key]