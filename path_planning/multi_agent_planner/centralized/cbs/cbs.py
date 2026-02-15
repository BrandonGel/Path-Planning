"""
Conflict-based search for multi-agent path planning
author: Brandon Ho 
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from typing import Any
from path_planning.common.environment.node import Node
from path_planning.multi_agent_planner.centralized.cbs.a_star import AStar
import heapq
import time
from math import fabs
from itertools import count,combinations
import numpy as np

class Location(object):
    def __init__(self, point:tuple = None):
        self.point = tuple(point) if point is not None else None
    def __hash__(self):
        return hash(self.point)
    def __eq__(self, other):
        return self.point == other.point
    def __str__(self):
        return str(self.point)
    def __len__(self):
        return len(self.point)
    def __getitem__(self, index):
        return self.point[index]
    def __add__(self, other):
        return Location(tuple(self.point + other.point))

class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location))

class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class Environment(object):
    def __init__(self, graph_map, agents, astar_max_iterations=-1, radius = 1.0, velocity = 0.0, use_constraint_sweep=True):
        self.graph_map = graph_map
        if radius > 0:
            self.graph_map.set_constraint_sweep()
        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self, astar_max_iterations)
        self.radius = radius
        self.velocity = velocity
        self.use_constraint_sweep = use_constraint_sweep
        self._constraint_sweep_cache = {}  # (p1, p2, r) -> (nodes, edges, start_nodes)
        self._constraint_segment_cache = {}  # (p1a, p1b, p2a, p2b, v1, v2, r1, r2) -> bool

    def get_neighbors(self, state):
        neighbors = []
        node = Node(tuple[Any, ...](state.location.point))
        nodes = self.graph_map.get_neighbors(node)
        # Wait action
        n = State(state.time + 1,  state.location)
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)

        # Move action
        for node in nodes:
            n = State(state.time + 1, Location(tuple(node.current)))
            if self.state_valid(n) and self.transition_valid(state, n):
                neighbors.append(n)
        return neighbors


    def get_conflicts(self, solution,get_first_conflict: bool = True):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        conflicts = []
        for t in range(max_t):
            if self.radius == 0:
                for agent_1, agent_2 in combinations(solution.keys(), 2):
                    state_1 = self.get_state(agent_1, solution, t)
                    state_2 = self.get_state(agent_2, solution, t)

                    # Vertex conflict check (default)
                    if state_1.is_equal_except_time(state_2):
                        result.time = t
                        result.type = Conflict.VERTEX
                        result.location_1 = state_1.location
                        result.agent_1 = agent_1
                        result.agent_2 = agent_2
                        if get_first_conflict:
                            return result
                        conflicts.append(result)

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if self.radius == 0:
                    # Edge conflict check (default)
                    if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                        result.time = t
                        result.type = Conflict.EDGE
                        result.agent_1 = agent_1
                        result.agent_2 = agent_2
                        result.location_1 = state_1a.location
                        result.location_2 = state_1b.location
                        if get_first_conflict:
                            return result
                        conflicts.append(result)
                    continue

                # Edge conflict check (use constraint sweep)
                if self.use_constraint_sweep :
                    state1_edges_locations = self._get_constraint_sweep_cached(state_1a.location.point,state_1b.location.point,self.velocity,2*self.radius)
                    state2_edges_locations = self._get_constraint_sweep_cached(state_2a.location.point,state_2b.location.point,self.velocity,2*self.radius)
                    edge_1 = (state_1a.location.point,state_1b.location.point)
                    edge_2 = (state_2a.location.point,state_2b.location.point)
                    edge_conflict = edge_1 in state2_edges_locations or edge_2 in state1_edges_locations
                else:
                    # Edge conflict check (use distances based on radius and vertices)                    
                    edge_conflict = self._get_constraint_segment_cached(
                        state_1a.location.point,state_1b.location.point,state_2a.location.point,state_2b.location.point,self.velocity,self.radius
                    )
                if edge_conflict:
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    conflicts.append(result)
                    if get_first_conflict:
                        return result
        return conflicts

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        if self.graph_map.in_collision_point(state.location.point):
            return False
        
        return  VertexConstraint(state.time,state.location) not in self.constraints.vertex_constraints

    def transition_valid(self, state_1, state_2):
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        goal = self.agent_dict[agent_name]["goal"]
        return sum([fabs(state.location[i] - goal.location[i]) for i in range(len(state.location))])

    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent['start']))
            goal_state = State(0, Location(agent['goal']))
            
            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    def _get_constraint_sweep_cached(self, p1, p2,v, r):
        """Cached wrapper for get_constraint_sweep to avoid duplicate queries."""
        key = (p1, p2, v, r)
        if key not in self._constraint_sweep_cache:
            self._constraint_sweep_cache[key] = self.graph_map.get_constraint_sweep(p1, p2,v, r)
        return self._constraint_sweep_cache[key]

    def _get_constraint_segment_cached(self, p1a, p1b, p2a, p2b, v, r):
        """Cached wrapper for get_constraint_sweep to avoid duplicate queries."""
        key = (p1a, p1b, p2a, p2b, v, r)
        if key not in self._constraint_segment_cache:
            self._constraint_segment_cache[key] = self.graph_map.get_constraint_segment(p1a, p1b, p2a, p2b, v,r)
        return self._constraint_segment_cache[key]

    def compute_solution(self):
        # Clear caches for fresh solution attempt
        solution = {}
        solution_cost = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            local_solution, local_cost = self.a_star.search(agent)
            if not local_solution:
                return False
            solution.update({agent:local_solution})
            solution_cost[agent] = local_cost
        return solution, solution_cost

    def compute_solution_cost(self, solution):
        cost = 0
        for agent, path in solution.items():
            path_array = np.array([list(p.values())[1:] for p in path])
            cost += np.linalg.norm(path_array[1:] - path_array[:-1], axis=1).sum()
        return cost

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.solution_cost = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash(self.cost)

    def __lt__(self, other):
        return self.cost < other.cost

class CBS(object):
    def __init__(self, environment: Environment, time_limit: float | None = None, max_iterations: int | None = None,verbose: bool = False):
        """
        :param environment: Environment instance
        :param time_limit: Optional wall-clock time limit (in seconds) for the
                           high-level CBS search. If None, no time limit is
                           enforced. If exceeded, the search terminates early
                           and returns an empty solution.
        """
        self.env = environment
        self.verbose = verbose
        self.open_list =  []  # for fast membership checks
        self.closed_set = set()
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.counter = count()

    def search(self):
        start = HighLevelNode()
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()

        start.solution, start.solution_cost = self.env.compute_solution()
        if not start.solution:
            if self.verbose:
                print("No initial solution found")
            return {}

        start.cost = sum(start.solution_cost.values())

        # Add start node to heap
        heapq.heappush(self.open_list, (start.cost, next(self.counter), start))

         
        st = time.time()
        iterations = 0
        while self.open_list :
            if self.time_limit is not None and (time.time() - st) > self.time_limit:
                if self.verbose:
                    print(f"Search terminated: time limit of {self.time_limit} seconds exceeded.")
                return {}

            if self.max_iterations is not None and iterations >= self.max_iterations:
                if self.verbose:
                    print(f"Search terminated: max iterations of {self.max_iterations} reached.")
                return {}

            _, _, P = heapq.heappop(self.open_list)

            
            if P is None:
                break
            state_key = self._get_state_key(P)
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_conflicts(P.solution)
            if not conflict_dict:
                if self.verbose:
                    print("solution found")
                return self.generate_plan(P.solution)
            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)
            for agent in constraint_dict.keys():
                new_node = HighLevelNode()
                new_node.solution = P.solution.copy()

                # Selective deep copy only for affected agent's constraints
                new_node.constraint_dict = {}
                for a in self.env.agent_dict.keys():
                    if a == agent:
                        # Deep copy only the modified agent's constraints
                        new_constraints = Constraints()
                        new_constraints.vertex_constraints = P.constraint_dict[a].vertex_constraints.copy()
                        new_constraints.edge_constraints = P.constraint_dict[a].edge_constraints.copy()
                        new_constraints.add_constraint(constraint_dict[agent])
                        new_node.constraint_dict[a] = new_constraints
                    else:
                        # Share unchanged constraints
                        new_node.constraint_dict[a] = P.constraint_dict[a]

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution, new_node.solution_cost = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = sum(new_node.solution_cost.values())
                heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))
            iterations += 1
        return {}


    def _get_state_key(self, node):
        """Generate a hashable state key for closed set checking."""
        # Create a frozen representation of the solution
        solution_tuple = tuple(
            (agent, tuple((s.time, s.location) for s in path))
            for agent, path in sorted(node.solution.items())
        )
        return solution_tuple
    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = []
            for state in path:
                if len(state.location) == 2:
                    path_dict_list.append({'t':state.time, 'x':state.location[0], 'y':state.location[1]})
                elif len(state.location) == 3:
                    path_dict_list.append({'t':state.time, 'x':state.location[0], 'y':state.location[1], 'z':state.location[2]})
                else:
                    raise ValueError(f"Invalid location dimension: {len(state.location)}")
            plan[agent] = path_dict_list
        return plan