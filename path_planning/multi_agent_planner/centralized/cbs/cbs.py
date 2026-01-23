"""
Conflict-based search for multi-agent path planning
author: Brandon Ho 
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from path_planning.common.environment.node import Node
from path_planning.multi_agent_planner.centralized.cbs.a_star import AStar
import time
from math import fabs
from itertools import combinations
from copy import deepcopy

class Location(object):
    def __init__(self, point:tuple = None):
        self.point = tuple(point) if point is not None else None
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
    def __init__(self, graph_map, agents):
        self.graph_map = graph_map
        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self)

    def get_neighbors(self, state):
        neighbors = []
        node = Node(tuple(state.location.point))
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


    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False

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
        return VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints

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
        for name in self.agents:
            agent = self.agents[name]
            start_state = State(0, Location(agent['start']))
            goal_state = State(0, Location(agent['goal']))
            
            self.agent_dict.update({name:{'start':start_state, 'goal':goal_state}})

    def compute_solution(self):
        solution = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            local_solution = self.a_star.search(agent)
            if not local_solution:
                return False
            solution.update({agent:local_solution})
        return solution

    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
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
    def __init__(self, environment, time_limit: float | None = None, max_iterations: int | None = None):
        """
        :param environment: Environment instance
        :param time_limit: Optional wall-clock time limit (in seconds) for the
                           high-level CBS search. If None, no time limit is
                           enforced. If exceeded, the search terminates early
                           and returns an empty solution.
        """
        self.env = environment
        # Use a priority queue (heap) for deterministic expansion order
        # instead of an unordered set whose iteration order can vary
        # between runs and Python versions.
        #
        # Each heap entry is a tuple: (cost, HighLevelNode)
        self.open_heap = []
        self.open_set = set()   # for fast membership checks
        self.closed_set = set()
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.iterations = 0

    def _push_open(self, node):
        """Push a node into the open list (heap) in a deterministic way."""
        import heapq
        heapq.heappush(self.open_heap, (node.cost, node))
        self.open_set.add(node)

    def _pop_open(self):
        """Pop the next node from the open list, skipping any stale entries."""
        import heapq
        while self.open_heap:
            _, node = heapq.heappop(self.open_heap)
            if node in self.open_set:
                self.open_set.remove(node)
                return node
        return None
    def search(self):
        start = HighLevelNode()
        # TODO: Initialize it in a better way
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()
        start.solution = self.env.compute_solution()
        if not start.solution:
            return {}
        start.cost = self.env.compute_solution_cost(start.solution)

        # Initialize open list with the start node
        self._push_open(start)

         
        st = time.time()
        while self.open_heap:
            P = self._pop_open()
            if P is None:
                break
            self.closed_set |= {P}

            # Enforce optional wall-clock time limit on the high-level search
            if self.time_limit is not None and (time.time() - st) > self.time_limit:
                print(f"Search terminated: time limit of {self.time_limit} seconds exceeded.")
                return {}

            if self.max_iterations is not None and self.iterations >= self.max_iterations:
                print(f"Search terminated: max iterations of {self.max_iterations} reached.")
                return {}

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_first_conflict(P.solution)
            if not conflict_dict:
                print("solution found")
                return self.generate_plan(P.solution)
            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)
            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)
                # TODO: ending condition
                if new_node not in self.closed_set and new_node not in self.open_set:
                    self._push_open(new_node)
            self.iterations += 1
        return {}

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