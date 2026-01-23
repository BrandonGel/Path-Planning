"""
Conflict-based search for multi-agent path planning
author: Brandon Ho 
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from path_planning.common.environment.node import Node
from path_planning.multi_agent_planner.centralized.cbs.a_star import AStar
from python_motion_planning.path_planner.graph_search.a_star import AStar as AStar2
from math import fabs
from itertools import combinations
from copy import deepcopy


class State(object):
    def __init__(self, time, node: Node):
        self.time = time
        self.node = node
    def __eq__(self, other):
        return self.time == other.time and self.node == other.node
    def __hash__(self):
        return hash(str(self.time)+str(self.node))
    def is_equal_except_time(self, state):
        return self.node == state.node
    def __str__(self):
        return str((self.time, self.node))

class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.node_1 = Node(None,None,0,0)
        self.node_2 = Node(None,None,0,0)

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.node_1) + ', ' + str(self.node_2) + ')'

class VertexConstraint(object):
    def __init__(self, time, node):
        self.time = time
        self.node = node

    def __eq__(self, other):
        return self.time == other.time and self.node == other.node
    def __hash__(self):
        return hash(str(self.time)+str(self.node))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.node) + ')'

class EdgeConstraint(object):
    def __init__(self, time, node_1, node_2):
        self.time = time
        self.node_1 = node_1
        self.node_2 = node_2
    def __eq__(self, other):
        return self.time == other.time and self.node_1 == other.node_1 \
            and self.node_2 == other.node_2
    def __hash__(self):
        return hash(str(self.time) + str(self.node_1) + str(self.node_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.node_1) +', '+ str(self.node_2) + ')'

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
        self.a_star2 = AStar2(graph_map,None,None)

    def get_neighbors(self, state):
        neighbors = []
        nodes = self.graph_map.get_neighbors(state.node)
        
        # Wait action
        n = State(state.time + 1, state.node)
        if self.state_valid(n):
            neighbors.append(n)

        # Move action
        for node in nodes:
            n = State(state.time + 1, node)
            if self.state_valid(n):
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
                    result.node_1 = state_1.node
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
                    result.node_1 = state_1a.node
                    result.node_2 = state_1b.node
                    return result
        return False

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.node_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.node_1, conflict.node_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.node_2, conflict.node_1)

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

        if self.graph_map.in_collision(state.node.current):
            return False
        return  VertexConstraint(state.time, state.node) not in self.constraints.vertex_constraints 

    def transition_valid(self, state_1, state_2):
        return EdgeConstraint(state_1.time, state_1.node, state_2.node) not in self.constraints.edge_constraints

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        goal = self.agent_dict[agent_name]["goal"]
        return sum([fabs(state.node.current[i] - goal.node.current[i]) for i in range(len(state.node.current))])


    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for name in self.agents:
            agent = self.agents[name]
            start_node = Node(agent['start'],None,0,0)
            goal_node = Node(agent['goal'],None,0,0)
            
            if start_node in self.graph_map.node_index_list:
                start_index= self.graph_map.node_index_list[start_node]
                start_node = self.graph_map.nodes[start_index]
            if goal_node in self.graph_map.node_index_list:
                goal_index = self.graph_map.node_index_list[goal_node]
                goal_node = self.graph_map.nodes[goal_index]

            start_state = State(0, start_node)
            goal_state = State(0, goal_node)
            
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
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost

class CBS(object):
    def __init__(self, environment):
        self.env = environment
        self.open_set = set()
        self.closed_set = set()
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

        self.open_set |= {start}

        while self.open_set:
            P = min(self.open_set)
            self.open_set -= {P}
            self.closed_set |= {P}

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
                if new_node not in self.closed_set:
                    self.open_set |= {new_node}

        return {}

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = []
            for state in path:
                if len(state.node.current) == 2:
                    path_dict_list.append({'t':state.time, 'x':state.node.current[0], 'y':state.node.current[1]})
                else:
                    path_dict_list.append({'t':state.time, 'x':state.node.current[0], 'y':state.node.current[1], 'z':state.node.current[2]})
            plan[agent] = path_dict_list
        return plan
