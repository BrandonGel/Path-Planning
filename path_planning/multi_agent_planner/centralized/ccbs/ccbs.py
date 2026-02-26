"""
Conflict-based search for multi-agent path planning
author: Brandon Ho
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

import heapq
import time
from path_planning.multi_agent_planner.centralized.ccbs.graph_generation import Conflict, Constraints, Environment
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippNode
from itertools import count, combinations
from copy import deepcopy

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.solution_action_cost = {}
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

class CCBS(object):
    def __init__(
        self,
        environment: Environment,
        time_limit: float | None = None,
        max_iterations: int | None = None,
        verbose: bool = False,
    ):
        """
        :param environment: IEnvironment instance
        :param time_limit: Optional wall-clock time limit (in seconds) for the
                           high-level CBS search. If None, no time limit is
                           enforced. If exceeded, the search terminates early
                           and returns an empty solution.
        """
        self.env = environment
        self.counter = count()
        self.open_list = []
        self.closed_set = set()
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _get_state_key(self, node):
        """Generate a hashable state key for closed set checking."""
        # Create a frozen representation of the solution
        solution_tuple = tuple(
            (agent, tuple((s.time, s.position, s.interval) for s in path))
            for agent, path in sorted(node.solution.items())
        )
        return solution_tuple

    def search(self):
        start = HighLevelNode()
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = deepcopy(Constraints())

        start.solution, start.solution_action_cost, start.solution_cost = self.env.compute_solution()
        if not start.solution:
            if self.verbose:
                print("No initial solution found")
            return {}

        start.cost = sum(start.solution_cost.values())

        # Add start node to heap
        heapq.heappush(self.open_list, (start.cost, next(self.counter), start))

        st = time.time()
        iterations = 0
        while self.open_list:
            if self.time_limit is not None and (time.time() - st) > self.time_limit:
                if self.verbose:
                    print(
                        f"Search terminated: time limit of {self.time_limit} seconds exceeded."
                    )
                return {}

            if self.max_iterations is not None and iterations >= self.max_iterations:
                if self.verbose:
                    print(
                        f"Search terminated: max iterations of {self.max_iterations} reached."
                    )
                return {}

            _, P_counter, P = heapq.heappop(self.open_list)
            state_key = self._get_state_key(P)
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)

            self.env.constraint_dict = P.constraint_dict

            conflict_list = self.env.get_conflicts(P.solution, P.solution_action_cost)
            if not conflict_list:
                if self.verbose:
                    print("solution found")
                return self.generate_plan(P.solution, P.solution_action_cost)

            constraint_dict = self.env.create_constraints_from_conflict(conflict_list[0])
            for agent in constraint_dict.keys():
                new_node = HighLevelNode()
                new_node.solution = P.solution.copy()
                new_node.solution_cost = P.solution_cost.copy()
                new_node.solution_action_cost = P.solution_action_cost.copy()
                # Selective deep copy only for affected agent's constraints
                new_node.constraint_dict = {}
                for i, a in enumerate(self.env.agent_dict.keys()):
                    if a == agent:
                        # Deep copy only the modified agent's constraints (copy SippNode values)
                        new_constraints = Constraints()
                        new_constraints.wait_constraints = {
                            k: v.copy() for k, v in P.constraint_dict[a].wait_constraints.items()
                        }
                        new_constraints.move_constraints = {
                            k: v.copy() for k, v in P.constraint_dict[a].move_constraints.items()
                        }
                        new_constraints.add_constraint(constraint_dict[agent])
                        new_node.constraint_dict[a] = new_constraints
                    else:
                        # Share unchanged constraints
                        new_node.constraint_dict[a] = P.constraint_dict[a]

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution, new_node.solution_action_cost, new_node.solution_cost = self.env.compute_solution(
                    affected_agent=agent,
                    base_solution=P.solution,
                    base_action_cost=P.solution_action_cost,
                    base_cost=P.solution_cost,
                )
                if not new_node.solution:
                    continue
                new_node.cost = sum(new_node.solution_cost.values())
                heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))

            iterations += 1
            # if iterations > 30:
            #     break
        return {}

    def generate_plan(self, solution, solution_action_cost):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = []
            for state, action in zip(path, solution_action_cost[agent]):
                wait_time, move_time = action
                if len(state.position) == 2:
                    path_dict_list.append({'t':state.time+wait_time, 'x':state.position[0], 'y':state.position[1]})
                    if wait_time > 0:
                        path_dict_list.append({'t':state.time+wait_time, 'x':state.position[0], 'y':state.position[1]})
                elif len(state.position) == 3:
                    path_dict_list.append({'t':state.time+wait_time, 'x':state.position[0], 'y':state.position[1], 'z':state.position[2]})
                    if wait_time > 0:
                        path_dict_list.append({'t':state.time+wait_time, 'x':state.position[0], 'y':state.position[1], 'z':state.position[2]})
                else:
                    raise ValueError(f"Invalid position dimension: {len(state.position)}")
            plan[agent] = path_dict_list
        return plan