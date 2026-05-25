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
        get_first_conflict: bool = True,
        find_num_conflicts: bool = True,
    ):
        """
        :param environment: IEnvironment instance
        :param time_limit: Optional wall-clock time limit (in seconds) for the
                           high-level CBS search. If None, no time limit is
                           enforced. If exceeded, the search terminates early
                           and returns an empty solution.
        :param get_first_conflict: If True, conflict detection returns as soon
                           as the first conflict is found (standard CBS). If
                           False, all conflicts are collected (used by ICCBS
                           for cardinal-conflict prioritization).
        :param find_num_conflicts: If True, the low-level SIPP search performs
                           conflict-aware tie-breaking against the other
                           agents' current paths, which tends to reduce the
                           number of high-level expansions.
        """
        self.env = environment
        self.counter = count()
        self.open_list = []
        self.closed_set = set()
        self.time_limit = time_limit if time_limit is not None and time_limit > 0 else float('inf')
        self.max_iterations = max_iterations if max_iterations is not None and max_iterations > 0 else float('inf')
        self.verbose = verbose
        self.get_first_conflict = get_first_conflict
        self.find_num_conflicts = find_num_conflicts
        self.total_time = 0
        self.total_iterations = 0
        self._reset_stats()

    def _reset_stats(self):
        """Reset the per-search conflict/expansion statistics."""
        self.total_conflicts = 0
        self.nodes_expanded = 0
        self.cardinal_count = 0
        self.semi_cardinal_count = 0
        self.non_cardinal_count = 0

    def _conflict_stats(self):
        """Return the conflict/expansion statistics as a dict for solution_info."""
        return {
            "total_conflicts": self.total_conflicts,
            "high_level_nodes_expanded": self.nodes_expanded,
            "cardinal_conflicts": self.cardinal_count,
            "semi_cardinal_conflicts": self.semi_cardinal_count,
            "non_cardinal_conflicts": self.non_cardinal_count,
        }

    def _record_conflict_class(self, score):
        """Record a conflict's cardinality (0=cardinal, 1=semi, 2=non-cardinal)."""
        if score == 0:
            self.cardinal_count += 1
        elif score == 1:
            self.semi_cardinal_count += 1
        elif score == 2:
            self.non_cardinal_count += 1

    def _finalize(self, solution, solution_info, st, iterations, success):
        """Populate solution_info and timing/iteration counters, then return."""
        self.total_time = min(self.time_limit, time.time() - st)
        self.total_iterations = min(self.max_iterations, iterations)
        solution_info["runtime"] = self.total_time
        solution_info["total_iterations"] = self.total_iterations
        solution_info["success"] = success
        solution_info.update(self._conflict_stats())
        return solution, solution_info

    def validate_solution(self, plan=None):
        """
        Validate a CCBS plan (the dict returned by ``generate_plan``/``search``)
        for time, velocity and collision anomalies.

        Returns a dict with keys ``no_time_anomaly``, ``no_velocity_anomaly``,
        ``collisions`` and ``conflict_free``.
        """
        from path_planning.utils.checker import check_solution_full
        if not plan:
            return {
                "conflict_free": False,
                "no_time_anomaly": False,
                "no_velocity_anomaly": False,
                "collisions": {},
            }
        result = check_solution_full(
            plan,
            self.env.radius,
            is_using_constant_speed=False,
            verbose=self.verbose,
        )
        result["conflict_free"] = (len(result["collisions"]) == 0)
        return result

    def _get_state_key(self, node):
        """Generate a hashable state key for closed set checking."""
        # Create a frozen representation of the solution
        solution_tuple = tuple(
            (agent, tuple((s.time, s.position, s.interval) for s in path))
            for agent, path in sorted(node.solution.items())
        )
        return solution_tuple

    def search(self):
        st = time.time()
        iterations = 1
        success = False
        self._reset_stats()
        start = HighLevelNode()
        start.constraint_dict = {}
        solution = {}
        solution_info = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = deepcopy(Constraints())

        start.solution, start.solution_action_cost, start.solution_cost = self.env.compute_solution(
            find_num_conflicts=self.find_num_conflicts,
        )
        if not start.solution:
            if self.verbose:
                print("No initial solution found")
            return self._finalize({}, solution_info, st, iterations, success)

        start.cost = sum(start.solution_cost.values())

        # Add start node to heap
        heapq.heappush(self.open_list, (start.cost, next(self.counter), start))
        while self.open_list:
            iterations += 1
            if (time.time() - st) > self.time_limit:
                if self.verbose:
                    print(
                        f"Search terminated: time limit of {self.time_limit} seconds exceeded."
                    )
                break

            if iterations >= self.max_iterations:
                if self.verbose:
                    print(
                        f"Search terminated: max iterations of {self.max_iterations} reached."
                    )
                break

            _, _, P = heapq.heappop(self.open_list)
            state_key = self._get_state_key(P)
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)
            self.nodes_expanded += 1

            self.env.constraint_dict = P.constraint_dict

            conflict_list = self.env.get_conflicts(P.solution, P.solution_action_cost, self.get_first_conflict)
            if not conflict_list:
                if self.verbose:
                    print("solution found")
                success = True
                solution = self.generate_plan(P.solution, P.solution_action_cost)
                break
            self.total_conflicts += len(conflict_list)

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
                    find_num_conflicts=self.find_num_conflicts,
                )
                if not new_node.solution:
                    continue
                new_node.cost = sum(new_node.solution_cost.values())
                heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))

        return self._finalize(solution, solution_info, st, iterations, success)

    def generate_plan(self, solution, solution_action_cost):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = []
            for i, (state, action) in enumerate(zip(path, solution_action_cost[agent])):
                wait_time, move_time = action
                if len(state.position) == 2:
                    path_dict_list.append({'t':state.time, 'x':state.position[0], 'y':state.position[1]})
                    if wait_time > 0:
                        path_dict_list.append({'t':state.time+wait_time, 'x':state.position[0], 'y':state.position[1]})
                elif len(state.position) == 3:
                    path_dict_list.append({'t':state.time, 'x':state.position[0], 'y':state.position[1], 'z':state.position[2]})
                    if wait_time > 0:
                        path_dict_list.append({'t':state.time+wait_time, 'x':state.position[0], 'y':state.position[1], 'z':state.position[2]})
                else:
                    raise ValueError(f"Invalid position dimension: {len(state.position)}")
            plan[agent] = path_dict_list
        return plan