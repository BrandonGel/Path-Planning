"""
Improved Conflict-based search for multi-agent path planning
author: Brandon Ho
original author: Ashwin Bose (@atb033)
description: This file implements the Improved Conflict-based search algorithm
(cardinal-conflict prioritization + conflict bypassing) on top of the
continuous CCBS high-level search.
"""

import heapq
import time
from path_planning.multi_agent_planner.centralized.ccbs.graph_generation import Conflict, Constraints, Environment
from path_planning.multi_agent_planner.centralized.sipp.graph_generation import SippNode
from itertools import count, combinations
from copy import deepcopy
from path_planning.multi_agent_planner.centralized.ccbs.ccbs import CCBS
from path_planning.multi_agent_planner.centralized.ccbs.ccbs import HighLevelNode

class ICCBS(CCBS):
    def __init__(
        self,
        environment: Environment,
        time_limit: float | None = None,
        max_iterations: int | None = None,
        verbose: bool = False,
        find_num_conflicts: bool = True,
    ):
        """
        :param environment: IEnvironment instance
        :param time_limit: Optional wall-clock time limit (in seconds) for the
                           high-level search. If None, no time limit is
                           enforced.
        :param max_iterations: Optional cap on high-level expansions.
        :param find_num_conflicts: Enable conflict-aware low-level tie-breaking.

        ICCBS always collects all conflicts (``get_first_conflict=False``) so it
        can classify them as cardinal / semi-cardinal / non-cardinal.
        """
        super().__init__(
            environment,
            time_limit=time_limit,
            max_iterations=max_iterations,
            verbose=verbose,
            get_first_conflict=False,
            find_num_conflicts=find_num_conflicts,
        )

    def _get_best_conflict(self, P, conflict_list):
        best_conflict = None
        best_score = float("inf")
        best_costs = None
        best_action_costs = None
        best_paths = None

        for conflict in conflict_list:
            costs, action_costs, paths = self.get_conflict_cost(P, conflict)
            c_vals = list(costs.values())

            # Scoring: Cardinal (0), Semi-Cardinal (1), Non-Cardinal (2)
            if all(c > 0 for c in c_vals): score = 0
            elif any(c > 0 for c in c_vals): score = 1
            else: score = 2

            if score < best_score:
                best_score = score
                best_conflict = conflict
                best_costs = costs
                best_action_costs = action_costs
                best_paths = paths
                if score == 0: break  # Found cardinal, stop

        return best_conflict, best_costs, best_action_costs, best_paths, best_score

    def get_conflict_cost(self, P: HighLevelNode, conflict: Conflict):
        constraint_dict = self.env.create_constraints_from_conflict(conflict)
        costs = {}
        action_costs = {}
        paths = {}

        # Save original constraint dict to restore after evaluation
        original_constraint_dict = self.env.constraint_dict

        for agent in constraint_dict:
            # Selective copy: only copy the modified agent's constraints
            temp_constraints = {}
            for a in P.constraint_dict.keys():
                if a == agent:
                    new_constraints = Constraints()
                    new_constraints.wait_constraints = P.constraint_dict[a].wait_constraints.copy()
                    new_constraints.move_constraints = P.constraint_dict[a].move_constraints.copy()
                    new_constraints.add_constraint(constraint_dict[agent])
                    temp_constraints[a] = new_constraints
                else:
                    temp_constraints[a] = P.constraint_dict[a]

            self.env.constraint_dict = temp_constraints
            self.env.constraints = self.env.constraint_dict.setdefault(agent, Constraints())
            path,action_cost,cost = self.env.compute_solution(
                    affected_agent=agent,
                    base_solution=P.solution,
                    base_action_cost=P.solution_action_cost,
                    base_cost=P.solution_cost,
                    find_num_conflicts=self.find_num_conflicts,
                )

            if not path:
                costs[agent] = float("inf")
                action_costs[agent] = None
                paths[agent] = None
            else:
                costs[agent] = cost[agent] - P.solution_cost[agent]
                action_costs[agent] = action_cost[agent]
                paths[agent] = path[agent]

        # Restore original constraint dict
        self.env.constraint_dict = original_constraint_dict
        return costs, action_costs, paths

    def search(self):
        st = time.time()
        iterations = 1
        success = False
        self._reset_stats()
        solution = {}
        solution_info = {}

        start = HighLevelNode()
        start.constraint_dict = {}
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

            _, P_counter, P = heapq.heappop(self.open_list)
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

            best_conflict, best_costs, best_action_costs, best_paths, score = self._get_best_conflict(P, conflict_list)
            if best_conflict is None:
                continue
            self._record_conflict_class(score)

            bypass_found = False
            for agent, cost_inc in best_costs.items():
                if cost_inc == 0:
                    # Attempt bypass: get a same-cost path that avoids this conflict
                    temp_constraints = self._get_updated_constraints(P, agent, best_conflict)
                    self.env.constraint_dict = temp_constraints
                    self.env.constraints = self.env.constraint_dict.setdefault(agent, Constraints())
                    new_path, new_action_cost, _ = self.env.sipp.search(agent,solution=P.solution,solution_action_cost=P.solution_action_cost)

                    if new_path:
                        new_node = HighLevelNode()
                        new_node.solution = P.solution.copy()
                        new_node.solution_cost = P.solution_cost.copy()
                        new_node.solution_action_cost = P.solution_action_cost.copy()
                        new_node.solution[agent] = new_path
                        new_node.solution_action_cost[agent] = new_action_cost
                        new_node.cost = P.cost
                        new_node.constraint_dict = temp_constraints

                        if len(self.env.get_conflicts(new_node.solution, new_node.solution_action_cost,self.get_first_conflict)) < len(conflict_list):
                            heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))
                            bypass_found = True
                            break
            if bypass_found:
                continue

            self._branch(P, best_conflict, best_costs)

        return self._finalize(solution, solution_info, st, iterations, success)

    def _get_updated_constraints(self, P, agent, conflict):
        """Helper to create a new constraint dictionary for a specific branch/bypass."""
        new_constraints_dict = {}
        conflict_constraints = self.env.create_constraints_from_conflict(conflict)

        for a in self.env.agent_dict.keys():
            if a == agent:
                # Deep copy and add new constraint for the target agent
                nc = Constraints()
                nc.wait_constraints = P.constraint_dict[a].wait_constraints.copy()
                nc.move_constraints = P.constraint_dict[a].move_constraints.copy()
                nc.add_constraint(conflict_constraints[a])
                new_constraints_dict[a] = nc
            else:
                new_constraints_dict[a] = P.constraint_dict[a]
        return new_constraints_dict

    def _branch(self, P, conflict, costs):
        """Expands the high-level tree by creating child nodes."""
        for agent in costs.keys():
            if costs[agent] == float('inf'): continue # Prune if no path exists

            new_node = HighLevelNode()
            new_node.solution = P.solution.copy()
            new_node.solution_cost = P.solution_cost.copy()
            new_node.solution_action_cost = P.solution_action_cost.copy()
            new_node.constraint_dict = self._get_updated_constraints(P, agent, conflict)

            self.env.constraint_dict = new_node.constraint_dict

            # Re-plan only the affected agent
            res = self.env.compute_solution(
                    affected_agent=agent,
                    base_solution=P.solution,
                    base_action_cost=P.solution_action_cost,
                    base_cost=P.solution_cost,
                    find_num_conflicts=self.find_num_conflicts,
                )

            if res and res[0]:
                new_node.solution, new_node.solution_action_cost, new_node.solution_cost = res
                new_node.cost = sum(new_node.solution_cost.values())
                heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))
