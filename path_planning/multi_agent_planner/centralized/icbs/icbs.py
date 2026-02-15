"""
Conflict-based search for multi-agent path planning
author: Brandon Ho
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, Constraints, HighLevelNode, CBS,Conflict
import heapq
import time

class IEnvironment(Environment):
    def __init__(
        self,
        graph_map,
        agents,
        astar_max_iterations=-1,
        radius=0.0,
        velocity=0.0,
        use_constraint_sweep=True,
    ):
        super().__init__(
            graph_map,
            agents,
            astar_max_iterations,
            radius,
            velocity,
            use_constraint_sweep,
        )

    def compute_agent_solution(self, target_agent, curr_solution,curr_solution_cost):
        for agent in self.agent_dict.keys():
            if agent == target_agent:
                self.constraints = self.constraint_dict.setdefault(agent, Constraints())
                local_solution, local_cost = self.a_star.search(agent)
                if not local_solution:
                    return False
                curr_solution.update({agent: local_solution})
                curr_solution_cost[agent] = local_cost
        return curr_solution, curr_solution_cost


class ICBS(CBS):
    def __init__(
        self,
        environment: IEnvironment,
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
        super().__init__(environment, time_limit, max_iterations, verbose)
        self.env = environment

    def get_conflict_cost(self, P: HighLevelNode, conflict: Conflict):
        constraint_dict = self.env.create_constraints_from_conflict(conflict)
        costs = {}

        for agent in constraint_dict:
            # Selective copy: only copy the modified agent's constraints
            temp_constraints = {}
            for a in P.constraint_dict.keys():
                if a == agent:
                    new_constraints = Constraints()
                    new_constraints.vertex_constraints = P.constraint_dict[a].vertex_constraints.copy()
                    new_constraints.edge_constraints = P.constraint_dict[a].edge_constraints.copy()
                    new_constraints.add_constraint(constraint_dict[agent])
                    temp_constraints[a] = new_constraints
                else:
                    temp_constraints[a] = P.constraint_dict[a]

            self.env.constraint_dict = temp_constraints

            #To Do: Fix the constraint as it is causing icbs to fail
            self.env.constraints = self.env.constraint_dict.setdefault(agent, Constraints())
            path, cost = self.env.a_star.search(agent)
            if not path:
                costs[agent] = float("inf")
            else:
                costs[agent] = cost - P.solution_cost[agent]
        return costs

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

            _, _, P = heapq.heappop(self.open_list)

            if P is None:
                break
            state_key = self._get_state_key(P)
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)

            self.env.constraint_dict = P.constraint_dict

            # First improvement, we check all of the conflicts and grab the cardinal conflicts to solve first
            conflict_list = self.env.get_conflicts(P.solution,get_first_conflict=False)

            if not conflict_list:
                if self.verbose:
                    print("solution found")
                return self.generate_plan(P.solution)

            best_conflict = None
            best_conflict_score = float("inf")
            best_conflict_costs = None

            for conflict in conflict_list:
                costs = self.get_conflict_cost(P, conflict=conflict)
                c_vals = list(costs.values())

                if all(c > 0 for c in c_vals):
                    # Cardinal conflict - highest priority, stop searching
                    best_conflict = conflict
                    best_conflict_costs = costs
                    best_conflict_score = 0
                    break
                elif any(c > 0 for c in c_vals):
                    score = 1  # semi-cardinal, some solutions will cause cost increase but not all
                else:
                    score = 2  # non-cardinal, there is no cost to fix this solution

                score += min(c_vals)  # smaller min cost increase is better

                if score < best_conflict_score:
                    best_conflict_score = score
                    best_conflict = conflict
                    best_conflict_costs = costs

            constraint_dict = self.env.create_constraints_from_conflict(best_conflict)

            # Second improvement, grabs the "best" agents (cheapest ones to resolve) and only adds those to heap
            min_cost = min(best_conflict_costs.values())
            best_agents = [a for a, c in best_conflict_costs.items() if c == min_cost]

            for agent in best_agents:
                new_node = HighLevelNode()
                new_node.solution = P.solution.copy()
                new_node.solution_cost = P.solution_cost.copy()

                # Selective deep copy only for affected agent's constraints
                new_node.constraint_dict = {}
                for a in self.env.agent_dict.keys():
                    if a == agent:
                        # Deep copy only the modified agent's constraints
                        new_constraints = Constraints()
                        new_constraints.vertex_constraints = P.constraint_dict[
                            a
                        ].vertex_constraints.copy()
                        new_constraints.edge_constraints = P.constraint_dict[
                            a
                        ].edge_constraints.copy()
                        new_constraints.add_constraint(constraint_dict[agent])
                        new_node.constraint_dict[a] = new_constraints
                    else:
                        # Share unchanged constraints
                        new_node.constraint_dict[a] = P.constraint_dict[a]

                self.env.constraint_dict = new_node.constraint_dict

                new_node.solution,new_node.solution_cost = self.env.compute_agent_solution(
                    target_agent=agent, curr_solution=new_node.solution, curr_solution_cost=new_node.solution_cost
                )
                if not new_node.solution:
                    continue
                new_node.cost = sum(new_node.solution_cost.values())
                heapq.heappush(
                    self.open_list, (new_node.cost, next(self.counter), new_node)
                )
            iterations += 1
        return {}
