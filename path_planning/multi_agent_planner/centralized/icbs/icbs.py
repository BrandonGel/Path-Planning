"""
Conflict-based search for multi-agent path planning
author: Brandon Ho
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, Constraints, HighLevelNode, CBS,Conflict
import heapq
import time
from path_planning.multi_agent_planner.centralized.icbs.a_star import AStar

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
        self.a_star = AStar(self, astar_max_iterations,radius)

    def compute_agent_solution(self, target_agent, curr_solution,curr_solution_cost):
        self.constraints = self.constraint_dict.setdefault(target_agent, Constraints())
        local_solution, local_cost = self.a_star.search(target_agent)
        if not local_solution:
            return False
        curr_solution[target_agent] = local_solution
        curr_solution_cost[target_agent] = local_cost
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

    def _get_best_conflict(self, P, conflict_list):
        best_conflict = None
        best_score = float("inf")
        best_costs = None
        best_paths = None

        for conflict in conflict_list:
            costs, paths = self.get_conflict_cost(P, conflict)
            c_vals = list(costs.values())

            # Scoring: Cardinal (0), Semi-Cardinal (1), Non-Cardinal (2)
            if all(c > 0 for c in c_vals): score = 0
            elif any(c > 0 for c in c_vals): score = 1
            else: score = 2

            if score < best_score:
                best_score = score
                best_conflict = conflict
                best_costs = costs
                best_paths = paths
                if score == 0: break  # Found cardinal, stop

        return best_conflict, best_costs, best_paths, best_score

    def get_conflict_cost(self, P: HighLevelNode, conflict: Conflict):
        constraint_dict = self.env.create_constraints_from_conflict(conflict)
        costs = {}
        paths = {}
        
        # Save original constraint dict to restore after evaluation
        original_constraint_dict = self.env.constraint_dict

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
            self.env.constraints = self.env.constraint_dict.setdefault(agent, Constraints())
            path, cost = self.env.a_star.search(agent)
            if not path:
                costs[agent] = float("inf")
                paths[agent] = None
            else:
                costs[agent] = cost - P.solution_cost[agent]
                paths[agent] = path
        
        # Restore original constraint dict
        self.env.constraint_dict = original_constraint_dict
        return costs, paths

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

            # 1. Prioritize Conflicts (Cardinal, Semi, Non)
            best_conflict, best_costs, best_paths, score = self._get_best_conflict(P, conflict_list)

            # 2. BYPASS STRATEGY
            # If any agent can resolve the conflict with 0 cost increase, check for bypass
            bypass_found = False
            for agent, cost_inc in best_costs.items():
                if cost_inc == 0:
                    # Attempt bypass: get a same-cost path that avoids this conflict
                    temp_constraints = self._get_updated_constraints(P, agent, best_conflict)
                    self.env.constraint_dict = temp_constraints
                    new_path, _ = self.env.a_star.search(agent, solution=P.solution)
                    
                    if new_path:
                        temp_sol = P.solution.copy()
                        temp_sol[agent] = new_path
                        if len(self.env.get_conflicts(temp_sol, False)) < len(conflict_list):
                            P.solution = temp_sol
                            heapq.heappush(self.open_list, (P.cost, next(self.counter), P))
                            bypass_found = True
                            break
            
            if bypass_found:
                continue
            
            # Branching (if no bypass)
            self._branch(P, best_conflict, best_costs)
            iterations += 1
        return {}


    def _get_updated_constraints(self, P, agent, conflict):
        """Helper to create a new constraint dictionary for a specific branch/bypass."""
        new_constraints_dict = {}
        conflict_constraints = self.env.create_constraints_from_conflict(conflict)
        
        for a in self.env.agent_dict.keys():
            if a == agent:
                # Deep copy and add new constraint for the target agent
                nc = Constraints()
                nc.vertex_constraints = P.constraint_dict[a].vertex_constraints.copy()
                nc.edge_constraints = P.constraint_dict[a].edge_constraints.copy()
                nc.add_constraint(conflict_constraints[agent])
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
            new_node.constraint_dict = self._get_updated_constraints(P, agent, conflict)
            
            self.env.constraint_dict = new_node.constraint_dict
            # Re-plan only the affected agent
            res = self.env.compute_agent_solution(agent, new_node.solution, new_node.solution_cost)
            
            if res:
                new_node.solution, new_node.solution_cost = res
                new_node.cost = sum(new_node.solution_cost.values())
                heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))