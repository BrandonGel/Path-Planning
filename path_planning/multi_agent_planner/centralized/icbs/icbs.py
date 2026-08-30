"""
Conflict-based search for multi-agent path planning
author: Brandon Ho
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, Constraints, HighLevelNode, CBS, Conflict, State, Location
import heapq
import time
from path_planning.multi_agent_planner.centralized.icbs.a_star import AStar
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE

class IEnvironment(Environment):
    def __init__(
        self,
        graph_map,
        agents,
        astar_max_iterations=10000,
        radius=0.0,
        velocity=0.0,
        use_constraint_sweep=True,
        heuristic_type: str = 'dijkstra',
    ):
        super().__init__(
            graph_map,
            agents,
            astar_max_iterations,
            radius,
            velocity,
            use_constraint_sweep,
            heuristic_type,
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
        max_scored_conflicts: int | None = 8,
        conflict_classifier: str = "mdd",
        conflict_order: str = "time",
        icbs_mode: str = "",
        focal_w: float = 1.5,
    ):
        """
        :param environment: IEnvironment instance
        :param time_limit: Optional wall-clock time limit (in seconds) for the
                           high-level CBS search. If None, no time limit is
                           enforced. If exceeded, the search terminates early
                           and returns an empty solution.
        :param max_scored_conflicts: Cap on how many conflicts
                           _get_best_conflict classifies per CT node. Scoring a
                           conflict runs a full low-level A* per involved
                           agent, which dominates runtime on conflict-heavy
                           (radius > 0) instances. Branching on any conflict
                           keeps CBS complete and optimal, so capping only
                           risks missing a cardinal conflict a full scan would
                           have found. None scans the whole list.
        """
        super().__init__(environment, time_limit, max_iterations, verbose)
        self.env = environment
        self.max_scored_conflicts = max_scored_conflicts
        # 'mdd' classifies conflicts from per-agent optimal-plateau tables
        # (one build per agent per CT node, covering every conflict) and
        # falls back to the A*-based scorer when a table exceeds its state
        # cap; 'astar' always uses the A*-based scorer.
        self.conflict_classifier = conflict_classifier
        # 'time' keeps get_conflicts' earliest-first order; 'fan' sorts by
        # descending constraint-fan size. Measured on 2d/2d_8agents: 'time'
        # is uniformly as good or better (fan-first roughly doubled CT nodes
        # and runtime on 2d discrete r=2 for both classifiers).
        self.conflict_order = conflict_order
        # icbs_mode='focal' switches the high-level search to bounded-suboptimal
        # focal search (as in ECBS): OPEN stays cost-ordered, and expansion
        # picks the node with the FEWEST conflicts among nodes whose cost is
        # within focal_w x the best open cost. Returned solutions cost at most
        # focal_w x optimal. Blank ('') keeps the plain best-cost-first search.
        if icbs_mode not in ("", "focal"):
            raise ValueError(f"Invalid icbs_mode: {icbs_mode!r} (expected '' or 'focal')")
        self.icbs_mode = icbs_mode
        self.focal_w = focal_w
        self._focal = []
        self._focal_lb = float("-inf")

    def _push_node(self, node):
        """Push a CT node onto OPEN (and FOCAL when within the current bound).

        In focal mode the node's full conflict list is computed here (once)
        and cached on the node: it provides the focal d-value now and is
        reused verbatim when the node is expanded.
        """
        heapq.heappush(self.open_list, (node.cost, next(self.counter), node))
        if self.icbs_mode == "focal":
            if not hasattr(node, "conflicts"):
                node.conflicts = self.env.get_conflicts(node.solution, get_first_conflict=False)
            node.d = len(node.conflicts)
            if node.cost <= self._focal_lb * self.focal_w + 1e-9:
                heapq.heappush(self._focal, (node.d, node.cost, next(self.counter), node))

    def _best_open_cost(self):
        """Min cost among unexpanded OPEN nodes (lazily discarding expanded)."""
        while self.open_list:
            cost, _, node = self.open_list[0]
            if getattr(node, "_expanded", False):
                heapq.heappop(self.open_list)
                continue
            return cost
        return None

    def _pop_focal(self):
        """Pop the fewest-conflicts node with cost <= focal_w x best open cost.

        FOCAL is rebuilt from OPEN whenever the lower bound rises (entries
        below the new bound may have been discarded earlier); stale entries
        (expanded nodes, or entries pushed under an older, larger bound) are
        skipped lazily. Falls back to the best-cost node if FOCAL is empty.
        """
        best = self._best_open_cost()
        if best is None:
            return None
        threshold = best * self.focal_w + 1e-9
        if best > self._focal_lb + 1e-9:
            self._focal_lb = best
            self._focal = [
                (n.d, c, k, n)
                for (c, k, n) in self.open_list
                if c <= threshold and not getattr(n, "_expanded", False)
            ]
            heapq.heapify(self._focal)
        while self._focal:
            _, c, _, n = heapq.heappop(self._focal)
            if c > threshold or getattr(n, "_expanded", False):
                continue
            return n
        while self.open_list:  # all focal entries were stale
            _, _, n = heapq.heappop(self.open_list)
            if not getattr(n, "_expanded", False):
                return n
        return None

    def _order_conflicts(self, conflict_list):
        """Order conflicts before classification (both classifiers stop at the
        first cardinal conflict they find).

        'time' (default) keeps get_conflicts' earliest-first order, which
        measured best. 'fan' tries likely-cardinal-first by descending
        constraint-fan size; on these maps it roughly doubled CT nodes and
        runtime, so it is opt-in only.
        """
        if self.conflict_order == "fan":
            return sorted(
                conflict_list,
                key=lambda c: (-(len(c.location_1_f) + len(c.location_2_f)), c.time),
            )
        return conflict_list

    def _build_opt_table(self, P, agent, t_max, state_cap=40000):
        """Weighted-graph analogue of an MDD for one agent at its current cost.

        Forward Dijkstra over the time-expanded graph under the agent's
        current constraints, keeping only states with g + h <= C (h = exact
        static Dijkstra-to-goal distance), i.e. states on some optimal-cost
        prefix. levels[t] maps location point -> g. Because h ignores
        constraints at later times, membership is a slight over-approximation:
        classification may call a truly-cardinal conflict non-cardinal (weaker
        selection) but never the reverse, so search soundness is unaffected.
        Returns None if the plateau exceeds state_cap (caller falls back to
        A*-based scoring).
        """
        env = self.env
        eps = 1e-6
        C = P.solution_cost[agent]
        agent_info = env.agent_dict[agent]
        goal_pt = agent_info["goal"].location.point
        env.constraints = P.constraint_dict.get(agent, Constraints())
        h_cache = {}
        def h(pt):
            v = h_cache.get(pt)
            if v is None:
                v = env._dijkstra_heuristic(pt, goal_pt)
                h_cache[pt] = v
            return v
        start_pt = agent_info["start"].location.point
        levels = {0: {start_pt: 0.0}}
        heap = [(0.0, 0, start_pt)]
        n_states = 1
        while heap:
            g, t, pt = heapq.heappop(heap)
            if g > levels.get(t, {}).get(pt, float("inf")) + 1e-12:
                continue  # stale entry
            if t >= t_max:
                continue
            st = State(t, Location(pt))
            for n in env.get_neighbors(st):
                npt = n.location.point
                ng = g + env.get_step_cost(st, n)
                if ng + h(npt) > C + eps:
                    continue
                lvl = levels.setdefault(t + 1, {})
                cur = lvl.get(npt)
                if cur is None or ng < cur - 1e-12:
                    if cur is None:
                        n_states += 1
                        if n_states > state_cap:
                            return None
                    lvl[npt] = ng
                    heapq.heappush(heap, (ng, t + 1, npt))
        # earliest time an optimal path can already have ended at the goal
        end_ts = [t for t, lvl in levels.items() if goal_pt in lvl]
        return {
            "levels": levels, "C": C, "h": h, "goal_pt": goal_pt,
            "min_end": min(end_ts) if end_ts else float("inf"),
            "constraints": env.constraints,
        }

    def _is_cardinal_for(self, table, conflict, which):
        """True iff every optimal-cost move of this agent at conflict.time is
        banned by the conflict's constraint for it (per the plateau table)."""
        env = self.env
        eps = 1e-6
        t = conflict.time
        C, levels, h, goal_pt = table["C"], table["levels"], table["h"], table["goal_pt"]
        ended = table["min_end"] <= t
        if conflict.type == Conflict.VERTEX:
            banned_pt = conflict.location_1.point
            for pt in levels.get(t, {}):
                if pt != banned_pt:
                    return False
            return not (ended and goal_pt != banned_pt)
        if conflict.type == Conflict.EDGE:
            if which == 1:
                banned = {(conflict.location_1.point, conflict.location_2.point)}
            else:
                banned = {(conflict.location_2.point, conflict.location_1.point)}
        else:  # SWEEP: start vertex + fan of forbidden destinations
            if which == 1:
                src, fan = conflict.location_1.point, conflict.location_1_f
            else:
                src, fan = conflict.location_2.point, conflict.location_2_f
            banned = {(src, loc.point) for loc in fan}
        env.constraints = table["constraints"]
        for pt, g in levels.get(t, {}).items():
            st = State(t, Location(pt))
            for n in env.get_neighbors(st):
                npt = n.location.point
                if (pt, npt) in banned:
                    continue
                if g + env.get_step_cost(st, n) + h(npt) <= C + eps:
                    return False
        # an already-finished agent sits at its goal (implicit self-loop)
        if ended and (goal_pt, goal_pt) not in banned:
            return False
        return True

    def _get_best_conflict_mdd(self, P, conflict_list):
        """Classify conflicts from plateau tables: 2 table builds per CT node
        replace 2 A* searches per scored conflict. Returns None when any
        needed table overflows (caller falls back to the A* scorer). The
        returned costs are cardinality proxies (1.0 cardinal / 0.0 not), which
        is all the bypass and branching logic consume.
        """
        t_max = max(c.time for c in conflict_list) + 1
        tables = {}
        best = None
        best_score = float("inf")
        for c in conflict_list:
            for a in (c.agent_1, c.agent_2):
                if a not in tables:
                    tables[a] = self._build_opt_table(P, a, t_max)
                if tables[a] is None:
                    return None
            card1 = self._is_cardinal_for(tables[c.agent_1], c, 1)
            card2 = self._is_cardinal_for(tables[c.agent_2], c, 2)
            score = 0 if (card1 and card2) else (1 if (card1 or card2) else 2)
            if score < best_score:
                best_score = score
                best = (c, {c.agent_1: 1.0 if card1 else 0.0,
                            c.agent_2: 1.0 if card2 else 0.0})
                if score == 0:
                    break
        c, costs = best
        return c, costs, None, best_score

    def _get_best_conflict(self, P, conflict_list):
        best_conflict = None
        best_score = float("inf")
        best_costs = None
        best_paths = None

        if self.max_scored_conflicts is not None:
            conflict_list = conflict_list[:self.max_scored_conflicts]
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
        st = time.time()
        iterations = 1
        success = False
        start = HighLevelNode()
        start.constraint_dict = {}
        solution = {}
        solution_info = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()

        start.solution, start.solution_cost = self.env.compute_solution()
        if not start.solution:
            if self.verbose:
                print("No initial solution found")
            self.total_time = min(self.time_limit, time.time() - st) 
            self.total_iterations = min(self.max_iterations, iterations)
            solution_info["runtime"] = self.total_time
            solution_info["total_iterations"] = self.total_iterations
            solution_info["success"] = success
            return solution,solution_info

        start.cost = sum(start.solution_cost.values())

        # Add start node to heap
        self._focal = []
        self._focal_lb = float("-inf")
        self._push_node(start)
        while self.open_list:
            iterations += 1
            if self.time_limit is not None and (time.time() - st) > self.time_limit:
                if self.verbose:
                    print(
                        f"Search terminated: time limit of {self.time_limit} seconds exceeded."
                    )
                break

            if self.max_iterations is not None and iterations >= self.max_iterations:
                if self.verbose:
                    print(
                        f"Search terminated: max iterations of {self.max_iterations} reached."
                    )
                break

            if self.icbs_mode == "focal":
                P = self._pop_focal()
            else:
                _, _, P = heapq.heappop(self.open_list)

            if P is None:
                break
            state_key = self._get_state_key(P)
            P._expanded = True  # purge from OPEN/FOCAL lazily
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)

            self.env.constraint_dict = P.constraint_dict

            # First improvement, we check all of the conflicts and grab the cardinal conflicts to solve first
            conflict_list = getattr(P, "conflicts", None)
            if conflict_list is None:
                conflict_list = self.env.get_conflicts(P.solution,get_first_conflict=False)

            if not conflict_list:
                if self.verbose:
                    print("solution found")
                success = True
                solution = self.generate_plan(P.solution)
                break

            # 1. Prioritize Conflicts (Cardinal, Semi, Non)
            conflict_list = self._order_conflicts(conflict_list)
            result = None
            if self.conflict_classifier == "mdd":
                result = self._get_best_conflict_mdd(P, conflict_list)
            if result is None:  # 'astar' mode, or plateau table overflow
                result = self._get_best_conflict(P, conflict_list)
            best_conflict, best_costs, best_paths, score = result

            # 2. BYPASS STRATEGY
            # If any agent can resolve the conflict with 0 cost increase, check for bypass
            bypass_found = False
            for agent, cost_inc in best_costs.items():
                if cost_inc == 0:
                    # Attempt bypass: get a same-cost path that avoids this conflict
                    temp_constraints = self._get_updated_constraints(P, agent, best_conflict)
                    self.env.constraint_dict = temp_constraints
                    self.env.constraints = self.env.constraint_dict.setdefault(agent, Constraints())
                    new_path, new_cost = self.env.a_star.search(agent, solution=P.solution)

                    # A bypass must keep this agent's cost unchanged; verify
                    # rather than trust the classifier (the mdd classifier's
                    # zero is a cardinality proxy, not a measured cost delta).
                    if new_path and abs(new_cost - P.solution_cost[agent]) <= 1e-6:
                        new_node = HighLevelNode()
                        new_node.solution = P.solution.copy()
                        new_node.solution_cost = P.solution_cost.copy()
                        new_node.solution[agent] = new_path                        
                        new_node.cost = P.cost
                        new_node.constraint_dict = temp_constraints
                        
                        n_before = len(conflict_list)
                        if self.icbs_mode == "focal":
                            # Full enumeration: the child's list doubles as its
                            # focal d-value, so cache it on the node.
                            child_conflicts = self.env.get_conflicts(new_node.solution, False)
                            improved = len(child_conflicts) < n_before
                            if improved:
                                new_node.conflicts = child_conflicts
                        else:
                            # Cap the re-enumeration: we only need to know whether
                            # the count drops below len(conflict_list).
                            improved = len(self.env.get_conflicts(new_node.solution, False, max_conflicts=n_before)) < n_before
                        if improved:
                            self._push_node(new_node)
                            bypass_found = True
                            break
            
            if bypass_found:
                continue
            
            # Branching (if no bypass)
            self._branch(P, best_conflict, best_costs, best_paths)
        self.total_time = min(self.time_limit, time.time() - st) 
        self.total_iterations = min(self.max_iterations, iterations)
        solution_info["runtime"] = self.total_time
        solution_info["total_iterations"] = self.total_iterations
        solution_info["success"] = success
        return solution,solution_info


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

    def _branch(self, P, conflict, costs, paths=None):
        """Expands the high-level tree by creating child nodes."""
        for agent in costs.keys():
            if costs[agent] == float('inf'): continue # Prune if no path exists

            new_node = HighLevelNode()
            new_node.solution = P.solution.copy()
            new_node.solution_cost = P.solution_cost.copy()
            new_node.constraint_dict = self._get_updated_constraints(P, agent, conflict)

            path = paths.get(agent) if paths else None
            if path is not None:
                # Reuse the path computed while scoring this conflict in
                # get_conflict_cost: that search ran under exactly these
                # constraints (P's plus this conflict's constraint for this
                # agent, built identically by _get_updated_constraints), so
                # re-running A* here would repeat identical work.
                new_node.solution[agent] = path
                new_node.solution_cost[agent] = P.solution_cost[agent] + costs[agent]
                new_node.cost = sum(new_node.solution_cost.values())
                self._push_node(new_node)
                continue

            self.env.constraint_dict = new_node.constraint_dict
            # Re-plan only the affected agent
            res = self.env.compute_agent_solution(agent, new_node.solution, new_node.solution_cost)
            
            if res:
                new_node.solution, new_node.solution_cost = res
                new_node.cost = sum(new_node.solution_cost.values())
                self._push_node(new_node)