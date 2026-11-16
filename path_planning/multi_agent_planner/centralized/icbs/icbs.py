"""
Conflict-based search for multi-agent path planning
author: Brandon Ho
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from path_planning.multi_agent_planner.centralized.cbs.cbs import (
    Environment, Constraints, HighLevelNode, CBS, Conflict, State, Location, VertexConstraint, EdgeConstraint, FREE)
from path_planning.multi_agent_planner.centralized.icbs.meta_agents import JointSearch
import heapq
import time
from collections import defaultdict
from path_planning.multi_agent_planner.centralized.icbs.a_star import AStar
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE

class IEnvironment(Environment, JointSearch):
    """The CBS ``Environment`` (same arguments) with ICBS's low-level search and the joint search of
    meta-agents."""

    def __init__(self, graph_map, agents, astar_max_iterations=10000, **options):
        super().__init__(graph_map, agents, astar_max_iterations, **options)
        self.a_star = AStar(self, astar_max_iterations)

    # ---- positive constraints (disjoint splitting): the agent has to be at ``must_be[time]``
    def state_valid(self, state):
        must = self.constraints.must_be
        if must:
            need = must.get(state.time)
            if need is not None and need != state.location:
                return False
        return super().state_valid(state)

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
        disjoint: bool = False,
        merge_bound: int | None = None,
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
                           (agents with bodies) instances. Branching on any conflict
                           keeps CBS complete and optimal, so capping only
                           risks missing a cardinal conflict a full scan would
                           have found. None scans the whole list.
        :param disjoint: Disjoint splitting (point agents only): a conflict
                           branches on ONE agent. One child forces it to make
                           its move (so every other agent is forbidden the
                           colliding moves), the other forbids it.
        :param merge_bound: Meta-agents with merge and restart (point agents,
                           discs and spheres): once two (meta-)agents have been chosen to
                           conflict more than this many times, they are merged
                           and planned jointly from then on, and the search
                           restarts from a fresh root. None: never merge.
        """
        super().__init__(environment, time_limit, max_iterations, verbose)
        self.env = environment
        self.max_scored_conflicts = max_scored_conflicts
        # 'mdd' classifies conflicts from per-agent optimal-plateau tables
        # (one build per agent per CT node, covering every conflict) and
        # falls back to the A*-based scorer when a table exceeds its state
        # cap; 'astar' always uses the A*-based scorer.
        self.conflict_classifier = conflict_classifier
        if disjoint and environment.has_bodies:
            raise ValueError("ICBS: disjoint splitting is implemented for point agents only (no Agent body)")
        if merge_bound is not None and environment.has_bodies and not all(
                environment.agent_type(a).is_ball for a in environment.agent_dict):
            raise ValueError("ICBS: meta-agents are implemented for point agents and for discs / spheres")
        self.disjoint = disjoint
        self.merge_bound = merge_bound
        self._group = {}                         # agent -> its meta-agent (a frozenset of names), merged agents only
        self._pair_conflicts = defaultdict(int)  # frozenset({a, b}) -> times their conflict was chosen
        self.restarts = 0
        self._stop_at = None                     # clock time at which the running search must stop

    # ------------------------------------------------------------ meta-agents
    def _members(self, agent):
        """The meta-agent ``agent`` belongs to (itself alone when it was never merged)."""
        return self._group.get(agent) or frozenset([agent])

    def _replan_agent(self, node, agent) -> bool:
        return self._replan(node, agent)

    def _replan(self, node, agent) -> bool:
        """Replan ``agent`` under ``node.constraint_dict``, with its whole meta-agent when it is merged.
        Updates ``node.solution`` / ``node.solution_cost``; False when there is no plan."""
        self.env.constraint_dict = node.constraint_dict
        group = self._group.get(agent)
        if group is None:
            return bool(self.env.compute_agent_solution(agent, node.solution, node.solution_cost))
        planned = self.env.joint_search(sorted(group), node.constraint_dict, stop_at=self._stop_at)
        if not planned:
            return False
        node.solution.update(planned[0])
        node.solution_cost.update(planned[1])
        return True

    def _root(self):
        """A fresh root: no constraints, every agent on its own shortest path and every meta-agent on its
        joint plan. None when some plan does not exist."""
        root = super()._root()
        if root is None:
            return None
        for group in set(self._group.values()):
            if not self._replan(root, next(iter(group))):
                return None
        root.cost = sum(root.solution_cost.values())
        return root

    def _merge_and_restart(self, conflict) -> bool:
        """Count the chosen ``conflict``; once its two (meta-)agents have conflicted more than
        ``merge_bound`` times, merge them and restart the search from a fresh root. True when restarted."""
        g1, g2 = self._members(conflict.agent_1), self._members(conflict.agent_2)
        self._pair_conflicts[frozenset((conflict.agent_1, conflict.agent_2))] += 1
        if sum(self._pair_conflicts[frozenset((a, b))] for a in g1 for b in g2) <= self.merge_bound:
            return False
        previous, merged = dict(self._group), g1 | g2
        self._group.update({a: merged for a in merged})
        root = self._root()
        if root is None:                             # the joint plan was not found: keep them apart
            self._group = previous
            return False
        self.open_list.clear()
        self.closed_set.clear()
        self.restarts += 1
        self._push_node(root)
        if self.verbose:
            print(f"merged {sorted(merged)}; restart {self.restarts}")
        return True

    # ------------------------------------------------------------ disjoint splitting
    def _branch_disjoint(self, P, conflict):
        """Branch on agent 1 of ``conflict`` only. One child forces it to make its move at that step: every
        agent outside its meta-agent is forbidden the colliding moves, and those now violating are
        replanned. The other child forbids it the move, as in standard splitting."""
        agent, t = conflict.agent_1, conflict.time
        here = conflict.location_1
        forced = Constraints()                       # for the agent
        banned = Constraints()                       # for everyone else
        if conflict.type == Conflict.VERTEX:
            forced.must_be[t] = here
            banned.vertex_constraints.add(VertexConstraint(t, here))
        else:                                        # it goes here -> there over step t
            there = conflict.location_2
            forced.must_be[t], forced.must_be[t + 1] = here, there
            banned.vertex_constraints.update((VertexConstraint(t, here), VertexConstraint(t + 1, there)))
            banned.edge_constraints.add(EdgeConstraint(t, there, here))
        mine = self._members(agent)

        child = HighLevelNode()
        child.solution, child.solution_cost = P.solution.copy(), P.solution_cost.copy()
        child.constraint_dict = {}
        for a, con in P.constraint_dict.items():
            child.constraint_dict[a] = con.copy()
            child.constraint_dict[a].add_constraint(forced if a == agent else banned if a not in mine else Constraints())
        position = lambda a, time: self.env.get_state(a, child.solution, time).location

        def violates(a):
            if conflict.type == Conflict.VERTEX:
                return position(a, t) == here
            return position(a, t) == here or position(a, t + 1) == there or \
                (position(a, t), position(a, t + 1)) == (there, here)

        replanned, feasible = set(), True
        for a in P.solution:
            if a not in mine and a not in replanned and violates(a):
                feasible = self._replan(child, a)
                replanned |= self._members(a)
                if not feasible:
                    break
        if feasible:
            child.cost = sum(child.solution_cost.values())
            self._push_node(child)

        other = HighLevelNode()                      # the agent may not make the move
        other.solution, other.solution_cost = P.solution.copy(), P.solution_cost.copy()
        other.constraint_dict = self._get_updated_constraints(P, agent, conflict)
        if self._replan(other, agent):
            other.cost = sum(other.solution_cost.values())
            self._push_node(other)

    def _build_opt_table(self, P, agent, t_max, state_cap=40000):
        """Weighted-graph analogue of an MDD for one agent at its current cost.

        Forward Dijkstra over the time-expanded graph under the agent's
        current constraints, keeping only states with g + h <= C (h = exact
        static Dijkstra-to-goal distance), i.e. states on some optimal-cost
        prefix. levels[t] maps (location point, heading) -> g. Because h ignores
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
        env.current_agent = agent
        h_cache = {}
        def h(pt):
            v = h_cache.get(pt)
            if v is None:
                v = env._dijkstra_heuristic(pt, goal_pt)
                h_cache[pt] = v
            return v
        start = agent_info["start"]
        start_key = (start.location.point, getattr(start, "ori", None))
        levels = {0: {start_key: 0.0}}
        heap = [(0.0, 0, next(self.counter), start_key)]
        n_states = 1
        while heap:
            g, t, _, key = heapq.heappop(heap)
            if g > levels.get(t, {}).get(key, float("inf")) + 1e-12:
                continue  # stale entry
            if t >= t_max:
                continue
            st = State(t, Location(key[0]), key[1])
            for n in env.get_neighbors(st):
                npt = n.location.point
                ng = g + env.get_step_cost(st, n)
                if ng + h(npt) > C + eps:
                    continue
                lvl = levels.setdefault(t + 1, {})
                nkey = (npt, getattr(n, "ori", None))
                cur = lvl.get(nkey)
                if cur is None or ng < cur - 1e-12:
                    if cur is None:
                        n_states += 1
                        if n_states > state_cap:
                            return None
                    lvl[nkey] = ng
                    heapq.heappush(heap, (ng, t + 1, next(self.counter), nkey))
        # earliest time an optimal path can already have ended at the goal
        end_ts = [t for t, lvl in levels.items() if any(pt == goal_pt for pt, _ in lvl)]
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
            for pt, _ in levels.get(t, {}):
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
                src, fan, no_turn = conflict.location_1.point, conflict.location_1_f, conflict.no_turn_1
            else:
                src, fan, no_turn = conflict.location_2.point, conflict.location_2_f, conflict.no_turn_2
            banned = {(src, loc.point) for loc in fan}
        env.constraints = table["constraints"]
        env.current_agent = conflict.agent_1 if which == 1 else conflict.agent_2
        for (pt, ori), g in levels.get(t, {}).items():
            st = State(t, Location(pt), ori)
            for n in env.get_neighbors(st):
                npt = n.location.point
                if (pt, npt) in banned:
                    continue
                if conflict.type == Conflict.SWEEP and no_turn and pt == src and npt == pt and n.ori != ori:
                    continue                                     # a forbidden turn in place
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
        if self._group and any(c.agent_1 in self._group or c.agent_2 in self._group for c in conflict_list):
            return None                                  # a merged agent is scored by a joint replan instead
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
        """Score conflicts by replanning (``get_conflict_cost``), at most ``max_scored_conflicts`` of them,
        and return the best as ``(conflict, cost increases, paths, score)``: 0 cardinal (both agents'
        costs rise), 1 semi-cardinal, 2 non-cardinal. Stops at the first cardinal one."""
        best = None
        for conflict in conflict_list[:self.max_scored_conflicts]:
            costs, paths = self.get_conflict_cost(P, conflict)
            raised = [c > 0 for c in costs.values()]
            score = 0 if all(raised) else 1 if any(raised) else 2
            if best is None or score < best[3]:
                best = (conflict, costs, paths, score)
                if score == 0:
                    break
        return best

    def get_conflict_cost(self, P: HighLevelNode, conflict: Conflict):
        """For each agent of ``conflict``: how much its cost rises under its constraint (``inf``: no path),
        and the path found (None for a merged agent, which is replanned jointly)."""
        costs, paths = {}, {}
        original = self.env.constraint_dict
        for agent, added in self.env.create_constraints_from_conflict(conflict).items():
            self.env.constraint_dict = constraints = self._with_constraint(P, agent, added)
            group = self._group.get(agent)
            if group is not None:                        # a merged agent: the cost change of its joint plan
                planned = self.env.joint_search(sorted(group), constraints, stop_at=self._stop_at)
                costs[agent] = sum(planned[1].values()) - sum(P.solution_cost[a] for a in group) if planned else float("inf")
                paths[agent] = None
                continue
            self.env.constraints = constraints[agent]
            path, cost = self.env.a_star.search(agent)
            costs[agent] = cost - P.solution_cost[agent] if path else float("inf")
            paths[agent] = path or None
        self.env.constraint_dict = original
        return costs, paths

    def _choose_conflict(self, P, conflict_list):
        """The conflict to branch on, cardinal ones first: ``(conflict, cost increases, paths, score)``.
        ``"mdd"`` classifies from per-agent tables and falls back to replanning when a table is too big
        or an agent is merged; ``"astar"`` always replans."""
        result = self._get_best_conflict_mdd(P, conflict_list) if self.conflict_classifier == "mdd" else None
        return result or self._get_best_conflict(P, conflict_list)

    def _bypass(self, P, conflict, costs, n_conflicts) -> bool:
        """Try to avoid branching on ``conflict``: if one of its agents has another path of the same cost
        that leaves the plan with fewer conflicts, push ``P`` with that path instead and return True. The
        node keeps ``P``'s constraints: adding the one used to find the path would cut off every solution
        where the agent makes the move, with no sibling branch to cover them."""
        for agent, increase in costs.items():
            if increase != 0 or agent in self._group:            # (a merged agent is not bypassed)
                continue
            self.env.constraint_dict = constraints = self._get_updated_constraints(P, agent, conflict)
            self.env.constraints = constraints[agent]
            # Only a path of the agent's current cost is of use, so the search stops early otherwise. The
            # cost is checked rather than trusted: the mdd classifier's zero is a proxy, not a measurement.
            path, cost = self.env.a_star.search(agent, solution=P.solution, cost_limit=P.solution_cost[agent])
            if not path or abs(cost - P.solution_cost[agent]) > 1e-6:
                continue
            node = HighLevelNode()
            node.solution, node.solution_cost = {**P.solution, agent: path}, P.solution_cost.copy()
            node.cost, node.constraint_dict = P.cost, P.constraint_dict
            # (counting stops at n_conflicts: all that matters is whether there are fewer)
            if len(self.env.get_conflicts(node.solution, False, max_conflicts=n_conflicts)) < n_conflicts:
                self._push_node(node)
                return True
        return False

    def search(self):
        """CBS's best-first search with ICBS's improvements at each node: branch on a cardinal conflict
        first, bypass a conflict when an agent has an equally cheap way around it, merge agents that keep
        conflicting (``merge_bound``) and, optionally, split disjointly. Returns ``(plan, solution_info)``."""
        started, iterations, solution = time.time(), 1, {}
        self._stop_at = started + self.time_limit
        root = self._root()
        if root is None:
            if self.verbose:
                print("No initial solution found")
            return solution, self._report(started, iterations, False)
        self._push_node(root)
        while self.open_list:
            iterations += 1
            if self._out_of_budget(started, iterations):
                break
            _, _, P = heapq.heappop(self.open_list)
            state_key = self._get_state_key(P)
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)

            self.env.constraint_dict = P.constraint_dict

            conflict_list = self.env.get_conflicts(P.solution, get_first_conflict=False)
            if not conflict_list:
                if self.verbose:
                    print("solution found")
                solution = self.generate_plan(P.solution)
                break

            # 1. Branch on a cardinal conflict when there is one.
            conflict, costs, paths, _ = self._choose_conflict(P, conflict_list)
            # 2. Bypass: an equally cheap path with fewer conflicts replaces the branching.
            if self._bypass(P, conflict, costs, len(conflict_list)):
                continue
            # 3. Merge and restart: agents that keep conflicting are planned jointly from now on.
            if self.merge_bound is not None and self._merge_and_restart(conflict):
                continue
            # 4. Branch.
            branches = self.env.branches(conflict)
            if any(len(branch) > 1 for branch in branches):      # a goal conflict: one child constrains many agents
                for branch in branches:
                    child = self._child(P, branch, selective=True)
                    if child is not None:
                        self._push_node(child)
            elif self.disjoint and conflict.type in (Conflict.VERTEX, Conflict.EDGE):
                self._branch_disjoint(P, conflict)
            else:
                self._branch(P, conflict, costs, paths)
        return solution, self._report(started, iterations, bool(solution))

    def _report(self, started, iterations, success) -> dict:
        info = super()._report(started, iterations, success)
        info["meta_agents"] = sorted(sorted(g) for g in set(self._group.values()))
        info["restarts"] = self.restarts
        return info


    @staticmethod
    def _with_constraint(P, agent, added):
        """``P``'s constraints with ``added`` joined to ``agent``'s (the other agents' sets are shared, not
        copied)."""
        constraints = dict(P.constraint_dict)
        constraints[agent] = P.constraint_dict[agent].copy()
        constraints[agent].add_constraint(added)
        return constraints

    def _get_updated_constraints(self, P, agent, conflict):
        """``P``'s constraints with ``conflict``'s constraint added for ``agent``."""
        return self._with_constraint(P, agent, self.env.create_constraints_from_conflict(conflict)[agent])

    def _branch(self, P, conflict, costs, paths=None):
        """Standard splitting: one child per agent of ``conflict``, with its constraint added. ``paths``
        holds the paths found while scoring the conflict, which ran under exactly these constraints, so
        they are reused instead of searching again."""
        for agent, cost in costs.items():
            if cost == float("inf"):                             # no path exists under the constraint
                continue
            node = HighLevelNode()
            node.solution, node.solution_cost = P.solution.copy(), P.solution_cost.copy()
            node.constraint_dict = self._get_updated_constraints(P, agent, conflict)
            path = paths.get(agent) if paths else None
            if path is not None:
                node.solution[agent] = path
                node.solution_cost[agent] = P.solution_cost[agent] + cost
            elif not self._replan(node, agent):
                continue
            node.cost = sum(node.solution_cost.values())
            self._push_node(node)
