"""Continuous-time Conflict-Based Search (CCBS) on a roadmap, for agents with bodies.

Follows the reference implementation (github.com/PathPlanning/Continuous-CBS; Andreychuk et al.,
"Multi-Agent Pathfinding with Continuous Time" and "Improving Continuous-time Conflict Based Search"),
with the bodies generalised from equal discs to each agent's ``Agent``.
"""
import heapq
import time
from itertools import count

import numpy as np

from path_planning.agent import Agent
from path_planning.multi_agent_planner.centralized.cbs.data_types import start_heading
from path_planning.multi_agent_planner.centralized.cbs.routing import RoadmapCosts
from path_planning.multi_agent_planner.centralized.ccbs.conflicts import ContinuousConflicts
from path_planning.multi_agent_planner.centralized.ccbs.data_types import (   # noqa: F401  (the types are re-exported)
    EPS, INF, Conflict, Constraint, Constraints, CTNode, Move)
from path_planning.multi_agent_planner.centralized.ccbs.low_level import SIPP
from path_planning.multi_agent_planner.sweep_cache import shared_sweep_cache


class Environment(RoadmapCosts, ContinuousConflicts):
    """The roadmap, the agents and their bodies, as CCBS uses them.

    ``agents`` is a list of dicts: ``name``, ``start``, ``goal`` (roadmap nodes), ``agent`` (an ``Agent``:
    the body) and optionally ``yaw`` / ``pitch`` (the start heading, degrees; default 0). Everything about a body comes
    from its ``Agent``:

    - Size: its shape (2D or 3D) and ``clearance``. Every agent needs one: point particles never meet
      in continuous time.
    - Timing: a move takes ``length / max_speed``; ``max_speed=None`` means 1 time unit per move. Waits
      can have any length.
    - Turning is instant (``max_rotation`` is not used yet): a body faces along each move and keeps that
      heading while it waits.

    ``precision``: how finely the end of an unsafe interval is located.
    """

    def __init__(self, graph_map, agents, sipp_max_iterations: int = 200_000, precision: float = 1e-6):
        missing = [a["name"] for a in agents if a.get("agent") is None]
        if missing:
            raise ValueError(f'CCBS: every agent dict needs an "agent" (an Agent); missing for {missing}')
        self.graph_map, self.agents, self.precision = graph_map, agents, float(precision)
        # Turns are instant here, so the bodies carry no turning rate.
        self.bodies = {a["name"]: Agent(a["agent"].shape, a["agent"].max_speed, None, a["agent"].type, a["agent"].clearance)
                       for a in agents}
        graph_map.set_collision_sweep()
        self.cache = shared_sweep_cache(graph_map, "ccbs")

        # The roadmap by node index (RoadmapCosts).
        self._moves_cache, self._links = {}, None
        self.links = self._roadmap_links()               # (successors, predecessors): [(node, heading, length)]
        self.xyz = np.array([n.current for n in graph_map.nodes], dtype=float)
        self.length = {(u, v): length for u, out in enumerate(self.links[0]) for v, _, length in out}
        self.facing = {(u, v): facing for u, out in enumerate(self.links[0]) for v, facing, _ in out}
        self._time_maps = {}                             # (target node, speed) -> {node: quickest time to it}
        self._moves = {}                                 # (agent, path) -> its Moves and their arrays (moves_of)
        self._successors = {}                            # speed -> per node [(next node, duration)]
        self._pairs = {}                                 # (agent, path, agent, path) -> first contact or None

        self.endpoints, self.start_ori = {}, {}
        for a in agents:
            ends = self._node_index(a["start"]), self._node_index(a["goal"])
            if None in ends:
                raise ValueError(f"CCBS: the start and goal of {a['name']} have to be roadmap nodes")
            self.endpoints[a["name"]] = ends
            self.start_ori[a["name"]] = start_heading(a)       # the given heading, else 0
        self.sipp = SIPP(self, sipp_max_iterations)

    def duration(self, agent, length: float) -> float:
        """How long ``agent`` takes over a move of ``length``."""
        speed = self.bodies[agent].max_speed
        return 1.0 if speed is None else length / speed

    def successors(self, agent) -> list:
        """Per node, the moves out of it for ``agent``: ``[(next node, duration)]``, built once per speed."""
        speed = self.bodies[agent].max_speed
        out = self._successors.get(speed)
        if out is None:
            out = self._successors[speed] = [[(v, self.duration(agent, length)) for v, _, length in moves]
                                             for moves in self.links[0]]
        return out

    def time_to(self, agent, target: int) -> list:
        """Per node, the quickest time from it to ``target`` for ``agent`` (ignoring the other agents;
        ``inf`` when it cannot get there). The exact estimate the low-level search uses."""
        key = (target, self.bodies[agent].max_speed)
        times = self._time_maps.get(key)
        if times is None:
            pred = self.links[1]
            times, heap = [INF] * len(pred), [(0.0, target)]
            times[target] = 0.0
            while heap:
                d, v = heapq.heappop(heap)
                if d > times[v]:
                    continue
                for u, _, length in pred[v]:
                    nd = d + self.duration(agent, length)
                    if nd < times[u]:
                        times[u] = nd
                        heapq.heappush(heap, (nd, u))
            self._time_maps[key] = times
        return times


class CCBS:
    """The high-level search. Options, as in the reference implementation:

    - ``use_cardinal``: classify each conflict by replanning its two agents with their constraints, and
      branch first on a cardinal one (both costs rise), then a semi-cardinal one (one rises).
    - ``disjoint``: disjoint splitting. One child forbids an agent its move in the unsafe window; the
      other makes that agent take it there (a landmark) and forbids the other agent its move.
    - ``hlh``: a lower bound on what the cardinal conflicts still cost, added to a node's cost:
      ``"lp"`` (a small linear program) or ``"greedy"`` (a matching). Needs ``use_cardinal``.

    ``search()`` returns ``(plan, solution_info)``; the plan is empty when none was found in the budget.
    """

    def __init__(self, environment: Environment, time_limit=None, max_iterations=None, verbose: bool = False,
                 use_cardinal: bool = False, disjoint: bool = False, hlh: "str | None" = None):
        if hlh not in (None, "lp", "greedy"):
            raise ValueError(f"hlh must be None, 'lp' or 'greedy', got {hlh!r}")
        if hlh is not None and not use_cardinal:
            raise ValueError("hlh needs use_cardinal=True: the bound is built from the cardinal conflicts")
        self.env = environment
        self.time_limit = time_limit if time_limit and time_limit > 0 else INF
        self.max_iterations = max_iterations if max_iterations and max_iterations > 0 else INF
        self.verbose, self.use_cardinal, self.disjoint, self.hlh = verbose, use_cardinal, disjoint, hlh
        self._lp_cache = {}                              # cardinal conflict set -> LP bound

    # ------------------------------------------------------------------ search
    def search(self):
        env, sipp = self.env, self.env.sipp
        started = time.time()
        self._stop_at = started + self.time_limit
        self._ids, self._generated, self._capped = count(1), 0, 0
        self._branched = {"cardinal": 0, "semi_cardinal": 0, "non_cardinal": 0}
        expanded_before, iterations, solved, open_list = sipp.expanded, 0, None, []

        root = CTNode()
        for agent in env.endpoints:
            path, cost = self._plan(agent, ())
            if path is None:
                return {}, self._report(started, iterations, expanded_before, None)
            root.paths[agent], root.costs[agent] = path, cost
        root.cost = sum(root.costs.values())
        if self._add_conflicts(root, env.find_conflicts(root.paths)):
            self._push(open_list, root)

        while open_list and time.time() < self._stop_at and iterations < self.max_iterations:
            node = heapq.heappop(open_list)[-1]
            iterations += 1
            if not node.conflicts:
                solved = node
                break
            conflict, kind = self._pick(node)
            self._branched[kind] += 1
            replans = self._replans(node, conflict)
            if time.time() >= self._stop_at:
                break
            (a1, (c1, path1, cost1)), (a2, (c2, path2, cost2)) = replans.items()
            positive = {}                                # agent whose child also gets a landmark -> the landmark
            if self.disjoint:
                landmark = self._landmark(node, conflict, replans)
                if landmark is not None:                 # the child that constrains the other agent carries it
                    positive[a2 if landmark.agent == a1 else a1] = landmark
            for agent, constraint, path, cost in ((a1, c1, path1, cost1), (a2, c2, path2, cost2)):
                if path is None:
                    continue
                child = CTNode(node, constraint, positive.get(agent))
                child.paths[agent], child.costs[agent] = path, cost
                child.cost = node.cost + cost - node.cost_of(agent)
                if not self._consistent(child):
                    continue
                child.cardinal = [c for c in node.cardinal if agent not in (c.agent_1, c.agent_2)]
                child.semi_cardinal = [c for c in node.semi_cardinal if agent not in (c.agent_1, c.agent_2)]
                child.non_cardinal = [c for c in node.non_cardinal if agent not in (c.agent_1, c.agent_2)]
                if self._add_conflicts(child, env.find_conflicts(child.all_paths(), only=agent)):
                    self._push(open_list, child)
        if self.verbose:
            print("solution found" if solved else "no solution found")
        return (self.generate_plan(solved.all_paths()) if solved else {}), self._report(started, iterations, expanded_before, solved)

    def _plan(self, agent, constraints):
        path, cost = self.env.sipp.find_path(agent, constraints, self._stop_at)
        self._capped += self.env.sipp.timed_out and time.time() < self._stop_at
        return path, cost

    def _push(self, open_list, node):
        node.id = next(self._ids)
        self._generated += 1
        node.h = self._bound(node.cardinal)
        # Cheapest first; then fewer conflicts, then more constraints (deeper nodes).
        heapq.heappush(open_list, (node.cost + node.h, len(node.conflicts), -node.total_cons, node.id, node))

    # ------------------------------------------------------------------ conflicts
    def _replans(self, node, conflict) -> dict:
        """For each agent of ``conflict``: ``(its constraint, its new path under its constraints in
        ``node`` plus that one, the path's cost)``; the path is None when there is none. Kept on the
        conflict and reused while the agent's constraints are the ones it was planned under."""
        moves = {conflict.agent_1: (conflict.move_1, conflict.agent_2, conflict.move_2),
                 conflict.agent_2: (conflict.move_2, conflict.agent_1, conflict.move_1)}
        kept = conflict.replans or {}
        result, fresh = {}, {}
        for agent, (move, other, other_move) in moves.items():
            held = node.constraints_of(agent)
            if agent in kept and kept[agent][3] == len(held):
                fresh[agent] = kept[agent]
            else:
                constraint = self.env.constraint_for(agent, move, other, other_move, conflict.t)
                path, cost = self._plan(agent, held + [constraint])
                fresh[agent] = (constraint, path, cost, len(held))
            result[agent] = fresh[agent][:3]
        conflict.replans = fresh
        return result

    def _add_conflicts(self, node, conflicts) -> bool:
        """File ``conflicts`` under the node's lists (by kind with ``use_cardinal``). False: one of them
        cannot be resolved by either agent, so the node is a dead end."""
        for conflict in conflicts:
            if not self.use_cardinal:
                node.non_cardinal.append(conflict)
                continue
            replans = self._replans(node, conflict)
            if time.time() >= self._stop_at:
                return True                              # out of time: the search loop stops next
            rises = [None if path is None else cost - node.cost_of(agent) for agent, (_, path, cost) in replans.items()]
            if rises[0] is None and rises[1] is None:
                return False
            possible = [r for r in rises if r is not None]
            if len(possible) == 1 or min(possible) > EPS:
                conflict.overcost = min(possible)
                node.cardinal.append(conflict)
            elif max(possible) > EPS:
                node.semi_cardinal.append(conflict)
            else:
                node.non_cardinal.append(conflict)
        return True

    @staticmethod
    def _pick(node):
        """The conflict to branch on: a cardinal one if any, else semi-cardinal, else any; among those the
        largest ``overcost``, then the latest (as the reference implementation does)."""
        for kind in ("cardinal", "semi_cardinal", "non_cardinal"):
            conflicts = getattr(node, kind)
            if conflicts:
                return max(conflicts, key=lambda c: (c.overcost, c.t)), kind

    def _bound(self, cardinal) -> float:
        """A lower bound on the cost the cardinal conflicts still add (0 without ``hlh``)."""
        if self.hlh is None or not cardinal:
            return 0.0
        if self.hlh == "greedy":                         # disjoint pairs, costliest first
            used, total = set(), 0.0
            for c in sorted(cardinal, key=lambda c: -c.overcost):
                if c.agent_1 not in used and c.agent_2 not in used:
                    used.update((c.agent_1, c.agent_2))
                    total += c.overcost
            return total
        need = {}                                        # per agent pair, the largest overcost
        for c in cardinal:
            pair = (c.agent_1, c.agent_2) if c.agent_1 < c.agent_2 else (c.agent_2, c.agent_1)
            need[pair] = max(need.get(pair, 0.0), c.overcost)
        agents = [a for pair in need for a in pair]
        if len(agents) == len(set(agents)):              # the pairs share no agent: the LP's optimum is the sum
            return sum(need.values())
        key = frozenset(need.items())
        known = self._lp_cache.get(key)
        if known is None:                                # minimise sum x, x_a + x_b >= overcost per pair
            from scipy.optimize import linprog
            index = {a: k for k, a in enumerate(sorted(set(agents)))}
            rows = np.zeros((len(need), len(index)))
            for r, (a1, a2) in enumerate(need):
                rows[r, index[a1]] = rows[r, index[a2]] = -1.0
            result = linprog(np.ones(len(index)), A_ub=rows, b_ub=[-w for w in need.values()], bounds=(0, None), method="highs")
            known = self._lp_cache[key] = float(result.fun) if result.success else 0.0
        return known

    # ------------------------------------------------------------------ disjoint splitting
    def _landmark(self, node, conflict, replans):
        """The positive constraint for this split, or None: one agent has to start its move inside its
        unsafe window (it does so now, so it is not replanned). Chosen as in the reference implementation:
        agent 1 when agent 2 already has more landmarks, else agent 2, else agent 1; never for a wait, for
        an agent that could not give way, or a window nested with one of its landmarks."""
        (a1, (c1, path1, _)), (a2, (c2, path2, _)) = replans.items()
        held = {a: [c for c in node.constraints_of(a) if c.positive] for a in (a1, a2)}

        def usable(agent, constraint, path):
            if constraint.is_wait or path is None:
                return None
            for c in held[agent]:                        # no window inside another for the same move
                if (c.a, c.b) == (constraint.a, constraint.b) and (
                        (c.t1 - EPS <= constraint.t1 and constraint.t2 <= c.t2 + EPS)
                        or (constraint.t1 - EPS <= c.t1 and c.t2 <= constraint.t2 + EPS)):
                    return None
            return constraint._replace(positive=True)

        if len(held[a2]) > len(held[a1]):
            return usable(a1, c1, path1) or usable(a2, c2, path2)
        return usable(a2, c2, path2) or usable(a1, c1, path1)

    @staticmethod
    def _consistent(node) -> bool:
        """Whether no landmark of the node's agents is wholly forbidden by a negative constraint on the
        same move."""
        for agent in {c.agent for c in (node.constraint, node.positive) if c is not None}:
            held = node.constraints_of(agent)
            for p in (c for c in held if c.positive):
                if any(not c.positive and (c.a, c.b) == (p.a, p.b) and c.t1 - EPS <= p.t1 and p.t2 <= c.t2 + EPS
                       for c in held):
                    return False
        return True

    # ------------------------------------------------------------------ results
    def _report(self, started, iterations, expanded_before, solved) -> dict:
        return {
            "runtime": min(time.time() - started, self.time_limit),
            "total_iterations": iterations,
            "success": solved is not None,
            "total_conflicts": sum(self._branched.values()),
            "high_level_nodes_expanded": iterations,
            "high_level_nodes_generated": self._generated,
            "cardinal_conflicts": self._branched["cardinal"],
            "semi_cardinal_conflicts": self._branched["semi_cardinal"],
            "non_cardinal_conflicts": self._branched["non_cardinal"],
            "low_level_expanded": self.env.sipp.expanded - expanded_before,
            "low_level_capped": int(self._capped),
            "cost": solved.cost if solved is not None else None,
        }

    def generate_plan(self, paths: dict) -> dict:
        """``{agent: [{t, x, y[, z], yaw[, pitch]}, ...]}``: a wait is the same position at a later ``t``.
        The heading is that of the last move made (before the first: the start heading, given or 0)."""
        env, plan = self.env, {}
        for agent, path in paths.items():
            facing = env.start_ori[agent]
            rows = []
            for k, (v, t) in enumerate(path):
                if k and path[k - 1][0] != v:
                    facing = env.facing[(path[k - 1][0], v)]
                row = {"t": float(t), **dict(zip("xyz", map(float, env.xyz[v])))}
                row.update(zip(("yaw", "pitch"), facing) if isinstance(facing, tuple) else {"yaw": facing})
                rows.append(row)
            plan[agent] = rows
        return plan
