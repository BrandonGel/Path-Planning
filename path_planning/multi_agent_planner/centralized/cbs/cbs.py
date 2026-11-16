"""
Modified Conflict-based search for multi-agent path planning
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

from typing import Any
from path_planning.common.environment.node import Node
from path_planning.multi_agent_planner.centralized.cbs.a_star import AStar
import heapq
import time
from math import fabs
from itertools import count,combinations
from path_planning.agent import Agent
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE
from path_planning.multi_agent_planner.centralized.cbs.bodies import BodyConflicts
from path_planning.multi_agent_planner.centralized.cbs.data_types import (   # noqa: F401  (re-exported)
    FREE, Conflict, Constraints, EdgeConstraint, HighLevelNode, Location, State, Step, VertexConstraint,
    _is_set, _make_ori, _ori_between)

class Environment(BodyConflicts):
    """CBS on a roadmap with unit time steps: every move and every wait takes 1 time unit.

    Agent bodies: ``agents[i]["agent"]`` (an ``Agent`` of any shape, 2D or 3D), else ``agent`` (one type for
    all), else a disc / sphere of ``radius``; without any of them the agents are points. Only an ``Agent``'s
    shape is used: its speed and turning rate, and ``velocity``, do not apply to unit steps.

    - Heading (``use_orientation``; default: agents whose body is not a ball): the heading is part of the
      state. An agent moves only along the edge it faces, and turning in place to another edge's direction
      takes one step, during which it is checked as its bounding ball. ``agents[i]["yaw"]`` (and
      ``"pitch"`` in 3D, default 0) is the start heading in degrees; without it the agent leaves in any
      direction. The plan gets ``yaw`` (and ``pitch`` in 3D) per step.
    - ``move_parked_agents`` (default True): an agent does not end its plan at its goal while a constraint
      still forbids it being or staying there; it moves off and comes back. This is the correct rule, and
      what makes point-agent searches optimal. False: its plan ends the first time it reaches the goal,
      so a parked agent is never moved and the other agent always goes around. Faster on some instances
      with bodies, but it can miss cheaper plans and problems where a parked agent has to give way.
    - Conflicts of agents with bodies: one ``graph_map.agent_conflicts`` call over the whole plan (when and
      where two bodies overlap).
    - ``goal_conflicts`` (agents with bodies, with ``move_parked_agents``): a conflict with an agent that
      has already stopped at its goal is resolved in one branching. Either that agent may not be stopped
      there yet at that step (it arrives later, or moves off and comes back), or it stays stopped and
      every other agent keeps out of the space its body covers from that step on. Without it the second
      branch only forbids the other agent's moves at that one step, which a slightly different route gets
      around, over and over.
    - ``conflict_span`` (agents with bodies): ``"step"`` (default): each time step of an overlap is its own
      conflict, so an overlap lasting k steps can take k branchings. ``"window"``: one conflict per
      overlap; a branch forbids its agent the colliding moves at every step of it at once. Far fewer
      nodes when two agents stay in each other's way for long, but a plan in which each agent keeps a
      different step of the overlap is not reachable from either branch, so it can cost more or miss a
      solution.
    - Constraints: per conflict, one sweep of each agent's move at that step; the other agent may not
      leave its vertex along a roadmap edge that move covers. ``clearance``: ``"radius"`` or ``"minkowski"`` as in
      ``GraphSampler.sweep_paths``."""

    def __init__(self, graph_map:GraphSampler, agents, astar_max_iterations=10000, radius = 0.0, velocity = 0.0, heuristic_type: str = 'dijkstra', agent: "Agent | None" = None, clearance: str = "radius", use_orientation: "bool | None" = None, move_parked_agents: bool = True, conflict_span: str = "step", goal_conflicts: bool = False):
        self.goal_conflicts = goal_conflicts
        self._regions = {}                   # (parked step, other body) -> (vertices, edges) it keeps out of
        if conflict_span not in ("step", "window"):
            raise ValueError(f"conflict_span must be 'step' or 'window', got {conflict_span!r}")
        self.conflict_span = conflict_span
        self.graph_map = graph_map
        self.use_orientation = use_orientation
        self.move_parked_agents = move_parked_agents
        self._moves_cache = {}               # point -> [(neighbour point, its Location, ori facing it)]
        default = agent if agent is not None else (Agent.disc(radius) if radius > 0 else None)
        self._default_agent = self._unit_step(default)
        self.agent_types = {a["name"]: self._unit_step(a["agent"]) for a in agents if a.get("agent") is not None}
        self.has_bodies = self._default_agent is not None or bool(self.agent_types)
        if self.has_bodies and self._default_agent is None and len(self.agent_types) < len(agents):
            raise ValueError("CBS: every agent needs a body (an Agent or a radius) once any agent has one")
        self.clearance = clearance
        if self.has_bodies:
            self.graph_map.set_collision_sweep()
        self.agents = agents
        self.agent_dict = {}
        if heuristic_type not in HEURISTIC_TYPE or heuristic_type is None:
            self.heuristic_type = HEURISTIC_TYPE["dijkstra"]
        else:
            self.heuristic_type = HEURISTIC_TYPE[heuristic_type]
        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self, astar_max_iterations)
        self._dijkstra_cache = {}  # (start_idx, goal_idx) -> (states, cost)
        self._heuristic_cache = {}  # (location.point, agent_name) -> heuristic value
        self._goal_dist_maps = {}  # goal_idx -> {node_idx: shortest-path cost to goal}

    def _static_shortest_path_states(self, agent_name: str):
        """Compute a static shortest path on the roadmap (no time expansion).

        Safe to use only when there are no time-indexed constraints for this agent.
        Returns (states, cost) in the same format as A*.
        """
        start_pt = tuple(float(x) for x in self.agent_dict[agent_name]["start"].location.point)
        goal_pt = tuple(float(x) for x in self.agent_dict[agent_name]["goal"].location.point)
        start_node = Node(start_pt, None, 0, 0)
        goal_node = Node(goal_pt, None, 0, 0)
        pos_index_dict = getattr(self.graph_map, "pos_index_dict", {})
        if start_node.current not in pos_index_dict or goal_node.current not in pos_index_dict:
            return False, float("inf")
        s = int(pos_index_dict[start_node.current])
        g = int(pos_index_dict[goal_node.current])
        if s == g:
            return [State(0, Location(start_pt))], 0.0

        cache_key = (s, g)
        if cache_key in self._dijkstra_cache:
            return self._dijkstra_cache[cache_key]

        road_map = getattr(self.graph_map, "road_map", None)
        if road_map is None or len(road_map) == 0:
            return False, float("inf")

        # Dijkstra over node indices.
        INF = float("inf")
        dist = {s: 0.0}
        prev = {}
        heap = [(0.0, s)]
        seen = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            if u == g:
                break
            u_node = self.graph_map.nodes[u]
            for v in road_map[u]:
                nd = d + float(self.graph_map.get_distance(u_node.current, self.graph_map.nodes[v].current))
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if g not in dist:
            return False, float("inf")

        # Reconstruct path indices.
        path_idx = [g]
        cur = g
        while cur != s:
            cur = prev[cur]
            path_idx.append(cur)
        path_idx.reverse()
        states = []
        for t, idx in enumerate(path_idx):
            pt = tuple(float(x) for x in self.graph_map.nodes[idx].current)
            states.append(State(t, Location(pt)))
        result = states, float(dist[g])
        self._dijkstra_cache[cache_key] = result
        return result

    def _goal_distance_map(self, goal_idx: int):
        """Single-source Dijkstra from the goal over the roadmap.

        Roadmap edges are bidirectional with symmetric costs, so distances from
        the goal equal shortest-path costs to the goal from every node. Computed
        once per goal and cached; used as the exact (admissible, consistent)
        low-level A* heuristic.
        """
        cached = self._goal_dist_maps.get(goal_idx)
        if cached is not None:
            return cached
        road_map = getattr(self.graph_map, "road_map", None)
        dist = {goal_idx: 0.0}
        if road_map is None or len(road_map) == 0:
            self._goal_dist_maps[goal_idx] = dist
            return dist
        heap = [(0.0, goal_idx)]
        seen = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            u_node = self.graph_map.nodes[u]
            for v in road_map[u]:
                nd = d + float(self.graph_map.get_distance(u_node.current, self.graph_map.nodes[v].current))
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._goal_dist_maps[goal_idx] = dist
        return dist

    def _moves_from(self, point):
        """The roadmap moves out of ``point``: ``[(neighbour point, its Location, ori facing it)]``, cached."""
        moves = self._moves_cache.get(point)
        if moves is None:
            targets = [tuple(n.current) for n in self.graph_map.get_neighbors(Node(tuple[Any, ...](point)))]
            moves = self._moves_cache[point] = [(q, Location(q), _ori_between(point, q)) for q in targets]
        return moves

    def get_neighbors(self, state):
        """The states one step after ``state``: wait; move along a roadmap edge (an agent with a heading only
        along the edge it faces, any edge while ``FREE``); turn in place to face another edge."""
        t, ori = state.time + 1, state.ori
        steps = [State(t, state.location, ori)]
        turns = {}                                               # an ordered set of the other headings
        for _, location, facing in self._moves_from(state.location.point):
            if ori is None:
                steps.append(State(t, location))
            elif ori == FREE or ori == facing:
                steps.append(State(t, location, facing))
            else:
                turns.setdefault(facing)
        steps += [State(t, state.location, facing) for facing in turns]
        return [n for n in steps if self.state_valid(n) and self.transition_valid(state, n)]

    def get_conflicts(self, solution, get_first_conflict: bool = True, max_conflicts: int | None = None):
        """Enumerate conflicts; ``max_conflicts`` stops the sweep once that
        many are collected (callers that only compare a count against a known
        bound don't need the full list)."""
        max_t = max(len(plan) for plan in solution.values())
        if self.has_bodies:
            return self._body_conflicts(solution, max_t, get_first_conflict, max_conflicts)

        conflicts = []
        agent_pairs = list(combinations(solution.keys(), 2))
        for t in range(max_t):
            for agent_1, agent_2 in agent_pairs:
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)

                if state_1.is_equal_except_time(state_2):
                    c = Conflict()
                    c.time = t
                    c.type = Conflict.VERTEX
                    c.location_1 = state_1.location
                    c.agent_1 = agent_1
                    c.agent_2 = agent_2
                    if get_first_conflict:
                        return c
                    conflicts.append(c)
                    if max_conflicts is not None and len(conflicts) >= max_conflicts:
                        return conflicts
                    continue

                state_1b = self.get_state(agent_1, solution, t + 1)
                state_2b = self.get_state(agent_2, solution, t + 1)
                if state_1.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2):
                    c = Conflict()
                    c.time = t
                    c.type = Conflict.EDGE
                    c.agent_1 = agent_1
                    c.agent_2 = agent_2
                    c.location_1 = state_1.location
                    c.location_2 = state_1b.location
                    if get_first_conflict:
                        return c
                    conflicts.append(c)
                    if max_conflicts is not None and len(conflicts) >= max_conflicts:
                        return conflicts
        return conflicts

    def branches(self, conflict) -> list:
        """The ways to resolve ``conflict``: a list of ``{agent: Constraints to add}``, one per child.

        Normally two children, each constraining one of the two agents. A goal conflict
        (``goal_conflicts``: one of the two has already stopped at its goal) instead gives:

        1. the stopped agent may not stay at its goal over that step, so it is not stopped there yet;
        2. it has stopped by that step for good, and every other agent keeps out of the roadmap vertices
           and edges its body covers from then on.

        (2) falls back to the normal child when another agent's goal lies in that space."""
        standard = self.create_constraints_from_conflict(conflict)
        normal = [{agent: constraints} for agent, constraints in standard.items()]
        if not (self.goal_conflicts and self.move_parked_agents and conflict.parked):
            return normal
        name, step = conflict.parked
        hold = Constraints()
        hold.deadline = conflict.time
        keep_out = {name: hold}
        for other, info in self.agent_dict.items():
            if other == name:
                continue
            vertices, edges = self._region(step, self.agent_type(other))
            if info["goal"].location in vertices:
                return normal
            keep_out[other] = Constraints()
            keep_out[other].regions.append((conflict.time, vertices, edges))
        return [{name: standard[name]}, keep_out]

    def path_allowed(self, agent, path, constraints) -> bool:
        """Whether ``agent``'s current ``path`` already obeys ``constraints`` (then it needs no replanning)."""
        self.constraints = constraints
        if not all(self.state_valid(s) for s in path[1:]):
            return False
        if not all(self.transition_valid(a, b) for a, b in zip(path, path[1:])):
            return False
        end = path[-1]                                           # then it stays where it ended, for good
        if any(end.location in vertices for _, vertices, _ in constraints.regions):
            return False
        return end.time >= constraints.free_from(end.location)

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
        elif conflict.type == Conflict.SWEEP:
            constraint1 = Constraints()
            constraint2 = Constraints()
            for c in [conflict, *conflict.later]:                # every step of the overlap ("window" span)
                constraint1.edge_constraints.update(EdgeConstraint(c.time, c.location_1, loc) for loc in c.location_1_f)
                constraint2.edge_constraints.update(EdgeConstraint(c.time, c.location_2, loc) for loc in c.location_2_f)
                if c.no_turn_1:
                    constraint1.edge_constraints.add(EdgeConstraint(c.time, c.location_1, c.location_1, True))
                if c.no_turn_2:
                    constraint2.edge_constraints.add(EdgeConstraint(c.time, c.location_2, c.location_2, True))
            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        con = self.constraints
        if con.deadline is not None and state.time > con.deadline:
            return False
        for start, vertices, _ in con.regions:
            if state.time >= start and state.location in vertices:
                return False
        return  VertexConstraint(state.time,state.location) not in con.vertex_constraints

    def transition_valid(self, state_1, state_2):
        for start, _, edges in self.constraints.regions:
            if state_1.time >= start and (state_1.location, state_2.location) in edges:
                return False
        banned = self.constraints.edge_constraints
        if not banned:
            return True
        if EdgeConstraint(state_1.time, state_1.location, state_2.location) in banned:
            return False
        turning = state_1.location == state_2.location and state_1.ori != state_2.ori
        return not (turning and EdgeConstraint(state_1.time, state_1.location, state_1.location, True) in banned)

    def admissible_heuristic(self, state, agent_name):
        key = (state.location.point, agent_name)
        cached = self._heuristic_cache.get(key)
        if cached is not None:
            return cached
        goal_pt = self.agent_dict[agent_name]["goal"].location.point
        loc_pt = state.location.point
        if self.heuristic_type == HEURISTIC_TYPE["manhattan"]:
            result = sum(fabs(a - b) for a, b in zip(loc_pt, goal_pt))
        elif self.heuristic_type == HEURISTIC_TYPE["euclidean"]:
            result = sum((a - b) ** 2 for a, b in zip(loc_pt, goal_pt)) ** 0.5
        elif self.heuristic_type == HEURISTIC_TYPE["dijkstra"]:
            result = self._dijkstra_heuristic(loc_pt, goal_pt)
        else:
            raise ValueError(f"Invalid heuristic type: {self.heuristic_type}")
        self._heuristic_cache[key] = result
        return result

    def _dijkstra_heuristic(self, loc_pt, goal_pt):
        """True roadmap shortest-path distance from loc_pt to goal_pt.

        Falls back to euclidean distance (an admissible lower bound, since move
        costs are euclidean edge lengths) when either point is not a roadmap
        node or the goal is unreachable from loc_pt.
        """
        pos_index_dict = getattr(self.graph_map, "pos_index_dict", {})
        loc_node = Node(tuple(float(x) for x in loc_pt), None, 0, 0)
        goal_node = Node(tuple(float(x) for x in goal_pt), None, 0, 0)
        if loc_node.current in pos_index_dict and goal_node.current in pos_index_dict:
            dist = self._goal_distance_map(int(pos_index_dict[goal_node.current]))
            d = dist.get(int(pos_index_dict[loc_node.current]))
            if d is not None:
                return d
        return sum((a - b) ** 2 for a, b in zip(loc_pt, goal_pt)) ** 0.5

    def get_step_cost(self, state_1: State, state_2: State) -> float:
        """Incremental cost between consecutive time-expanded states.

        Matches roadmap edge weights via ``GraphSampler.get_cost`` for moves,
        so low-level costs align with discrete Dijkstra in
        ``_static_shortest_path_states``. Waiting in place consumes one timestep
        with unit cost so idle steps are not free.
        """
        p1 = tuple(state_1.location.point)
        p2 = tuple(state_2.location.point)
        if p1 == p2:
            return 1.0
        node1 = Node(tuple[Any, ...](p1), None, 0, 0)
        node2 = Node(tuple[Any, ...](p2), None, 0, 0)
        return float(self.graph_map.get_distance(node1.current, node2.current))

    def is_at_goal(self, state, agent_name):
        """Whether the agent can end its plan in ``state``: at its goal, with no constraint left that
        forbids it being or staying there later (it then has to move off and come back)."""
        goal_state = self.agent_dict[agent_name]["goal"]
        if not state.is_equal_except_time(goal_state):
            return False
        if self.move_parked_agents:
            return state.time >= self.constraints.free_from(goal_state.location)
        return state.time >= max(self.constraints.must_be, default=float("-inf"))     # (forced moves still hold)

    def make_agent_dict(self,t=0):
        for agent in self.agents:
            ori = None
            if self.has_orientation(agent['name']):              # the start heading, or free to leave any way
                ori = FREE
                if agent.get('yaw') is not None:
                    ori = _make_ori(agent['yaw'], agent.get('pitch', 0.0) if len(agent['start']) == 3 else None)
            start_state = State(t, Location(agent['start']), ori)
            goal_state = State(t, Location(agent['goal']))
            
            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    def compute_solution(self):
        # Clear caches for fresh solution attempt
        solution = {}
        solution_cost = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            if (
                self.constraints.is_empty()
                and self.agent_dict[agent]["start"].ori is None      # turns cost steps: needs the time-expanded search
            ):
                local_solution, local_cost = self._static_shortest_path_states(agent)
            else:
                local_solution, local_cost = self.a_star.search(agent)
            if not local_solution:
                return False, float('inf')
            solution.update({agent:local_solution})
            solution_cost[agent] = local_cost
        return solution, solution_cost

class CBS(object):
    def __init__(self, environment: Environment, time_limit: float | None = None, max_iterations: int | None = None,verbose: bool = False):
        """
        :param environment: Environment instance
        :param time_limit: Optional wall-clock time limit (in seconds) for the
                           high-level CBS search. If None, no time limit is
                           enforced. If exceeded, the search terminates early
                           and returns an empty solution.
        """
        self.env = environment
        self.verbose = verbose
        self.open_list =  []  # for fast membership checks
        self.closed_set = set()
        self.time_limit = time_limit if time_limit is not None and time_limit > 0 else float('inf')
        self.max_iterations = max_iterations if max_iterations is not None and max_iterations > 0 else float('inf')
        self.counter = count()
        self.total_time = 0
        self.total_iterations = 0
        
    def _root(self):
        """The root of the search tree: no constraints, every agent on its own best path. None when some
        agent has no path at all."""
        root = HighLevelNode()
        root.constraint_dict = {agent: Constraints() for agent in self.env.agent_dict}
        self.env.constraint_dict = root.constraint_dict
        root.solution, root.solution_cost = self.env.compute_solution()
        if not root.solution:
            return None
        root.cost = sum(root.solution_cost.values())
        return root

    def _push_node(self, node):
        heapq.heappush(self.open_list, (node.cost, next(self.counter), node))

    def _out_of_budget(self, started, iterations) -> bool:
        """Whether the time limit or the node limit has been reached."""
        if time.time() - started > self.time_limit:
            reason = f"time limit of {self.time_limit} seconds exceeded"
        elif iterations >= self.max_iterations:
            reason = f"max iterations of {self.max_iterations} reached"
        else:
            return False
        if self.verbose:
            print(f"Search terminated: {reason}.")
        return True

    def _report(self, started, iterations, success) -> dict:
        """Record the run's totals and return them as the ``solution_info`` dict."""
        self.total_time = min(self.time_limit, time.time() - started)
        self.total_iterations = min(self.max_iterations, iterations)
        return {"runtime": self.total_time, "total_iterations": self.total_iterations, "success": success}

    def search(self):
        """Best-first search over the constraint tree. Returns ``(plan, solution_info)``; the plan is empty
        when none was found within the limits."""
        started, iterations, solution = time.time(), 1, {}
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
            conflict = self.env.get_conflicts(P.solution)
            if not conflict:
                if self.verbose:
                    print("solution found")
                solution = self.generate_plan(P.solution)
                break
            for branch in self.env.branches(conflict):
                child = self._child(P, branch, selective=len(branch) > 1)
                if child is not None:
                    self._push_node(child)
        return solution, self._report(started, iterations, bool(solution))

    def _replan_agent(self, node, agent) -> bool:
        """Replan ``agent`` under ``node.constraint_dict``; False when it has no plan."""
        self.env.constraint_dict = node.constraint_dict
        self.env.constraints = node.constraint_dict[agent]
        path, cost = self.env.a_star.search(agent)
        if not path:
            return False
        node.solution[agent], node.solution_cost[agent] = path, cost
        return True

    def _child(self, P, branch, selective: bool):
        """The child of ``P`` with ``branch`` (``{agent: Constraints to add}``) applied, or None when some
        agent has no plan. Only the constrained agents get new constraint sets; the rest are shared.
        ``selective``: replan just the agents whose path breaks their new constraints; otherwise every
        agent's plan is recomputed (``Environment.compute_solution``, as plain CBS always has)."""
        node = HighLevelNode()
        node.solution, node.solution_cost = P.solution.copy(), P.solution_cost.copy()
        node.constraint_dict = dict(P.constraint_dict)
        for agent, added in branch.items():
            node.constraint_dict[agent] = P.constraint_dict[agent].copy()
            node.constraint_dict[agent].add_constraint(added)
        self.env.constraint_dict = node.constraint_dict
        if selective:
            for agent in branch:
                if not self.env.path_allowed(agent, node.solution[agent], node.constraint_dict[agent]) \
                        and not self._replan_agent(node, agent):
                    return None
        else:
            node.solution, node.solution_cost = self.env.compute_solution()
            if not node.solution:
                return None
        node.cost = sum(node.solution_cost.values())
        return node

    def _get_state_key(self, node):
        """A hashable key for closed-set deduplication.

        - Point agents: every agent's path AND its constraints. Nodes with equal paths but different
          constraints have different children, so they are kept apart; dropping one would lose its
          subtree and with it the optimal solution.
        - Agents with bodies: the paths only. Their constraint fans are conservative, so those searches
          are not optimal to begin with, and many constraint sets lead to the same paths: keying on the
          paths alone prunes that redundancy and keeps the tree from blowing up at large radius."""
        if self.env.has_bodies:
            return tuple((agent, tuple((s.time, s.location.point, s.ori) for s in path))
                         for agent, path in sorted(node.solution.items()))
        return tuple(
            (agent, tuple((s.time, s.location.point, s.ori) for s in path),
             frozenset(con.vertex_constraints), frozenset(con.edge_constraints), frozenset(con.must_be.items()))
            for agent, path in sorted(node.solution.items())
            for con in (node.constraint_dict[agent],)
        )

    def generate_plan(self, solution):
        """The plan as ``{agent: [entry, ...]}``. An entry is ``t, x, y`` in 2D and ``t, x, y, z`` in 3D; agents
        with a heading also get ``yaw`` (2D) or ``yaw, pitch`` (3D), in degrees. Before the agent's first
        move the heading is that of the move (0 if it never moves)."""
        plan = {}
        for agent, path in solution.items():
            if len(path[0].location) not in (2, 3):
                raise ValueError(f"Invalid location dimension: {len(path[0].location)}")
            unset = 0.0 if len(path[0].location) == 2 else (0.0, 0.0)
            first = next((s.ori for s in path if _is_set(s.ori)), unset)     # shown while the heading is FREE
            path_dict_list = []
            for state in path:
                entry = {'t': state.time, **dict(zip('xyz', state.location.point))}
                if state.ori is not None:
                    ori = first if state.ori == FREE else state.ori
                    if isinstance(ori, tuple):
                        entry['yaw'], entry['pitch'] = ori
                    else:
                        entry['yaw'] = ori
                path_dict_list.append(entry)
            plan[agent] = path_dict_list
        return plan
