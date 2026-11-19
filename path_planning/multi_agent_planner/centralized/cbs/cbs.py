"""
Modified Conflict-based search for multi-agent path planning
original author: Ashwin Bose (@atb033)
description: This file implements the Conflict-based search algorithm for multi-agent path planning. Modified from the original implementation to work with the new common environment.
"""

import heapq
import time
from collections import defaultdict
from itertools import combinations, count

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.multi_agent_planner.centralized.cbs.a_star import AStar
from path_planning.multi_agent_planner.centralized.cbs.bodies import BodyConflicts
from path_planning.multi_agent_planner.centralized.cbs.data_types import (   # noqa: F401  (the types are re-exported)
    Conflict, Constraints, EdgeConstraint, HighLevelNode, Location, State, VertexConstraint, _turn_toward, put_heading, start_heading)
from path_planning.multi_agent_planner.centralized.cbs.routing import RoadmapCosts
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE

class Environment(BodyConflicts, RoadmapCosts):
    """CBS on a roadmap with unit time steps: every move and every wait takes 1 time unit.

    ``agents`` is a list of dicts: ``name``, ``start``, ``goal`` and optionally ``agent`` (an ``Agent``: the
    body) and ``yaw`` / ``pitch`` (a start heading). Everything about an agent's body comes from its
    ``Agent``. When no dict has one the agents are point particles: no body, no heading, no clearance.
    Bodies and point particles cannot be mixed.

    - Size: the ``Agent``'s shape (2D or 3D). Clearance: the ``Agent``'s ``clearance``, a buffer that
      expands its body radially, so two agents stay at least the sum of their clearances apart (0: they
      may touch). The moves a conflict forbids are found by growing one body by the other's bounding
      radius and clearance, which is exact for discs / spheres and a safe margin for polygons.
    - Timing: every move and every wait takes 1 time unit, for points and bodies alike; an ``Agent``'s
      ``max_speed`` is not used.
    - Turning: an ``Agent`` with a finite ``max_rotation`` has to turn when it changes direction, whatever
      its shape (discs and spheres too). With ``max_rotation`` None or infinite it turns instantly: it may
      leave along any edge and no step is spent turning (a polygon still faces along each move for the
      collision checks, and keeps its last heading while it waits).
    - The plan shows every body's heading as ``yaw`` (and ``pitch`` in 3D), whichever way it turns: for
      instant turns, the direction of the move it last made. Point particles have none.
    - Heading (agents that have to turn): the heading is part of the state. ``max_rotation`` is in radians
      per time step: turning in place through an angle ``a`` takes ``ceil(a / max_rotation)`` steps, each
      at the same position with the heading ``max_rotation`` closer (15 degrees per step: 6 steps for a
      90 degree turn). The agent may stop part-way and wait. An agent moves only along the edge it faces, and turning in place to another edge's direction
      takes one step, during which it is checked as its bounding ball. The plan gets ``yaw`` (and ``pitch``
      in 3D) per step.
    - Start heading (every body): ``agents[i]["yaw"]`` (and ``"pitch"`` in 3D, default 0) in degrees when
      given, else 0 (``(0, 0)`` in 3D). An agent that has to turn turns from it before its first move.
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
    - ``conflict_span`` (agents with bodies): ``"window"`` (default): one conflict per overlap of two
      bodies; a branch forbids its agent the colliding moves at every step of the overlap at once.
      ``"step"``: each time step of an overlap is its own conflict, so an overlap lasting k steps can take
      k branchings. Window needs far fewer nodes when two agents stay in each other's way, and solved
      more of the benchmark instances at no extra cost. Its one drawback: a plan in which each agent
      keeps a different step of the overlap is not reachable from either branch.
    """

    def __init__(self, graph_map: GraphSampler, agents, astar_max_iterations=10000, heuristic_type: str = 'dijkstra',
                 move_parked_agents: bool = True, conflict_span: str = "window", goal_conflicts: bool = False):
        if conflict_span not in ("step", "window"):
            raise ValueError(f"conflict_span must be 'step' or 'window', got {conflict_span!r}")
        self.graph_map = graph_map
        self.agents = agents
        self.move_parked_agents = move_parked_agents
        self.conflict_span = conflict_span
        self.goal_conflicts = goal_conflicts
        self.heuristic_type = HEURISTIC_TYPE[heuristic_type if heuristic_type in HEURISTIC_TYPE else "dijkstra"]

        # The agents' bodies and how they turn (BodyConflicts): each agent dict's "agent" (an Agent). Without
        # any, the agents are point particles.
        bodies = {a["name"]: a.get("agent") for a in agents}
        self.has_bodies = any(body is not None for body in bodies.values())
        if self.has_bodies and None in bodies.values():
            raise ValueError('CBS: every agent dict needs an "agent" (an Agent) once any of them has one')
        self.agent_types = {name: self._unit_step(body) for name, body in bodies.items()}
        self._turn_rates = {name: self._turn_rate_of(body) for name, body in bodies.items()}   # degrees per step; None: instant
        if self.has_bodies:
            self.graph_map.set_collision_sweep()
        self._start_ori = {}                 # agent with a body -> its start heading (make_agent_dict)
        self._regions = {}                   # (parked step, other body) -> (vertices, edges) it keeps out of

        # Roadmap distances (RoadmapCosts).
        self._moves_cache = {}               # point -> [(neighbour point, its Location, ori facing it)]
        self._links = None                   # directed roadmap edges by node index (_roadmap_links)
        self._dijkstra_cache = {}            # (start index, goal index) -> (states, cost)
        self._goal_dist_maps = {}            # goal index -> {node: distance to the goal}
        self._turn_dist_maps = {}            # (goal index, turn rate) -> {(node, heading): cost to the goal}
        self._heuristic_cache = {}           # (point, agent[, heading]) -> estimate
        self._lengths = {}                   # (point, point) -> roadmap edge length

        # The agent being planned: its constraints, and whose turning rate get_neighbors uses.
        self.constraints = Constraints()
        self.constraint_dict = {}
        self.current_agent = None
        self.agent_dict = {}
        self.make_agent_dict()
        self.a_star = AStar(self, astar_max_iterations)

    def get_neighbors(self, state):
        """The states one step after ``state`` for ``current_agent``: wait; move along a roadmap edge (an
        agent with a heading only along the edge it faces); turn in place towards
        another edge, by at most its turning rate (``turn_rate``) per step."""
        t, ori = state.time + 1, state.ori
        steps = [State(t, state.location, ori)]
        turns = {}                                               # an ordered set of the headings one turn step away
        rate = self.turn_rate(self.current_agent) if ori is not None else None
        for _, location, facing in self._moves_from(state.location.point):
            if ori is None:
                steps.append(State(t, location))
            elif ori == facing:
                steps.append(State(t, location, facing))
            else:
                turns.setdefault(_turn_toward(ori, facing, rate))
        steps += [State(t, state.location, heading) for heading in turns]
        return [n for n in steps if self.state_valid(n) and self.transition_valid(state, n)]

    def get_conflicts(self, solution, get_first_conflict: bool = True, max_conflicts: int | None = None):
        """The conflicts of ``solution``, earliest first: the first one (``get_first_conflict``; a false
        value when there is none) or the list of them, cut off at ``max_conflicts`` for callers that only
        compare the count against a bound."""
        max_t = max(len(plan) for plan in solution.values())
        if self.has_bodies:
            return self._body_conflicts(solution, max_t, get_first_conflict, max_conflicts)

        conflicts = []
        names = list(solution.keys())
        # Each agent's location per step up to max_t, holding its last one once its plan has ended.
        at = [[s.location for s in path] + [path[-1].location] * (max_t + 1 - len(path)) for path in solution.values()]
        for t in range(max_t):
            # Hash each agent by vertex and by move, then visit the clashing pairs in pair order (i < j).
            by_vertex, by_move = defaultdict(list), defaultdict(list)
            for i, row in enumerate(at):
                by_vertex[row[t]].append(i)
                if row[t] != row[t + 1]:
                    by_move[(row[t], row[t + 1])].append(i)
            pairs = [(i, j) for group in by_vertex.values() if len(group) > 1 for i, j in combinations(group, 2)]
            pairs += [tuple(sorted((i, j))) for (here, nxt), group in by_move.items() if (nxt, here) in by_move
                      for i in group for j in by_move[(nxt, here)]]
            for i, j in sorted(set(pairs)):
                here_1, next_1 = at[i][t], at[i][t + 1]
                if here_1 == at[j][t]:                           # both at one vertex
                    c = Conflict(Conflict.VERTEX, t, names[i], names[j], here_1)
                else:                                            # swapping along an edge
                    c = Conflict(Conflict.EDGE, t, names[i], names[j], here_1, next_1)
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
        """What ``conflict`` forbids each of its two agents: ``{agent: Constraints}``."""
        first, second = Constraints(), Constraints()
        t = conflict.time
        if conflict.type == Conflict.VERTEX:                     # neither may be there then
            first.vertex_constraints.add(VertexConstraint(t, conflict.location_1))
            second.vertex_constraints.add(VertexConstraint(t, conflict.location_1))
        elif conflict.type == Conflict.EDGE:                     # neither may make its half of the swap
            first.edge_constraints.add(EdgeConstraint(t, conflict.location_1, conflict.location_2))
            second.edge_constraints.add(EdgeConstraint(t, conflict.location_2, conflict.location_1))
        elif conflict.type == Conflict.SWEEP:                    # each agent's fan, at every step of the overlap
            for c in [conflict, *conflict.later]:
                first.edge_constraints.update(EdgeConstraint(c.time, c.location_1, loc) for loc in c.location_1_f)
                second.edge_constraints.update(EdgeConstraint(c.time, c.location_2, loc) for loc in c.location_2_f)
                if c.no_turn_1:
                    first.edge_constraints.add(EdgeConstraint(c.time, c.location_1, c.location_1, True))
                if c.no_turn_2:
                    second.edge_constraints.add(EdgeConstraint(c.time, c.location_2, c.location_2, True))
        else:
            return {}
        return {conflict.agent_1: first, conflict.agent_2: second}

    def get_state(self, agent_name, solution, t):
        """The agent's state at step ``t``: its last one once its plan has ended."""
        path = solution[agent_name]
        return path[t] if t < len(path) else path[-1]

    def state_valid(self, state):
        con = self.constraints
        if con.deadline is not None and state.time > con.deadline:
            return False
        for start, vertices, _ in con.regions:
            if state.time >= start and state.location in vertices:
                return False
        return not con.vertex_constraints or (state.time, state.location.point) not in con.bans()[0]

    def transition_valid(self, state_1, state_2):
        for start, _, edges in self.constraints.regions:
            if state_1.time >= start and (state_1.location, state_2.location) in edges:
                return False
        if not self.constraints.edge_constraints:
            return True
        banned = self.constraints.bans()[1]
        p1, p2 = state_1.location.point, state_2.location.point
        if (state_1.time, p1, p2, False) in banned:
            return False
        turning = p1 == p2 and state_1.ori != state_2.ori
        return not (turning and (state_1.time, p1, p1, True) in banned)

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
        self._start_ori = {}
        for agent in self.agents:
            name = agent['name']
            if self.agent_type(name) is not None:            # a body starts with the given heading, else 0
                self._start_ori[name] = start_heading(agent)
            # An agent that has to turn carries its heading in its state.
            ori = self._start_ori[name] if self.has_orientation(name) else None
            self.agent_dict[name] = {'start': State(t, Location(agent['start']), ori), 'goal': State(t, Location(agent['goal']))}

    def compute_solution(self):
        """Every agent's best path under its constraints in ``constraint_dict``: ``(paths, costs)`` by agent,
        or ``(False, inf)`` when one has none. An agent without constraints or turn steps takes its plain
        shortest path; the others run the time-expanded search."""
        solution, solution_cost = {}, {}
        for agent, info in self.agent_dict.items():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            if self.constraints.is_empty() and info["start"].ori is None:
                path, cost = self._static_shortest_path_states(agent)
            else:
                path, cost = self.a_star.search(agent)
            if not path:
                return False, float('inf')
            solution[agent], solution_cost[agent] = path, cost
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
        self.open_list = []                  # heap of (cost, push order, node)
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
            if self._expand(P):
                if self.verbose:
                    print("solution found")
                solution = self.generate_plan(P.solution)
                break
        return solution, self._report(started, iterations, bool(solution))

    def _expand(self, P) -> bool:
        """Branch ``P`` on its first conflict, pushing its children. True when ``P`` has no conflict: it is
        the solution."""
        conflict = self.env.get_conflicts(P.solution)
        if not conflict:
            return True
        for branch in self.env.branches(conflict):
            child = self._child(P, branch, selective=len(branch) > 1)
            if child is not None:
                self._push_node(child)
        return False

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
        constrained agent is replanned. (The others keep their paths: their constraints, and so their
        searches, are unchanged.)"""
        node = HighLevelNode()
        node.solution, node.solution_cost = P.solution.copy(), P.solution_cost.copy()
        node.constraint_dict = self._with_constraints(P, branch)
        self.env.constraint_dict = node.constraint_dict
        for agent in branch:
            if selective and self.env.path_allowed(agent, node.solution[agent], node.constraint_dict[agent]):
                continue
            if not self._replan_agent(node, agent):
                return None
        node.cost = sum(node.solution_cost.values())
        return node

    @staticmethod
    def _with_constraints(P, branch) -> dict:
        """``P``'s constraints with ``branch`` (``{agent: Constraints to add}``) joined in; the other agents'
        sets are shared, not copied."""
        constraints = dict(P.constraint_dict)
        for agent, added in branch.items():
            constraints[agent] = P.constraint_dict[agent].copy()
            constraints[agent].add_constraint(added)
        return constraints

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
        """The plan as ``{agent: [entry, ...]}``. An entry is ``t, x, y`` in 2D and ``t, x, y, z`` in 3D; every
        agent with a body also gets its heading in degrees, ``yaw`` (2D) or ``yaw, pitch`` (3D): see
        ``Environment.plan_oris``."""
        plan = {}
        for agent, path in solution.items():
            if len(path[0].location) not in (2, 3):
                raise ValueError(f"Invalid location dimension: {len(path[0].location)}")
            oris = self.env.plan_oris(agent, [s.location.point for s in path], [s.ori for s in path])
            plan[agent] = [put_heading({'t': state.time, **dict(zip('xyz', state.location.point))}, ori)
                           for state, ori in zip(path, oris)]
        return plan
