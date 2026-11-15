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
from math import fabs, atan2, floor, ceil, inf, degrees, radians, cos, sin, hypot
from itertools import count,combinations
from collections import namedtuple
from functools import lru_cache, partial
import numpy as np
from path_planning.agent import Agent
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.world.collision_checking.sweep.polytope_sweep import _travel_frames
from path_planning.multi_agent_planner.data_type import HEURISTIC_TYPE
from path_planning.multi_agent_planner.sweep_cache import shared_sweep_cache

class Location(object):
    def __init__(self, point:tuple = None):
        self.point = tuple(point) if point is not None else None
    def __hash__(self):
        return hash(self.point)
    def __eq__(self, other):
        return self.point == other.point
    def __str__(self):
        return str(self.point)
    def __len__(self):
        return len(self.point)
    def __getitem__(self, index):
        return self.point[index]
    def __add__(self, other):
        return Location(tuple(self.point + other.point))

#: ``State.ori`` of an agent that has a heading but has not taken one yet (it may leave in any direction).
FREE = "free"


def _is_set(ori) -> bool:
    """Whether ``ori`` is an actual heading (not None: no heading; not ``FREE``: none taken yet)."""
    return ori is not None and ori != FREE


def _make_ori(yaw, pitch=None):
    """A ``State.ori`` from degrees: the yaw in ``[0, 360)``, or ``(yaw, pitch)`` in 3D. Every heading is
    built here, so headings from edges and from the agents' start values compare equal."""
    yaw = round(float(yaw) % 360.0, 6) % 360.0
    return yaw if pitch is None else (yaw, round(float(pitch), 6))


@lru_cache(maxsize=1 << 18)
def _ori_between(a, b):
    """The ``State.ori`` of an agent facing along ``a -> b`` (two points)."""
    d = np.subtract(b, a).astype(float)
    flat = hypot(d[0], d[1])
    return _make_ori(degrees(atan2(d[1], d[0])) if flat > 1e-12 else 0.0,
                     None if len(d) == 2 else degrees(atan2(d[2], flat)))


@lru_cache(maxsize=1 << 14)
def _heading_of(ori):
    """A ``State.ori`` as the fixed heading the collision queries take: radians in 2D; in 3D a rotation whose
    first axis is the travel direction (shared: do not modify)."""
    if not isinstance(ori, tuple):
        return radians(ori)
    yaw, pitch = radians(ori[0]), radians(ori[1])
    return _travel_frames(np.array([[cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch)]]))[0]


@lru_cache(maxsize=None)
def _bounding_ball(body: Agent) -> Agent:
    """``body``'s bounding ball: what it sweeps while turning in place."""
    return body if body.is_ball else Agent(body.radius, None, None, body.type)


#: One step of an agent's plan: ``agent`` (its body) goes ``a -> b`` (equal: it stays). ``body`` is what it
#: sweeps meanwhile (its bounding ball while ``turning`` in place) at ``heading`` (None: a move faces along
#: itself; a ball has none). ``oriented``: the agent plans its heading.
Step = namedtuple("Step", "agent body a b heading turning oriented")


class State(object):
    """``ori``: the agent's heading, in degrees in 2D (``[0, 360)``, counterclockwise from +x) and
    ``(yaw, pitch)`` degrees in 3D; ``FREE`` before it is set; None for an agent without a heading."""
    def __init__(self, time, location, ori=None):
        self.time = time
        self.location = location
        self.ori = ori
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location and self.ori == other.ori
    def __hash__(self):
        return hash((self.time, self.location.point, self.ori))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location) if self.ori is None else (self.time, self.location, self.ori))

class Conflict(object):
    VERTEX = 1
    EDGE = 2
    SWEEP = 3
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()
        # What each agent may not do at ``time``, as (fan 1, no turn 1, fan 2, no turn 2): a fan is the set
        # of destinations forbidden from the agent's location; "no turn" forbids turning in place there.
        # A SWEEP conflict holds a function instead, called when the fans are first read.
        self.forbidden = (set(), False, set(), False)

    def _forbidden(self, k):
        if callable(self.forbidden):
            self.forbidden = self.forbidden()
        return self.forbidden[k]

    location_1_f = property(lambda self: self._forbidden(0))
    no_turn_1 = property(lambda self: self._forbidden(1))
    location_2_f = property(lambda self: self._forbidden(2))
    no_turn_2 = property(lambda self: self._forbidden(3))

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2)  + ')'

class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash((self.time, self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    """Forbids the step ``location_1 -> location_2`` at ``time``; with equal locations, staying there
    (waiting or turning). ``turn``: forbids only turning in place at ``location_1``."""
    def __init__(self, time, location_1, location_2, turn=False):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
        self.turn = turn
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2 and self.turn == other.turn
    def __hash__(self):
        return hash((self.time, self.location_1, self.location_2, self.turn))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + (', turn' if self.turn else '') + ')'

class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class Environment(object):
    """CBS on a roadmap with unit time steps: every move and every wait takes 1 time unit.

    Agent bodies: ``agents[i]["agent"]`` (an ``Agent`` of any shape, 2D or 3D), else ``agent`` (one type for
    all), else a disc / sphere of ``radius``; without any of them the agents are points. Only an ``Agent``'s
    shape is used: its speed and turning rate, and ``velocity``, do not apply to unit steps.

    - Heading (``use_orientation``; default: agents whose body is not a ball): the heading is part of the
      state. An agent moves only along the edge it faces, and turning in place to another edge's direction
      takes one step, during which it is checked as its bounding ball. ``agents[i]["yaw"]`` (and
      ``"pitch"`` in 3D, default 0) is the start heading in degrees; without it the agent leaves in any
      direction. The plan gets ``yaw`` (and ``pitch`` in 3D) per step.
    - Conflicts of agents with bodies: one ``graph_map.agent_conflicts`` call over the whole plan (when and
      where two bodies overlap).
    - Constraints: per conflict, one sweep of each agent's move at that step; the other agent may not
      leave its vertex along a roadmap edge that move covers (``use_collision_sweep``; off: only the
      move it made is forbidden). ``clearance``: ``"radius"`` or ``"minkowski"`` as in
      ``GraphSampler.sweep_paths``."""

    def __init__(self, graph_map:GraphSampler, agents, astar_max_iterations=10000, radius = 0.0, velocity = 0.0, use_collision_sweep=True,heuristic_type: str = 'dijkstra', agent: "Agent | None" = None, clearance: str = "radius", use_orientation: "bool | None" = None):
        self.graph_map = graph_map
        self.use_orientation = use_orientation
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
        self.use_collision_sweep = use_collision_sweep
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

    # ------------------------------------------------------------ agents with bodies
    @staticmethod
    def _unit_step(agent):
        """``agent``'s shape as a unit-step type (no speed, instant turns); None stays None."""
        return None if agent is None else Agent(agent.shape, None, None, agent.type)

    def agent_type(self, name) -> "Agent | None":
        """The body of agent ``name`` (None: a point agent)."""
        return self.agent_types.get(name, self._default_agent)

    def has_orientation(self, name) -> bool:
        """Whether agent ``name`` plans its heading: ``use_orientation``, else any body that is not a ball."""
        body = self.agent_type(name)
        if self.use_orientation is not None:
            return bool(self.use_orientation) and body is not None
        return body is not None and not body.is_ball

    def _headings(self, name, points, oris):
        """The attitude of agent ``name``'s body at each state of its plan, for the collision queries: its
        heading ``ori`` when it has one; else that of its last move, or of its first move before it has
        moved. None for a ball, and for a body that never moves (it faces +x)."""
        if self.agent_type(name).is_ball:
            return [None] * len(points)
        moves = [_heading_of(_ori_between(points[k - 1], points[k])) if points[k] != points[k - 1] else None
                 for k in range(1, len(points))]
        last = next((h for h in moves if h is not None), None)
        heads = []
        for k, ori in enumerate(oris):
            if k and moves[k - 1] is not None:
                last = moves[k - 1]
            heads.append(_heading_of(ori) if _is_set(ori) else last)
        return heads

    def _step(self, name, points, oris, heads, t) -> Step:
        """Step ``t`` of agent ``name``'s plan (past its end the agent stays at its last state)."""
        k, k1 = min(t, len(points) - 1), min(t + 1, len(points) - 1)
        agent, a, b = self.agent_type(name), points[k], points[k1]
        turning = a == b and oris[k] != oris[k1]
        return Step(agent, _bounding_ball(agent) if turning else agent, a, b,
                    heads[k] if a == b and not turning else None, turning, oris[k] is not None)

    def _plan_pieces(self, names, plans, oris, heads):
        """The plan as timed pieces for ``agent_conflicts``: one 2-point path per move (``t0`` = its step),
        one 1-point path per run of waits (held for the run; forever at the end of the plan) and one
        1-point bounding-ball piece per turn in place. Returns ``(owner index, agents, paths, t0,
        goal_hold, heading)`` tuples, one entry per piece."""
        pieces = []
        for i, (name, points, ori, head) in enumerate(zip(names, plans, oris, heads)):
            body, n, t = self.agent_type(name), len(points), 0
            while t < n:
                k = t                                            # the agent stays as it is over [t, k]
                while k + 1 < n and points[k + 1] == points[t] and ori[k + 1] == ori[t]:
                    k += 1
                last = k == n - 1
                if k > t or last:
                    pieces.append((i, body, [points[t]], t, inf if last else float(k - t), head[t]))
                if not last and points[k + 1] == points[k]:      # then turns in place for one step
                    pieces.append((i, _bounding_ball(body), [points[k]], k, 1.0, None))
                elif not last:                                   # or moves to the next vertex
                    pieces.append((i, body, [points[k], points[k + 1]], k, 0.0, None))
                t = k + 1
        owner, agents, paths, t0, hold, heading = zip(*pieces)
        return owner, agents, [np.array(p, dtype=float) for p in paths], [float(t) for t in t0], hold, heading

    def _covered(self, step: Step, other: Agent):
        """What ``other`` may not use during ``step``: ``({source node index: {target points}}, {covered
        node indices})`` from one roadmap sweep, cached for every CBS environment on this map."""
        body, a, b, heading = step.body, step.a, step.b, step.heading
        key = (body.key, a, b, None if heading is None else np.asarray(heading, dtype=float).tobytes(),
               other.key, self.clearance)

        def compute():
            wait = a == b
            (vid, _, _), (src, tgt, _, _) = self.graph_map.sweep_paths(
                body, [np.array([a] if wait else [a, b], dtype=float)], goal_hold=1.0 if wait else 0.0,
                other=other, clearance=self.clearance, heading=None if heading is None else [heading])[0]
            nodes, edges = self.graph_map.nodes, {}
            for u, v in zip(src.tolist(), tgt.tolist()):
                edges.setdefault(u, set()).add(tuple(nodes[v].current))
            return edges, set(vid.tolist())

        return shared_sweep_cache(self.graph_map, "cbs").get_or_compute(key, compute)

    def _fan(self, mine: Step, other: Step):
        """What an agent may not do from its vertex during its step ``mine``, as ``(fan, no_turn)``: the
        destinations it may not take (its vertex itself: it may not stay) and whether it may not turn there.

        - Its own step is always forbidden: a move, a wait (then turning collides too) or a turn.
        - ``use_collision_sweep``: also every roadmap edge from its vertex that the ``other`` agent's step
          covers. When the vertex itself is covered, an agent without a heading may not stay; one with a
          heading may not turn (its bounding ball would be hit) but may still wait."""
        fan, no_turn = (set() if mine.turning else {Location(mine.b)}), mine.turning
        src = self.graph_map.pos_index_dict.get(mine.a)
        if self.use_collision_sweep and src is not None:
            edges, vertices = self._covered(other, mine.agent)
            fan.update(Location(p) for p in edges.get(int(src), ()))
            if int(src) in vertices:
                if mine.oriented:
                    no_turn = True
                else:
                    fan.add(Location(mine.a))
        return fan, no_turn

    def _fans(self, step_1: Step, step_2: Step):
        """Both agents' ``_fan``, as ``Conflict.forbidden``."""
        return (*self._fan(step_1, step_2), *self._fan(step_2, step_1))

    def _body_conflicts(self, solution, max_t, get_first_conflict, max_conflicts):
        """``get_conflicts`` for agents with bodies. One ``agent_conflicts`` call gives every time window in
        which two bodies overlap; each step ``t`` such a window touches is one ``SWEEP`` conflict of that
        pair, ordered by step then pair. A conflict's constraint fans are computed when first read."""
        names = list(solution)
        plans = [[s.location.point for s in solution[name]] for name in names]
        oris = [[getattr(s, "ori", None) for s in solution[name]] for name in names]   # states of other planners have none
        heads = [self._headings(*plan) for plan in zip(names, plans, oris)]
        owner, agents, paths, t0, hold, heading = self._plan_pieces(names, plans, oris, heads)
        i, j, t1, t2 = self.graph_map.agent_conflicts(
            agents, paths, t0=t0, goal_hold=hold, as_arrays=True,
            heading=None if all(h is None for h in heading) else list(heading))
        found = set()
        for a, b, w1, w2 in zip(i.tolist(), j.tolist(), t1.tolist(), t2.tolist()):
            a, b = sorted((owner[a], owner[b]))
            if a != b:                                           # the steps the window [w1, w2] touches
                first = floor(w1 + 1e-9)
                last = max_t if w2 == inf else min(max_t, ceil(w2 - 1e-9))
                found.update((t, a, b) for t in range(first, min(max(last, first + 1), max_t)))
        conflicts = []
        for t, a, b in ([min(found)] if get_first_conflict and found else sorted(found)):
            step_1, step_2 = (self._step(names[k], plans[k], oris[k], heads[k], t) for k in (a, b))
            c = Conflict()
            c.time, c.type, c.agent_1, c.agent_2 = t, Conflict.SWEEP, names[a], names[b]
            c.location_1, c.location_2 = Location(step_1.a), Location(step_2.a)
            c.forbidden = partial(self._fans, step_1, step_2)
            if get_first_conflict:
                return c
            conflicts.append(c)
            if max_conflicts is not None and len(conflicts) >= max_conflicts:
                break
        return conflicts

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
            for loc in conflict.location_1_f:
                e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, loc)
                constraint1.edge_constraints |= {e_constraint1}
            for loc in conflict.location_2_f:
                e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, loc)
                constraint2.edge_constraints |= {e_constraint2}
            if conflict.no_turn_1:
                constraint1.edge_constraints.add(EdgeConstraint(conflict.time, conflict.location_1, conflict.location_1, True))
            if conflict.no_turn_2:
                constraint2.edge_constraints.add(EdgeConstraint(conflict.time, conflict.location_2, conflict.location_2, True))
            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        return  VertexConstraint(state.time,state.location) not in self.constraints.vertex_constraints

    def transition_valid(self, state_1, state_2):
        banned = self.constraints.edge_constraints
        if not banned:
            return True
        if EdgeConstraint(state_1.time, state_1.location, state_2.location) in banned:
            return False
        turning = state_1.location == state_2.location and state_1.ori != state_2.ori
        return not (turning and EdgeConstraint(state_1.time, state_1.location, state_1.location, True) in banned)

    def is_solution(self, agent_name):
        pass

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
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

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
                len(self.constraints.vertex_constraints) == 0
                and len(self.constraints.edge_constraints) == 0
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

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
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
        heapq.heappush(self.open_list, (start.cost, next(self.counter), start))
        while self.open_list :
            iterations += 1
            if self.time_limit is not None and (time.time() - st) > self.time_limit:
                if self.verbose:
                    print(f"Search terminated: time limit of {self.time_limit} seconds exceeded.")
                break

            if self.max_iterations is not None and iterations >= self.max_iterations:
                if self.verbose:
                    print(f"Search terminated: max iterations of {self.max_iterations} reached.")                
                break

            _, _, P = heapq.heappop(self.open_list)

            
            if P is None:
                break
            state_key = self._get_state_key(P)
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_conflicts(P.solution)
            if not conflict_dict:
                if self.verbose:
                    print("solution found")
                success = True
                solution = self.generate_plan(P.solution)
                break
            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)
            for agent in constraint_dict.keys():
                new_node = HighLevelNode()
                new_node.solution = P.solution.copy()
                new_node.solution_cost = P.solution_cost.copy()

                # Selective deep copy only for affected agent's constraints
                new_node.constraint_dict = {}
                for a in self.env.agent_dict.keys():
                    if a == agent:
                        # Deep copy only the modified agent's constraints
                        new_constraints = Constraints()
                        new_constraints.vertex_constraints = P.constraint_dict[a].vertex_constraints.copy()
                        new_constraints.edge_constraints = P.constraint_dict[a].edge_constraints.copy()
                        new_constraints.add_constraint(constraint_dict[agent])
                        new_node.constraint_dict[a] = new_constraints
                    else:
                        # Share unchanged constraints
                        new_node.constraint_dict[a] = P.constraint_dict[a]

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution, new_node.solution_cost = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = sum(new_node.solution_cost.values())
                heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))
        
        self.total_time = min(self.time_limit, time.time() - st) 
        self.total_iterations = min(self.max_iterations, iterations)
        solution_info["runtime"] = self.total_time
        solution_info["total_iterations"] = self.total_iterations
        solution_info["success"] = success
        return solution,solution_info

    def _get_state_key(self, node):
        """Generate a hashable state key for closed-set deduplication.

        Keyed on the *solution* (every agent's full path), NOT the constraint
        set. Constraints uniquely determine a solution, but the converse is
        false: an "inactive" constraint (one the agent's optimal path already
        satisfies) leaves the solution unchanged, so two CBS nodes reached via
        different constraints can share an identical solution. Those are pure
        redundancy — same solution -> same conflicts -> same children — and
        must be deduped by solution. A constraint-based key fails to catch
        them and the search tree blows up combinatorially at large radius.
        """
        return tuple(
            (agent, tuple((s.time, s.location.point, s.ori) for s in path))
            for agent, path in sorted(node.solution.items())
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

