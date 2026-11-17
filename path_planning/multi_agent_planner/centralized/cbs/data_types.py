"""The data types of CBS: locations, states, headings, conflicts, constraints and search-tree nodes."""
from collections import namedtuple
from functools import lru_cache
from math import acos, atan2, ceil, cos, degrees, hypot, radians, sin

import numpy as np

from path_planning.agent import Agent
from path_planning.common.world.collision_checking.sweep.polytope_sweep import _travel_frames


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
def _make_ori(yaw, pitch=None):
    """A ``State.ori`` from degrees: the yaw in ``[0, 360)``, or ``(yaw, pitch)`` in 3D. Every heading is
    built here, so headings from edges and from the agents' start values compare equal."""
    yaw = round(float(yaw) % 360.0, 6) % 360.0
    return yaw if pitch is None else (yaw, round(float(pitch), 6))


def start_heading(agent: dict):
    """An agent dict's start heading as a ``State.ori``: its ``yaw`` (and ``pitch`` in 3D, default 0) in
    degrees when given, else 0 (``(0, 0)`` in 3D)."""
    three_d = len(agent["start"]) == 3
    yaw = agent.get("yaw")
    if yaw is None:
        return _make_ori(0.0, 0.0 if three_d else None)
    return _make_ori(yaw, (agent.get("pitch") or 0.0) if three_d else None)


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
    return _travel_frames(_direction(ori)[None])[0]


def _direction(ori):
    """The unit vector of a 3D ``State.ori`` ``(yaw, pitch)``."""
    yaw, pitch = radians(ori[0]), radians(ori[1])
    return np.array([cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch)])


@lru_cache(maxsize=1 << 18)
def _turn_toward(ori, target, rate: float):
    """The heading after one turn step from ``ori`` towards ``target`` (both ``State.ori``), turning at most
    ``rate`` degrees the short way round: ``target`` itself once it is within reach. A turn through an
    angle ``a`` so takes ``ceil(a / rate)`` steps."""
    if not isinstance(ori, tuple):                               # 2D: a signed angle
        diff = (target - ori + 180.0) % 360.0 - 180.0
        if diff == -180.0:
            diff = 180.0
        return target if abs(diff) <= rate + 1e-6 else _make_ori(ori + (rate if diff > 0 else -rate))
    u, v = _direction(ori), _direction(target)                   # 3D: along the great arc between them
    angle = degrees(acos(max(-1.0, min(1.0, float(u @ v)))))
    if angle <= rate + 1e-6:
        return target
    if angle > 180.0 - 1e-6:                                     # opposite: any way round; turn about the z or x axis
        axis = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        v = np.cross(axis, u) / np.linalg.norm(np.cross(axis, u))
        angle = 90.0
    s = rate / angle
    w = (sin(radians((1.0 - s) * angle)) * u + sin(radians(s * angle)) * v) / sin(radians(angle))
    flat = hypot(w[0], w[1])
    return _make_ori(degrees(atan2(w[1], w[0])) if flat > 1e-9 else 0.0, degrees(atan2(w[2], flat)))


@lru_cache(maxsize=1 << 18)
def _turn_steps(ori, target, rate: float) -> int:
    """How many ``_turn_toward`` steps take ``ori`` to ``target`` at ``rate`` degrees per step:
    ``ceil(angle / rate)``."""
    if ori == target:
        return 0
    if not isinstance(ori, tuple):
        angle = abs((target - ori + 180.0) % 360.0 - 180.0)
    else:
        angle = degrees(acos(max(-1.0, min(1.0, float(_direction(ori) @ _direction(target))))))
    return max(1, ceil((angle - 1e-6) / rate))


@lru_cache(maxsize=None)
def _bounding_ball(body: Agent) -> Agent:
    """``body``'s bounding ball: what it sweeps while turning in place."""
    return body if body.is_ball else Agent(body.radius, None, None, body.type, body.clearance)


#: One step of an agent's plan: ``agent`` (its body) goes ``a -> b`` (equal: it stays). ``body`` is what it
#: sweeps meanwhile (its bounding ball while ``turning`` in place) at ``heading`` (None: a move faces along
#: itself; a ball has none). ``oriented``: the agent plans its heading.
Step = namedtuple("Step", "agent body a b heading turning oriented")


class State(object):
    """``ori``: the agent's heading, in degrees in 2D (``[0, 360)``, counterclockwise from +x) and
    ``(yaw, pitch)`` degrees in 3D; None for an agent without a heading."""
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
    """Two agents in each other's way at step ``time``. ``VERTEX`` / ``EDGE`` (point agents): both at
    ``location_1``, or swapping along ``location_1 -> location_2``. ``SWEEP`` (bodies): their bodies
    overlap while each is at, or leaving, its own ``location_1`` / ``location_2``."""
    VERTEX = 1
    EDGE = 2
    SWEEP = 3

    def __init__(self, type=-1, time=-1, agent_1='', agent_2='', location_1=None, location_2=None):
        self.type = type
        self.time = time
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.location_1 = location_1 if location_1 is not None else Location()
        self.location_2 = location_2 if location_2 is not None else Location()
        # What each agent may not do at ``time``, as (fan 1, no turn 1, fan 2, no turn 2): a fan is the set
        # of destinations forbidden from the agent's location; "no turn" forbids turning in place there.
        # A SWEEP conflict holds a function instead, called when the fans are first read.
        self.forbidden = (set(), False, set(), False)
        # ``conflict_span="window"``: the later steps of the same overlap, one SWEEP conflict each. Their
        # constraints are added together with this one's, so one branching covers the whole overlap.
        self.later = []
        # (name, Step) of the agent already stopped at its goal when this SWEEP conflict happens, when
        # exactly one of the two is: a goal conflict (``Environment.branches``).
        self.parked = None

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
    """One agent's constraints: forbidden vertices and steps; (disjoint splitting, ICBS) ``must_be``:
    ``{time: Location}`` the agent has to be at; (goal conflicts) ``regions``: ``(from time, vertices,
    edges)`` it has to keep out of from that time on, and ``deadline``: the time by which its plan must
    have ended at its goal."""
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()
        self.must_be = {}
        self.regions = []
        self.deadline = None
        self._free = (None, {})              # (sizes when computed, {location: free_from})

    def free_from(self, location):
        """The first time the agent may stop at ``location`` for good: after the last constraint that
        forbids it being there or staying there, and not before its last ``must_be``. Cached until the
        constraints change."""
        sizes = (len(self.vertex_constraints), len(self.edge_constraints), len(self.must_be))
        if self._free[0] != sizes:
            self._free = (sizes, {})
        t = self._free[1].get(location)
        if t is None:
            times = [c.time + 1 for c in self.vertex_constraints if c.location == location]
            times += [c.time + 1 for c in self.edge_constraints
                      if not c.turn and c.location_1 == location and c.location_2 == location]
            t = self._free[1][location] = max(times + list(self.must_be), default=float("-inf"))
        return t

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints
        self.must_be.update(other.must_be)
        self.regions = self.regions + [r for r in other.regions if r not in self.regions]
        if other.deadline is not None:
            self.deadline = other.deadline if self.deadline is None else min(self.deadline, other.deadline)

    def copy(self) -> "Constraints":
        new = Constraints()
        new.vertex_constraints = self.vertex_constraints.copy()
        new.edge_constraints = self.edge_constraints.copy()
        new.must_be = self.must_be.copy()
        new.regions = list(self.regions)
        new.deadline = self.deadline
        return new

    def is_empty(self) -> bool:
        return not (self.vertex_constraints or self.edge_constraints or self.must_be or self.regions
                    or self.deadline is not None)

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])


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
