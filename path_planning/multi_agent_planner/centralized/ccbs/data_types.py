"""The types CCBS works with: timed moves, conflicts, constraints and the nodes of its search tree.

Vertices are roadmap node indices. A path is a list of ``(vertex, time)``: the agent is at ``vertex`` at
``time``; two entries in a row with the same vertex are a wait. After the last entry the agent stays.
"""
from bisect import insort
from collections import namedtuple

INF = float("inf")
EPS = 1e-9          # time tolerance in comparisons


class Move(namedtuple("Move", "a b t1 t2 facing")):
    """What an agent does over ``[t1, t2]``: goes ``a -> b``, or waits at ``a`` (``a == b``; ``t2 = inf``
    for the stay after its plan ends). ``facing``: the heading a waiting body keeps (None for a move,
    which faces along itself, and for bodies whose heading does not matter)."""
    __slots__ = ()

    @property
    def is_wait(self) -> bool:
        return self.a == self.b


class Constraint(namedtuple("Constraint", "agent a b t1 t2 positive facing", defaults=(False, None))):
    """``agent`` may not start the move ``a -> b`` during ``[t1, t2)``; with ``a == b`` it may not be at
    ``a`` then. ``positive``: it has to start the move in that window instead (a landmark). ``facing``
    (waits only): the ban holds only while the agent faces that way at ``a`` (a ``Move.facing``); None:
    whatever its heading (balls, which have none)."""
    __slots__ = ()

    @property
    def is_wait(self) -> bool:
        return self.a == self.b



class Conflict:
    """Two agents' bodies overlap from time ``t``, while they make ``move_1`` and ``move_2``. ``overcost``:
    for a cardinal conflict, the least either agent's cost rises when it has to give way."""
    __slots__ = ("agent_1", "agent_2", "move_1", "move_2", "t", "overcost", "replans")

    def __init__(self, agent_1, agent_2, move_1: Move, move_2: Move, t: float):
        self.agent_1, self.agent_2, self.move_1, self.move_2, self.t = agent_1, agent_2, move_1, move_2, t
        self.overcost = 0.0
        self.replans = None         # {agent: (constraint, path, cost, constraints it was planned under)}

    def __repr__(self):
        return f"Conflict({self.agent_1}, {self.agent_2}, t={self.t:.4g}, overcost={self.overcost:.4g})"


def merge_interval(intervals: list, t1: float, t2: float) -> None:
    """Add ``[t1, t2)`` to a sorted list of disjoint intervals, joining the ones it meets."""
    insort(intervals, (t1, t2))
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    intervals[:] = merged


class Constraints:
    """One agent's constraints, as the low-level search reads them:

    - ``unsafe[v]``: when it may not be at vertex ``v`` whatever its heading (sorted, disjoint);
    - ``unsafe_facing[v][facing]``: when it may not be at ``v`` facing that way (sorted, disjoint);
    - ``windows[a][b]``: when it may not start the move ``a -> b`` (sorted, disjoint);
    - ``landmarks``: the moves it has to make, as ``(t1, t2, a, b)`` ordered by ``t1``."""

    def __init__(self, constraints=()):
        self.unsafe, self.unsafe_facing, self.windows, self.landmarks = {}, {}, {}, []
        for c in constraints:
            if c.positive:
                insort(self.landmarks, (c.t1, c.t2, c.a, c.b))
            elif c.is_wait and c.facing is not None:
                merge_interval(self.unsafe_facing.setdefault(c.a, {}).setdefault(c.facing, []), c.t1, c.t2)
            elif c.is_wait:
                merge_interval(self.unsafe.setdefault(c.a, []), c.t1, c.t2)
            else:
                merge_interval(self.windows.setdefault(c.a, {}).setdefault(c.b, []), c.t1, c.t2)

    def safe_intervals(self, v, facing=None) -> list:
        """When the agent may be at ``v`` (facing ``facing``, when given): the gaps between its unsafe
        intervals over ``[0, inf)``."""
        unsafe = self.unsafe.get(v, ())
        if facing is not None and v in self.unsafe_facing:
            headed = self.unsafe_facing[v].get(facing)
            if headed:
                unsafe = sorted(list(unsafe) + headed)
        safe, start = [], 0.0
        for lo, hi in unsafe:
            if lo > start + EPS:
                safe.append((start, lo))
            start = max(start, hi)
        if start < INF:
            safe.append((start, INF))
        return safe


class CTNode:
    """A node of the constraint tree. It stores only what changed from its parent: the replanned paths
    (the root: every agent's), the constraint added and, with disjoint splitting, the positive one. Paths,
    costs and constraints of the whole node are read back along the parent chain."""
    __slots__ = ("parent", "paths", "costs", "constraint", "positive", "cost", "h", "total_cons", "id",
                 "cardinal", "semi_cardinal", "non_cardinal")

    def __init__(self, parent=None, constraint=None, positive=None):
        self.parent, self.constraint, self.positive = parent, constraint, positive
        self.paths, self.costs = {}, {}
        self.cost, self.h, self.id = 0.0, 0.0, 0
        self.total_cons = 0 if parent is None else parent.total_cons + 1 + (positive is not None)
        self.cardinal, self.semi_cardinal, self.non_cardinal = [], [], []

    def chain(self):
        node = self
        while node is not None:
            yield node
            node = node.parent

    def cost_of(self, agent) -> float:
        return next(node.costs[agent] for node in self.chain() if agent in node.costs)

    def all_paths(self) -> dict:
        """Every agent's path at this node, in the root's (agent) order."""
        latest = {}
        for node in self.chain():
            for agent, path in node.paths.items():
                latest.setdefault(agent, path)
        root = node                                          # (the loop ends at the root)
        return {agent: latest[agent] for agent in root.paths}

    def constraints_of(self, agent) -> list:
        """Every constraint on ``agent`` from the root down to this node (negative and positive)."""
        found = []
        for node in self.chain():
            for c in (node.constraint, node.positive):
                if c is not None and c.agent == agent:
                    found.append(c)
        return found

    @property
    def conflicts(self) -> list:
        return self.cardinal + self.semi_cardinal + self.non_cardinal
