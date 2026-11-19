"""CCBS's low-level search: safe-interval path planning (SIPP) for one agent under its constraints.

It follows the reference implementation (PathPlanning/Continuous-CBS, ``sipp.cpp``):

- a state is a vertex and one of its safe intervals; waiting is implicit (arriving later at the next
  vertex means the agent waited at this one) and written out when the path is rebuilt;
- a move may not start inside one of its forbidden windows, has to start while the agent is still safe
  where it is, and has to arrive inside a safe interval of the next vertex;
- the agent finishes only in the goal's last safe interval, where it can stay for good;
- a wait constraint may hold for one heading only (a body that is not a ball): at such a vertex the
  safe intervals, and so the search states, are per heading the agent arrives with;
- landmarks (moves it has to make in a time window) split the search into legs.
"""
import heapq
import time
from itertools import count

from path_planning.multi_agent_planner.centralized.ccbs.data_types import EPS, INF, Constraints


_ALWAYS = ((0.0, INF),)                                 # the safe intervals of a vertex with no constraint


class _Node:
    """At ``v`` from time ``g``, in its safe interval ``[lo, hi]``, facing ``f`` (kept where it matters:
    at a vertex with heading constraints)."""
    __slots__ = ("v", "g", "lo", "hi", "parent", "f")

    def __init__(self, v, g, lo, hi, parent=None, f=None):
        self.v, self.g, self.lo, self.hi, self.parent, self.f = v, g, lo, hi, parent, f


class SIPP:
    """``find_path(agent, constraints)`` for the agents of ``env`` (a CCBS ``Environment``)."""

    def __init__(self, env, max_expansions: int = 200_000):
        self.env = env
        self.max_expansions = max_expansions
        self.expanded = 0                   # low-level expansions over all searches
        self.timed_out = False              # the last search stopped at its deadline or expansion cap
        self._found = {}                    # (agent, constraints) -> (path, cost): searches are deterministic

    def find_path(self, agent, constraints=(), stop_at=None):
        """The agent's quickest path under ``constraints`` (a list of ``Constraint``): ``(path, cost)``
        with ``path`` a tuple of ``(vertex, time)`` and ``cost`` its arrival time, or ``(None, inf)``.
        ``stop_at``: a ``time.time()`` deadline."""
        self.timed_out = False
        key = (agent, frozenset(constraints))
        known = self._found.get(key)
        if known is not None:                                   # the same search was made before
            return known
        result = self._find_path(agent, Constraints(constraints), stop_at)
        if not self.timed_out:                                  # (a search cut short may succeed later)
            if len(self._found) > 200_000:
                self._found.clear()
            self._found[key] = result
        return result

    def _find_path(self, agent, con, stop_at):
        env = self.env
        start, goal = env.endpoints[agent]
        facing = None if env.bodies[agent].is_ball else env.start_ori[agent]
        first = con.safe_intervals(start, facing)
        if not first or first[0][0] > EPS:                      # not allowed to be at its start at t = 0
            return None, INF
        starts = [_Node(start, 0.0, *first[0], None, facing)]
        # (The intervals without a heading are a superset of every heading's: the quick checks use them.)
        for t1, t2, a, b in con.landmarks:                      # one leg per landmark: reach a, then make the move
            open_ = any(max(lo, t1) < min(hi, t2) - EPS for lo, hi in con.safe_intervals(a))
            ends = self._search(agent, con, starts, a, (t1, t2), stop_at) if open_ else []
            starts = self._make_move(agent, con, ends, a, b, t2)
            if not starts:
                return None, INF
        last = con.safe_intervals(goal)
        if not last or last[-1][1] < INF:                       # it could never stay at its goal
            return None, INF
        ends = self._search(agent, con, starts, goal, None, stop_at)
        if not ends:
            return None, INF
        path = self._rebuild(agent, ends[0])
        return path, path[-1][1]

    # ------------------------------------------------------------------ one leg
    def _search(self, agent, con, starts, target, span, stop_at):
        """From any of ``starts``, the earliest ways to be at ``target``: for good (``span`` None: in a
        safe interval that never ends; one node), or inside ``span = (t1, t2)``: the earliest per safe
        interval of ``target`` (per heading too, where its intervals depend on it; the agent waits for
        the span to open when it may). Returns the nodes found."""
        env, tick, push, pop = self.env, count(), heapq.heappush, heapq.heappop
        succ, h = env.successors(agent), env.time_to(agent, target)
        unsafe, headed, facing = con.unsafe, con.unsafe_facing, env.facing    # (headed: vertices with heading bans)
        if span is None:
            t1, t2, wanted = 0.0, INF, 1
        else:
            t1, t2 = span
            wanted = INF if target in headed else sum(       # how many to find before stopping
                1 for lo, hi in con.safe_intervals(target) if max(lo, t1) < min(hi, t2) - EPS)

        def key_of(v, f, lo):                                   # the search state of a node
            if v in headed:
                return v, f, lo
            return v if lo == 0.0 else (v, lo)                  # (no tuple for a vertex's first interval)

        max_f, found, best, safe = t2, {}, {}, {}
        heap = []
        for node in starts:
            if h[node.v] < INF:
                best[key_of(node.v, node.f, node.lo)] = node.g
                push(heap, (node.g + h[node.v], -node.g, next(tick), node))
        while heap:
            node = pop(heap)[3]
            here, g0, leave_by = node.v, node.g, node.hi + EPS
            if best.get(key_of(here, node.f, node.lo), INF) < g0 - EPS:
                continue                                        # a quicker way here was found since
            if here == target:
                if span is None:
                    if node.hi == INF:
                        found[0] = node
                else:
                    k = (node.f, node.lo) if here in headed else node.lo
                    if k not in found and max(g0, t1) < min(node.hi, t2) - EPS:
                        found[k] = node if g0 >= t1 else _Node(here, t1, node.lo, node.hi, node, node.f)
                if len(found) >= wanted:
                    break
            self.expanded += 1
            if self.expanded % 256 == 0 and stop_at is not None and time.time() > stop_at:
                self.timed_out = True
                return []
            if len(best) > self.max_expansions:
                self.timed_out = True
                return []
            leaving = con.windows.get(here)             # forbidden departure windows from here
            for v, dur in succ[here]:
                h_v = h[v]
                if h_v == INF:
                    continue                                    # the target cannot be reached from there
                f_v = None
                if v in headed:                                 # it arrives facing along the move
                    f_v = facing[(here, v)]
                    intervals = safe.get((v, f_v))
                    if intervals is None:
                        intervals = safe[(v, f_v)] = con.safe_intervals(v, f_v)
                elif v in unsafe:
                    intervals = safe.get(v)
                    if intervals is None:
                        intervals = safe[v] = con.safe_intervals(v)
                else:
                    intervals = _ALWAYS
                forbidden = leaving.get(v, ()) if leaving else ()
                for lo, hi in intervals:
                    g = g0 + dur
                    if g > hi + EPS:
                        continue
                    if g < lo:
                        g = lo                                  # wait here until the next vertex is safe
                    for w1, w2 in forbidden:                    # and until the move may start
                        if w1 - EPS < g - dur < w2 - EPS:
                            g = w2 + dur
                    if g - dur > leave_by or g > hi + EPS or g + h_v > max_f:
                        continue                                # it would have to stay here too long
                    key = key_of(v, f_v, lo)
                    if best.get(key, INF) <= g + EPS:
                        continue
                    best[key] = g
                    push(heap, (g + h_v, -g, next(tick), _Node(v, g, lo, hi, node, f_v)))
        return sorted(found.values(), key=lambda n: n.g)

    def _make_move(self, agent, con, ends, a, b, t2):
        """A landmark: from the nodes ``ends`` (at ``a``, inside the landmark's window), start ``a -> b``
        before ``t2``. Returns the arrival nodes at ``b``, the earliest per safe interval of ``b``."""
        dur = self.env.duration(agent, self.env.length[(a, b)])
        f_b = self.env.facing[(a, b)] if b in con.unsafe_facing else None
        arrivals = {}
        for end in ends:
            for lo, hi in con.safe_intervals(b, f_b):
                leave = max(end.g, lo - dur)
                for w1, w2 in con.windows.get(a, {}).get(b, ()):
                    if w1 - EPS < leave < w2 - EPS:
                        leave = w2
                if leave > end.hi + EPS or leave >= t2 - EPS or leave + dur > hi + EPS:
                    continue
                if leave + dur < arrivals.get(lo, (INF,))[0]:
                    waited = end if leave <= end.g + EPS else _Node(a, leave, end.lo, end.hi, end, end.f)
                    arrivals[lo] = (leave + dur, _Node(b, leave + dur, lo, hi, waited, f_b))
        return [node for _, node in arrivals.values()]

    def _rebuild(self, agent, node):
        """The path that ends at ``node``, with each wait written out as a repeated vertex."""
        nodes = []
        while node is not None:
            nodes.append((node.v, node.g))
            node = node.parent
        nodes.reverse()
        path = [nodes[0]]
        for (u, g1), (v, g2) in zip(nodes, nodes[1:]):
            if u != v:
                leave = g2 - self.env.duration(agent, self.env.length[(u, v)])
                if leave > path[-1][1] + EPS:
                    path.append((u, leave))                     # waited at u before the move
                path.append((v, g2))
            elif g2 > path[-1][1] + EPS:
                path.append((v, g2))
        return tuple(path)
