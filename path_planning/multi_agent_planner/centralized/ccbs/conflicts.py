"""Where the agents of a continuous-time plan collide, and what each may not do because of it.

Mixed into ``ccbs.Environment``. Bodies are whatever each agent's ``Agent`` says (discs, polygons, 3D
shapes, with clearance); the overlap tests are the map's exact ``agent_conflicts``.
"""
from itertools import combinations

import numpy as np

from path_planning.common.world.collision_checking.sweep.disc_sweep import _ball_pair_windows
from path_planning.multi_agent_planner.centralized.cbs.data_types import _heading_of
from path_planning.multi_agent_planner.centralized.ccbs.data_types import EPS, INF, Conflict, Constraint, Move

_UNKNOWN = object()      # not in the pair cache yet
_PAIR_A, _PAIR_B = np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)     # one pair: pieces 0 and 1


class ContinuousConflicts:
    """The collision part of ``Environment``. Uses its ``graph_map``, ``bodies``, ``xyz``, ``facing``,
    ``start_ori``, ``endpoints``, ``precision``, ``cache``, ``_moves`` and ``_pairs``."""

    STRICT_EPS = 1e-9        # as the map's collision tests: bodies closer than their reach less this collide

    # ------------------------------------------------------------------ a path as timed moves
    def moves_of(self, agent, path) -> list:
        """``path`` as ``Move``s: its moves and waits, then the stay at its last vertex for good. A body
        that is not a ball keeps, while waiting, the heading of its last move (before its first move: its
        start heading, given or 0). Remembered per path."""
        return self._moves_entry(agent, path)[0]

    def _moves_entry(self, agent, path):
        """``(moves_of(agent, path), the moves as arrays: start point, velocity, start and end time)``."""
        path = path if isinstance(path, tuple) else tuple(map(tuple, path))
        known = self._moves.get((agent, path))
        if known is not None:
            return known
        facing = None if self.bodies[agent].is_ball else self.start_ori[agent]
        moves = []
        for (u, t1), (v, t2) in zip(path, path[1:]):
            if u != v:
                moves.append(Move(u, v, t1, t2, None))
                if facing is not None:
                    facing = self.facing[(u, v)]
            elif t2 > t1 + EPS:
                moves.append(Move(u, u, t1, t2, facing))
        moves.append(Move(path[-1][0], path[-1][0], path[-1][1], INF, facing))
        # The moves as arrays for the broad phase: start point, velocity, start and end time.
        a, b = self.xyz[[m.a for m in moves]], self.xyz[[m.b for m in moves]]
        t1 = np.array([m.t1 for m in moves])
        t2 = np.array([m.t2 for m in moves])
        span = np.where(np.isfinite(t2), t2 - t1, 1.0)[:, None]
        velocity = np.where(a == b, 0.0, (b - a) / np.maximum(span, 1e-12))
        if len(self._moves) > 200_000:
            self._moves.clear()
        entry = self._moves[(agent, path)] = (moves, (a, velocity, t1, t2))
        return entry

    def _piece(self, agent, move: Move, t1=None):
        """``move`` as an ``agent_conflicts`` piece ``(body, path, t0, hold, heading)``, started at ``t1``
        instead of its own start time when given."""
        t0 = move.t1 if t1 is None else t1
        if move.is_wait:
            heading = None if move.facing is None else _heading_of(move.facing)
            return self.bodies[agent], self.xyz[[move.a]], t0, move.t2 - move.t1, heading
        return self.bodies[agent], self.xyz[[move.a, move.b]], t0, 0.0, None

    def _overlaps(self, pieces, pairs=None):
        """When the pieces' bodies overlap: ``(i, j, t1, t2)`` arrays, one row per window."""
        bodies, paths, t0, hold, heading = zip(*pieces)
        heading = None if all(h is None for h in heading) else list(heading)
        return self.graph_map.agent_conflicts(list(bodies), list(paths), t0=list(t0), goal_hold=list(hold),
                                              heading=heading, pairs=pairs, as_arrays=True)

    # ------------------------------------------------------------------ detection
    def find_conflicts(self, paths: dict, only=None) -> list:
        """The conflicts of a plan ``{agent: path}``: the earliest one of every pair of agents that collide
        (of every pair with agent ``only``, when given), earliest first (ties in agent order). Each pair
        of paths is tested once per environment and remembered."""
        names = [name for name in self.endpoints if name in paths]       # always in agent order
        paths = {name: path if isinstance(path, tuple) else tuple(map(tuple, path)) for name, path in paths.items()}
        if only is None:
            wanted = list(combinations(range(len(names)), 2))
        else:
            k0 = names.index(only)
            wanted = [(min(k0, k), max(k0, k)) for k in range(len(names)) if k != k0]
        cache, found, missing = self._pairs, {}, []
        for k1, k2 in wanted:
            hit = cache.get((names[k1], paths[names[k1]], names[k2], paths[names[k2]]), _UNKNOWN)
            if hit is _UNKNOWN:
                missing.append((k1, k2))
            elif hit is not None:
                found[(k1, k2)] = hit
        if missing:
            computed = self._first_contacts(names, paths, missing, None if only is None else names.index(only))
            if len(cache) > 500_000:
                cache.clear()
            for k1, k2 in missing:
                hit = computed.get((k1, k2))
                cache[(names[k1], paths[names[k1]], names[k2], paths[names[k2]])] = hit
                if hit is not None:
                    found[(k1, k2)] = hit
        return [Conflict(names[k1], names[k2], m1, m2, max(t, m1.t1, m2.t1))
                for (k1, k2), (t, m1, m2) in sorted(found.items(), key=lambda item: (item[1][0], item[0]))]

    def _first_contacts(self, names, paths, pairs, k0=None) -> dict:
        """``{(k1, k2): (first contact time, k1's move, k2's move)}`` for the agent pairs ``pairs`` (agent
        indices into ``names``, ``k1 < k2``) whose paths collide. ``k0``: every pair involves this agent."""
        involved = sorted({k for pair in pairs for k in pair})
        moves, arrays = [], []
        for k in involved:
            agent_moves, agent_arrays = self._moves_entry(names[k], paths[names[k]])
            moves.extend(agent_moves)
            arrays.append(agent_arrays)
        counts = [len(arr[0]) for arr in arrays]
        owner = np.repeat(involved, counts)
        start, velocity, t1, t2 = (np.concatenate(part) for part in zip(*arrays))
        reach = np.repeat([self.bodies[names[k]].reach for k in involved], counts)

        # Broad phase: moves of the pairs' agents that overlap in time, and whose bounding balls come
        # closer than their reaches (+ a margin) while they do. The exact test only sees those.
        if k0 is not None:                               # k0's moves against the others'
            rows, cols = np.flatnonzero(owner == k0), np.flatnonzero(owner != k0)
            i, j = np.nonzero(np.maximum(t1[rows, None], t1[None, cols]) < np.minimum(t2[rows, None], t2[None, cols]) - EPS)
            i, j = rows[i], cols[j]
        else:
            allowed = np.zeros((len(names), len(names)), dtype=bool)
            allowed[tuple(np.array(pairs).T)] = True
            i, j = np.nonzero(np.triu(allowed[owner[:, None], owner[None, :]], 1)
                              & (np.maximum(t1[:, None], t1[None, :]) < np.minimum(t2[:, None], t2[None, :]) - EPS))
        if not len(i):
            return {}
        lo, hi = np.maximum(t1[i], t1[j]), np.minimum(t2[i], t2[j])
        w = (start[j] + velocity[j] * (lo - t1[j])[:, None]) - (start[i] + velocity[i] * (lo - t1[i])[:, None])
        v = velocity[j] - velocity[i]
        vv = np.einsum("ij,ij->i", v, v)
        moving = vv > 1e-18
        at = np.zeros(len(i))
        at[moving] = np.clip(-np.einsum("ij,ij->i", w[moving], v[moving]) / vv[moving], 0.0, (hi - lo)[moving])
        gap = w + v * at[:, None]
        close = np.einsum("ij,ij->i", gap, gap) < (reach[i] + reach[j] + 1e-6) ** 2
        i, j = i[close], j[close]
        if not len(i):
            return {}
        first = {}

        def keep(p, q, t):
            if owner[p] > owner[q]:
                p, q = q, p
            key = (int(owner[p]), int(owner[q]))
            if key not in first or t < first[key][0]:
                first[key] = (t, moves[p], moves[q])

        # Two balls: the overlap window in closed form, with the map's own disc test (each radius less
        # STRICT_EPS / 2, and only a window longer than 1e-12 counts, so touching is not contact).
        ball = np.array([self.bodies[name].is_ball for name in names])
        balls = ball[owner[i]] & ball[owner[j]]
        if balls.any():
            bi, bj = i[balls], j[balls]
            hit, contact, _ = _ball_pair_windows(bi, bj, t1, t2, start, velocity, reach - 0.5 * self.STRICT_EPS)
            for p, q, t in zip(bi[hit].tolist(), bj[hit].tolist(), contact[hit].tolist()):
                keep(p, q, t)
            i, j = i[~balls], j[~balls]
            if not len(i):
                return first
        pairs = sorted({(min(p, q), max(p, q)) for p, q in zip(i.tolist(), j.tolist())})
        used = sorted({k for pair in pairs for k in pair})       # only the moves in a candidate pair are tested
        slot = {k: n for n, k in enumerate(used)}
        pi, pj, c1, _ = self._overlaps([self._piece(names[owner[k]], moves[k]) for k in used],
                                       [(slot[p], slot[q]) for p, q in pairs])
        for p, q, t in zip(np.asarray(pi, dtype=int).tolist(), np.asarray(pj, dtype=int).tolist(), np.asarray(c1).tolist()):
            keep(used[p], used[q], t)
        return first

    # ------------------------------------------------------------------ constraints
    def constraint_for(self, agent, move: Move, other, other_move: Move, t: float) -> Constraint:
        """What ``agent`` may not do because its ``move`` collides (from time ``t``) with ``other``'s
        ``other_move``, following the reference implementation:

        - waiting: it may not be at its vertex, facing as it does, while ``other``'s body covers it there;
        - ``other`` is stopped for good: it may not start the move from its start time on;
        - else: it may not start the move from its start time until the first time it could start it
          without meeting ``other_move`` (found by bisection to ``precision``)."""
        mine, theirs = self.bodies[agent], self.bodies[other]
        if move.is_wait:
            key = ("wait", mine.key, move.a, move.facing, theirs.key, other_move.a, other_move.b, other_move.facing,
                   round(other_move.t2 - other_move.t1, 9))

            def windows():                                      # times from the start of other_move
                held = Move(move.a, move.a, 0.0, other_move.t2 - other_move.t1, move.facing)
                _, _, w1, w2 = self._overlaps([self._piece(agent, held), self._piece(other, other_move, 0.0)])
                return sorted(zip(np.asarray(w1).tolist(), np.asarray(w2).tolist()))

            at = t - other_move.t1
            hits = [w for w in self.cache.get_or_compute(key, windows) if w[0] - 1e-6 <= at <= w[1] + 1e-6]
            lo, hi = hits[0] if hits else (at, at)
            lo, hi = lo + other_move.t1, hi + other_move.t1
            # Worked out for the heading it waits with, so it bans only that heading (None for a ball).
            return Constraint(agent, move.a, move.a, min(lo, t), max(hi, t + self.precision), facing=move.facing)
        if other_move.t2 == INF:
            return Constraint(agent, move.a, move.b, move.t1, INF)
        ahead = other_move.t1 - move.t1                         # how long after this move the other starts
        key = ("move", mine.key, move.a, move.b, theirs.key, other_move.a, other_move.b, other_move.facing,
               round(ahead, 9), round(other_move.t2 - other_move.t1, 9))

        def first_safe_start():                                 # as a delay of this move
            fixed = self._piece(other, other_move, ahead)
            balls = mine.is_ball and theirs.is_ball
            lo, hi = 0.0, other_move.t2 - move.t1               # it collides now; once the other is done it cannot
            while hi - lo > self.precision:
                mid = 0.5 * (lo + hi)
                if balls:
                    hit = self._balls_meet(agent, move, mid, other, other_move, ahead)
                else:
                    hit = len(self._overlaps([self._piece(agent, move, mid), fixed])[0]) > 0
                lo, hi = (mid, hi) if hit else (lo, mid)
            return hi

        delay = max(self.cache.get_or_compute(key, first_safe_start), self.precision)
        return Constraint(agent, move.a, move.b, move.t1, move.t1 + delay)

    def _balls_meet(self, agent, m1: Move, s1: float, other, m2: Move, s2: float) -> bool:
        """Whether the balls of ``agent`` and ``other`` overlap when ``m1`` starts at ``s1`` and ``m2`` at
        ``s2`` (each keeping its duration; a ball is gone once its move ends), by the map's disc test."""
        x0 = self.xyz[[m1.a, m2.a]]
        span = np.array([m1.t2 - m1.t1, m2.t2 - m2.t1])
        velocity = np.where(span[:, None] > 0.0, (self.xyz[[m1.b, m2.b]] - x0) / np.maximum(span, 1e-300)[:, None], 0.0)
        rad = np.array([self.bodies[agent].reach, self.bodies[other].reach]) - 0.5 * self.STRICT_EPS
        w0 = np.array([s1, s2])
        return bool(_ball_pair_windows(_PAIR_A, _PAIR_B, w0, w0 + span, x0, velocity, rad)[0][0])
