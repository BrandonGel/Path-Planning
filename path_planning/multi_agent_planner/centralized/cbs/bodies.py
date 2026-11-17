"""CBS for agents with bodies (any ``Agent`` shape, 2D or 3D): which body each agent has, where two bodies
overlap along a plan, and which moves a conflict forbids. Mixed into ``cbs.Environment``."""
from functools import partial
from math import ceil, degrees, floor, inf, isinf

import numpy as np

from path_planning.agent import Agent
from path_planning.multi_agent_planner.centralized.cbs.data_types import (
    Conflict, Location, Step, _bounding_ball, _heading_of, _ori_between)
from path_planning.multi_agent_planner.sweep_cache import shared_sweep_cache


class BodyConflicts:
    """The body-aware part of ``Environment``. Uses its ``graph_map``, ``agent_types``, ``_turn_rates``,
    ``_start_ori``, ``conflict_span`` and ``_regions``."""

    #: How one body is kept clear of another in the roadmap sweeps (``GraphSampler.sweep_paths``): grown
    #: by the other's bounding radius.
    CLEARANCE = "radius"

    @staticmethod
    def _unit_step(agent):
        """``agent``'s shape and clearance as the type used in the collision queries: no speed (every step
        takes 1 time unit) and no turning time (a turn is its own step here). None stays None."""
        return None if agent is None else Agent(agent.shape, None, None, agent.type, agent.clearance)

    def agent_type(self, name) -> "Agent | None":
        """The body of agent ``name`` (None: a point agent)."""
        return self.agent_types.get(name)

    @staticmethod
    def _turn_rate_of(agent):
        """How far ``agent`` turns in one time step, in degrees: its ``max_rotation`` (radians per time
        unit). None when it turns instantly (``max_rotation`` None or infinite) or is a point particle."""
        if agent is None or agent.max_rotation is None or isinf(agent.max_rotation):
            return None
        if agent.max_rotation == 0:
            raise ValueError("CBS: max_rotation = 0 (a body that never turns) is not supported; "
                             "use None for instant turns or a positive rate")
        return degrees(agent.max_rotation)

    def turn_rate(self, name):
        """Degrees agent ``name`` turns per step, or None when it turns instantly (or has no body)."""
        return self._turn_rates.get(name)

    def has_orientation(self, name) -> bool:
        """Whether agent ``name`` plans its heading: it has to turn to change direction (a finite
        ``max_rotation``). It then moves only along the edge it faces, and turning in place through an
        angle ``a`` takes ``ceil(a / max_rotation)`` steps."""
        return self.turn_rate(name) is not None

    def plan_oris(self, name, points, oris):
        """The heading of agent ``name`` at each state of its plan, as ``State.ori`` values (degrees; all
        None for a point particle).

        - An agent that has to turn: its planned heading.
        - One that turns instantly: the direction of the move it last made.
        - Before its first move: its start heading (``agents[i]["yaw"]`` when given, else 0)."""
        if self.agent_type(name) is None:
            return [None] * len(points)
        moves = [_ori_between(points[k - 1], points[k]) if points[k] != points[k - 1] else None
                 for k in range(1, len(points))]
        last = self._start_ori[name]
        shown = []
        for k, ori in enumerate(oris):
            if ori is not None:
                last = ori
            elif k and moves[k - 1] is not None:                      # an instant turn, on leaving
                last = moves[k - 1]
            shown.append(last)
        return shown

    def _headings(self, name, points, oris):
        """The attitude of agent ``name``'s body at each state of its plan, for the collision queries
        (``plan_oris`` as fixed headings). None for a ball: its heading does not change its shape."""
        if self.agent_type(name).is_ball:
            return [None] * len(points)
        return [_heading_of(ori) for ori in self.plan_oris(name, points, oris)]

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
        key = (body.key, a, b, None if heading is None else np.asarray(heading, dtype=float).tobytes(), other.key)

        def compute():
            wait = a == b
            (vid, _, _), (src, tgt, _, _) = self.graph_map.sweep_paths(
                body, [np.array([a] if wait else [a, b], dtype=float)], goal_hold=1.0 if wait else 0.0,
                other=other, clearance=self.CLEARANCE, heading=None if heading is None else [heading])[0]
            nodes, edges = self.graph_map.nodes, {}
            for u, v in zip(src.tolist(), tgt.tolist()):
                edges.setdefault(u, set()).add(tuple(nodes[v].current))
            return edges, set(vid.tolist())

        return shared_sweep_cache(self.graph_map, "cbs").get_or_compute(key, compute)

    def _fan(self, mine: Step, other: Step):
        """What an agent may not do from its vertex during its step ``mine``, as ``(fan, no_turn)``: the
        destinations it may not take (its vertex itself: it may not stay) and whether it may not turn there.

        - Its own step is always forbidden: a move, a wait (then turning collides too) or a turn.
        - So is every roadmap edge from its vertex that the ``other`` agent's step covers. When the vertex
          itself is covered, an agent without a heading may not stay; one with a heading may not turn (its
          bounding ball would be hit) but may still wait."""
        fan, no_turn = (set() if mine.turning else {Location(mine.b)}), mine.turning
        src = self.graph_map.pos_index_dict.get(mine.a)
        if src is not None:
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
        oris = [[s.ori for s in solution[name]] for name in names]
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
        def conflict_at(t, a, b):
            step_1, step_2 = (self._step(names[k], plans[k], oris[k], heads[k], t) for k in (a, b))
            c = Conflict(Conflict.SWEEP, t, names[a], names[b], Location(step_1.a), Location(step_2.a))
            c.forbidden = partial(self._fans, step_1, step_2)
            stopped = [t >= len(plans[k]) - 1 for k in (a, b)]
            if stopped[0] != stopped[1]:                         # exactly one of them is at its goal for good
                c.parked = (names[a], step_1) if stopped[0] else (names[b], step_2)
            return c

        if self.conflict_span == "window":                       # one entry per run of consecutive steps
            runs = {}
            for t, a, b in sorted(found):
                run = runs.setdefault((a, b), [])
                if run and run[-1][-1] == t - 1:
                    run[-1].append(t)
                else:
                    run.append([t])
            spans = sorted((steps[0], a, b, steps[1:]) for (a, b), run in runs.items() for steps in run)
        else:
            spans = [(t, a, b, ()) for t, a, b in sorted(found)]
        conflicts = []
        for t, a, b, later in ([min(spans)] if get_first_conflict and spans else spans):
            c = conflict_at(t, a, b)
            c.later = [conflict_at(u, a, b) for u in later]
            if get_first_conflict:
                return c
            conflicts.append(c)
            if max_conflicts is not None and len(conflicts) >= max_conflicts:
                break
        return conflicts

    def _region(self, step: Step, other: Agent):
        """Where ``other`` may not be while the body of ``step`` (an agent stopped at its goal) stays there:
        ``(vertices, directed edges)`` as frozensets of ``Location`` s, from the cached roadmap sweep."""
        key = (step.body.key, step.a, None if step.heading is None else np.asarray(step.heading, dtype=float).tobytes(),
               other.key)
        region = self._regions.get(key)
        if region is None:
            edges, vertices = self._covered(step, other)
            nodes = self.graph_map.nodes
            region = self._regions[key] = (
                frozenset(Location(tuple(nodes[v].current)) for v in vertices),
                frozenset((Location(tuple(nodes[u].current)), Location(q)) for u, targets in edges.items() for q in targets))
        return region
