"""
Lightweight simulation harness for Neural-ATTF on arbitrary graphs.

Mirrors ``~/Documents/code/Neural_ATTF/Simulation/simulation.py`` but works on
point-tuple positions instead of integer grid cells, so it composes with the
generalized :class:`NeuralATTF` planner.

Per timestep:

1. Drives ``algorithm.time_forward(t, position, delayed_agents)``.
2. Reads each agent's planned next position from the token; optionally injects
   a random per-agent delay (``delay_probability``) that holds the agent in
   place.
3. Resolves vertex / edge collisions by repeatedly holding the colliding
   agents until no conflicts remain; collided agents are reported as
   ``delayed_agents`` on the next step.
4. Appends ``{'t', 'x', 'y'}`` (or ``..., 'z'``) entries to ``actual_paths``
   so the schedule is compatible with :class:`Visualizer2D.animate` and
   :func:`mapf_solver.summarize_solution`. Agents with a body also get ``yaw``
   (+ ``pitch`` in 3D, degrees): the direction of their last move.

Each agent's body and speed come from its dict's ``"agent"`` (an ``Agent``), as
for the planner.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from path_planning.multi_agent_planner.centralized.cbs.data_types import (
    _heading_of, _ori_between, put_heading, start_heading,
)


def _point_to_dict(t: int, pt, heading=None) -> dict:
    d = {"t": t, "x": float(pt[0]), "y": float(pt[1])}
    if len(pt) == 3:
        d["z"] = float(pt[2])
    return put_heading(d, heading)


def _position(entry: dict) -> tuple:
    """The position of an ``actual_paths`` entry."""
    return (entry["x"], entry["y"], entry["z"]) if "z" in entry else (entry["x"], entry["y"])


class Simulation:
    def __init__(
        self,
        tasks: Iterable[dict],
        agents: List[dict],
        delay_probability: float = 0.0,
        rng: random.Random | None = None,
        timestep_duration: float = 1.0,
    ):
        self.tasks = list(tasks)
        self.agents = agents
        self.delay_probability = float(delay_probability)
        self.rng = rng or random.Random()

        self.bodies = {a["name"]: a.get("agent") for a in agents}
        # Constant-speed motion: each tick an agent travels at most this much arc
        # length (world units, its ``max_speed`` x the tick) along its committed
        # fine-waypoint polyline, passing through multiple waypoints. 0 (no body or
        # no speed) -> one waypoint per tick.
        self.timestep_duration = float(timestep_duration)
        self.arc_budget = {
            name: (b.max_speed or 0.0) * self.timestep_duration if b is not None else 0.0
            for name, b in self.bodies.items()
        }
        # Bounding reach (radius + clearance) per agent, for the hold backstop (see time_forward).
        self.reach = {name: 0.0 if b is None else b.reach for name, b in self.bodies.items()}
        # Heading of each agent with a body: its start yaw, then the direction of its last move.
        self.heading = {a["name"]: start_heading(a) for a in agents if self.bodies[a["name"]] is not None}

        self.time = 0
        self.delayed_agents: set = set()
        self.actual_paths: Dict[str, List[dict]] = {}
        self.times_agent_delayed: defaultdict = defaultdict(int)
        self.algo_time = 0.0

        for agent in self.agents:
            self.actual_paths[agent["name"]] = [_point_to_dict(0, agent["start"], self.heading.get(agent["name"]))]

    # ---------------------------------------------------------- public API

    def get_time(self) -> int:
        return self.time

    def get_algo_time(self) -> float:
        return self.algo_time

    def get_actual_paths(self) -> Dict[str, List[dict]]:
        return self.actual_paths

    def get_new_tasks(self) -> List[dict]:
        return [t for t in self.tasks if int(t["start_time"]) == self.time]

    def get_delayed_agents(self) -> set:
        return self.delayed_agents

    # ---------------------------------------------------------- step

    def time_forward(self, algorithm) -> None:
        self.time += 1

        position = {agent["name"]: _position(self.actual_paths[agent["name"]][-1]) for agent in self.agents}
        delayed_in = list(self.delayed_agents)

        t0 = time.time()
        algorithm.time_forward(self.time, position, delayed_in)
        self.algo_time += time.time() - t0

        self.delayed_agents = set()
        agents_to_move = list(self.agents)
        self.rng.shuffle(agents_to_move)

        token = algorithm.get_token()
        agents_pos_now: Dict[str, tuple] = {}
        agent_plan: Dict[str, tuple] = {}      # name -> (n_consumed, traversed waypoints)
        agent_pos_next: Dict[str, tuple] = {}  # name -> end-of-tick position

        delayed = set()  # agents held this tick; reported to the planner next tick
        for agent in agents_to_move:
            name = agent["name"]
            cur = position[name]
            agents_pos_now[name] = cur
            planned = token["agents"][name]
            if len(planned) <= 1:
                agent_plan[name] = (0, [], None)
                agent_pos_next[name] = cur
            elif self.rng.random() < self.delay_probability:
                # Random hold. It MUST be reported: the committed path is time-sampled
                # (waypoint k == tick k), so from now on this agent would run one tick
                # behind everything the others planned against; the planner resets and
                # re-plans a delayed agent from where it actually stands.
                agent_plan[name] = (0, [], None)
                agent_pos_next[name] = cur
                delayed.add(name)
                self.times_agent_delayed[name] += 1
            else:
                # Advance up to one tick's worth of arc length (multiple fine waypoints).
                n, traversed, fracs = self._advance_plan(planned, self.arc_budget[name])
                agent_plan[name] = (n, traversed, fracs)
                agent_pos_next[name] = traversed[-1] if traversed else cur

        # Resolve vertex / edge collisions by holding the colliding agents. Compares
        # end-of-tick positions; a held agent gets zero motion this tick. (SIPP already
        # plans collision-free against other committed paths; this is a backstop.)
        collision = True
        while collision:
            collision = False
            collision_set = set()
            for i in range(len(agents_to_move)):
                for j in range(i + 1, len(agents_to_move)):
                    a = agents_to_move[i]["name"]
                    b = agents_to_move[j]["name"]
                    if a in delayed or b in delayed:
                        continue
                    if agent_pos_next[a] == agent_pos_next[b]:
                        collision = True
                        collision_set.add(a)
                        collision_set.add(b)
                    elif (
                        agents_pos_now[a] == agent_pos_next[b]
                        and agents_pos_now[b] == agent_pos_next[a]
                    ):
                        collision = True
                        collision_set.add(a)
                        collision_set.add(b)
            for name in collision_set:
                delayed.add(name)
                agent_pos_next[name] = agents_pos_now[name]
                agent_plan[name] = (0, [], None)

        # Body-aware backstop. Committed paths are mutually consistent, but a hold
        # (random delay, or one of the holds above) is applied AFTER everyone's motion
        # for this tick was planned on the assumption that the held agent moves on.
        # Anyone whose body, moving this tick, would touch an agent that is standing
        # still is therefore held too - and reported as delayed, so the planner
        # re-syncs its path next tick - instead of driving through a stopped body
        # (see :meth:`_hits_resting`). Consistent plans never do that, so this only
        # fires when a hold has already broken the plan. A mover that already touches
        # the resting body (it was held inside the footprint, or the two were parked
        # next to each other) is only held if its motion brings it CLOSER than it
        # starts - moving away is the escape the planner just computed for it.
        graph_map = getattr(algorithm, "graph_map", None)
        if any(r > 0.0 for r in self.reach.values()):
            names = [a["name"] for a in agents_to_move]
            changed = True
            while changed:
                changed = False
                stationary = [
                    n for n in names
                    if all(p == agents_pos_now[n] for p in agent_plan[n][1])
                ]
                for name in names:
                    traversed = agent_plan[name][1]
                    if name in delayed or not traversed or name in stationary:
                        continue
                    legs = [agents_pos_now[name]] + list(traversed)
                    for other in stationary:
                        if other == name:
                            continue
                        if self._hits_resting(graph_map, name, legs, other, agents_pos_now[other]):
                            delayed.add(name)
                            agent_pos_next[name] = agents_pos_now[name]
                            agent_plan[name] = (0, [], None)
                            changed = True
                            break

        for agent in agents_to_move:
            name = agent["name"]
            n, traversed, fracs = agent_plan[name]
            if name in delayed:
                self.delayed_agents.add(name)
            if n > 0:
                token["agents"][name] = token["agents"][name][n:]
            if traversed:
                # Record every fine waypoint traversed this tick with fractional
                # sub-stamps over (t-1, t]: the planner's own tick fractions when the
                # waypoint carries them (``via_t`` - waits inside the tick stay
                # waits), else proportional to arc length (constant speed within the
                # tick). The last point lands exactly on the integer tick.
                if fracs is None:
                    legs = [agents_pos_now[name]] + list(traversed)
                    seg = [math.dist(legs[m], legs[m + 1]) for m in range(len(legs) - 1)]
                    total = sum(seg)
                    fracs, acc = [], 0.0
                    for m in range(1, len(legs)):
                        acc += seg[m - 1]
                        fracs.append(acc / total if total > 0 else m / len(traversed))
                prev = agents_pos_now[name]
                for wp, frac in zip(traversed, fracs):
                    if name in self.heading and tuple(wp) != tuple(prev):
                        self.heading[name] = _ori_between(prev, wp)
                    prev = wp
                    self.actual_paths[name].append(
                        _point_to_dict((self.time - 1) + frac, wp, self.heading.get(name)))
            else:
                self.actual_paths[name].append(
                    _point_to_dict(self.time, agent_pos_next[name], self.heading.get(name)))

    def _hits_resting(self, graph_map, name, legs, other, rest) -> bool:
        """Whether ``name`` moving along ``legs`` this tick runs into ``other`` resting at ``rest``.
        Two balls (or no map to test on): their centres come closer than the two reaches. Otherwise the
        exact bodies (``graph_map.agent_conflicts``; the mover faces along each leg, the resting one its
        last heading). Either way, touching from the start counts only if the motion brings the centres
        closer than they start (moving away is an escape)."""
        clearance = self.reach[name] + self.reach[other] - 1e-9
        if clearance <= 0.0:
            return False
        d_min = self._polyline_point_dist(legs, rest)
        d_start = math.dist(legs[0], rest)
        if d_min >= clearance:
            return False                                  # not even the bounding balls meet
        mover, body = self.bodies[name], self.bodies[other]
        if not (mover.is_ball and body.is_ball) and graph_map is not None:
            pts = [legs[0]] + [q for p, q in zip(legs, legs[1:]) if tuple(q) != tuple(p)]
            hits = graph_map.agent_conflicts(
                [mover, body], [np.array(pts, dtype=float), np.array([rest], dtype=float)],
                goal_hold=[0.0, float("inf")],
                heading=[None, None if body.is_ball else _heading_of(self.heading[other])], first_only=True)
            if not hits:
                return False
            if next(iter(hits.values())) > 1e-9:
                return True                               # it would drive into the resting body
        return d_min < d_start - 1e-9                     # touching (from the start): only if closing in

    @staticmethod
    def _polyline_point_dist(legs, point) -> float:
        """Minimum Euclidean distance from ``point`` to the polyline ``legs``."""
        best = float("inf")
        px = point
        for a, b in zip(legs, legs[1:]):
            ab = [bb - aa for aa, bb in zip(a, b)]
            ap = [pp - aa for aa, pp in zip(a, px)]
            denom = sum(c * c for c in ab)
            t = 0.0 if denom <= 0.0 else max(0.0, min(1.0, sum(x * y for x, y in zip(ap, ab)) / denom))
            d = math.dist(px, [aa + t * c for aa, c in zip(a, ab)])
            if d < best:
                best = d
        if len(legs) == 1:
            best = math.dist(px, legs[0])
        return best

    @staticmethod
    def _step_points(planned, i):
        """Points visited going from ``planned[i]`` to ``planned[i + 1]``: any roadmap
        vertices the waypoint carries as ``via`` (see NeuralATTF._Waypoint), then the
        waypoint itself."""
        nxt = planned[i + 1]
        return [tuple(v) for v in getattr(nxt, "via", ())] + [tuple(nxt)]

    def _advance_plan(self, planned, budget):
        """Walk the committed waypoint list ``planned`` (``planned[0]`` == current pos)
        from the head, accumulating Euclidean arc length until the next waypoint would
        exceed ``budget``. Returns ``(n_consumed, traversed, fracs)``: the number of
        head entries to pop, the points actually visited this tick (excluding the
        current position; a waypoint's ``via`` vertices precede it) and their tick
        fractions, or ``None`` when the executor should stamp them by arc length.

        A time-sampled waypoint (one carrying ``via_t``, i.e. produced by the SIPP
        low level: "position at the end of this tick") is consumed exactly one per
        tick whatever the budget - its timing is the contract every other agent
        planned against, so running two of them in one tick (e.g. a short
        wait-then-move sample followed by a short arrival sample) would put the
        agent a tick ahead of where the others expect it. A wait (zero-length
        step) consumes exactly one entry and stops, preserving holds.
        ``budget <= 0`` falls back to one waypoint per tick.
        """
        traversed: List[tuple] = []
        used = 0.0
        i = 0
        nxt = planned[1] if len(planned) > 1 else None
        via_t = getattr(nxt, "via_t", None)
        if via_t is not None:
            traversed = self._step_points(planned, 0)
            return 1, traversed, list(via_t) + [1.0]
        while i + 1 < len(planned):
            step = self._step_points(planned, i)
            legs = [tuple(planned[i])] + step
            seg = sum(math.dist(legs[m], legs[m + 1]) for m in range(len(legs) - 1))
            if seg == 0.0:
                # Wait at the vertex: advance exactly one entry, then stop.
                if not traversed:
                    i += 1
                    traversed.append(tuple(planned[i]))
                break
            if budget > 0.0 and used + seg > budget and traversed:
                break  # next waypoint would overrun the arc-length budget
            used += seg
            i += 1
            traversed.extend(step)
            if budget <= 0.0 or used >= budget:
                break
        return i, traversed, None
