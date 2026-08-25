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
   :func:`mapf_solver.summarize_solution`.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List


def _point_to_dict(t: int, pt) -> dict:
    if len(pt) == 3:
        return {"t": t, "x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}
    return {"t": t, "x": float(pt[0]), "y": float(pt[1])}


class Simulation:
    def __init__(
        self,
        tasks: Iterable[dict],
        agents: List[dict],
        delay_probability: float = 0.0,
        rng: random.Random | None = None,
        velocity: float = 0.0,
        timestep_duration: float = 1.0,
    ):
        self.tasks = list(tasks)
        self.agents = agents
        self.delay_probability = float(delay_probability)
        self.rng = rng or random.Random()

        # Constant-velocity motion: each tick an agent travels at most this much arc
        # length (world units) along its committed fine-waypoint polyline, passing
        # through multiple waypoints. 0 -> fall back to one waypoint per tick.
        self.velocity = float(velocity)
        self.timestep_duration = float(timestep_duration)
        self.arc_budget = self.velocity * self.timestep_duration

        self.time = 0
        self.delayed_agents: set = set()
        self.actual_paths: Dict[str, List[dict]] = {}
        self.times_agent_delayed: defaultdict = defaultdict(int)
        self.algo_time = 0.0

        for agent in self.agents:
            self.actual_paths[agent["name"]] = [_point_to_dict(0, agent["start"])]

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

        position = {
            agent["name"]: tuple(
                v for k, v in self.actual_paths[agent["name"]][-1].items() if k != "t"
            )
            for agent in self.agents
        }
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

        for agent in agents_to_move:
            name = agent["name"]
            cur = position[name]
            agents_pos_now[name] = cur
            planned = token["agents"][name]
            if len(planned) <= 1 or self.rng.random() < self.delay_probability:
                agent_plan[name] = (0, [], None)
                agent_pos_next[name] = cur
            else:
                # Advance up to one tick's worth of arc length (multiple fine waypoints).
                n, traversed, fracs = self._advance_plan(planned, self.arc_budget)
                agent_plan[name] = (n, traversed, fracs)
                agent_pos_next[name] = traversed[-1] if traversed else cur

        # Resolve vertex / edge collisions by holding the colliding agents. Compares
        # end-of-tick positions; a held agent gets zero motion this tick. (SIPP already
        # plans collision-free against other committed paths; this is a backstop.)
        collision = True
        delayed = set()
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
                for wp, frac in zip(traversed, fracs):
                    self.actual_paths[name].append(_point_to_dict((self.time - 1) + frac, wp))
            else:
                self.actual_paths[name].append(_point_to_dict(self.time, agent_pos_next[name]))

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
