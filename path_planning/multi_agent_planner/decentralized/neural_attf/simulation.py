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
    ):
        self.tasks = list(tasks)
        self.agents = agents
        self.delay_probability = float(delay_probability)
        self.rng = rng or random.Random()

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
        agent_pos_next: Dict[str, tuple] = {}

        for agent in agents_to_move:
            name = agent["name"]
            cur = position[name]
            agents_pos_now[name] = cur
            planned = token["agents"][name]
            if len(planned) <= 1:
                agent_pos_next[name] = cur
            elif self.rng.random() < self.delay_probability:
                agent_pos_next[name] = cur
            else:
                agent_pos_next[name] = tuple(planned[1])

        # Resolve vertex / edge collisions by holding the colliding agents.
        collision = True
        delayed = set()
        while collision:
            collision = False
            collision_set = set()
            for i in range(len(agents_to_move)):
                for j in range(i + 1, len(agents_to_move)):
                    a = agents_to_move[i]["name"]
                    b = agents_to_move[j]["name"]
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

        for agent in agents_to_move:
            name = agent["name"]
            if name in delayed:
                self.delayed_agents.add(name)
            elif len(token["agents"][name]) > 1:
                token["agents"][name] = token["agents"][name][1:]
            self.actual_paths[name].append(_point_to_dict(self.time, agent_pos_next[name]))
