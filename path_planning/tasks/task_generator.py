"""The fleet and its tasks from a file (``config/tasks.yaml``), with agent types from ``config/agents.yaml``.

::

    agents: {amr: 3, forklift: 1}       # how many of each type. Without agent types: `agents: 4`

    # or, assignments already made (a list instead of counts):
    # agents:
    #   - {type: amr, start: [5.5, 15.5], goal: [44.5, 19.5]}        # name optional (agent0, agent1, ...)
    #   - {type: forklift, goals: [[15.5, 12.5], [15.5, 9.5]]}       # start optional; a sequence of goals

    tasks:                              # a pool of tasks: name -> goal(s)
      pallet: {goals: [[15.5, 12.5], [15.5, 9.5]], types: [forklift]}   # only forklifts may take it
      tote:   [[15.5, 6.5], [35.5, 25.5]]                               # short form: the sequence of goals
      drop:   [45.5, 25.5]                                              # short form: one goal
    mode: singular        # singular | repeated | achieved | endless
    count: 3              # repeated: rounds of the whole pool; achieved: tasks to complete
    frequency: 0.1        # releases per time unit (left out: everything is released at t = 0)
    per_release: 1        # tasks created at each release
    order: sequential     # sequential | random: the order tasks are drawn from the pool

Coordinates are world coordinates, as in the map files.

- A type name must be one of the agent types given. Without agent types no entry may name a type, and
  the agents are point particles (unit moves, instant turning), as the planners take them today.
- ``mode``: ``singular``, the run is finished when every task was done once; ``repeated``, when every
  task was done ``count`` times (the same pool again); ``achieved``, when ``count`` tasks were done;
  ``endless``, never (tasks keep coming until the planner's horizon or time limit).
- Agents listed with goals are tasks too: each agent's goals become one task that only it may take.

A one-shot solver (``solve_mapf``) takes ``mapf_agents``: assignments with one goal each. The lifelong
planner (``solve_mapd``, Neural-ATTF) takes ``agents`` and the generator itself as its task stream
(``release`` / ``finished``).
"""
import math
import random

import yaml

from path_planning.agent.agent_types import load_agent_types
from path_planning.utils.util import points_to_roadmap_frame

MODES = ("singular", "repeated", "achieved", "endless")
ORDERS = ("sequential", "random")
KEYS = {"agents", "tasks", "mode", "count", "frequency", "per_release", "order"}
AGENT_KEYS = {"name", "type", "start", "goal", "goals", "yaw", "pitch"}
ONE_SHOT = "a one-shot solver plans one start to one goal per agent; use Neural-ATTF (solve_mapd) for this"


def _is_point(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) in (2, 3) and all(isinstance(v, (int, float)) for v in value)


def _goals(value, where) -> list:
    """A goal or a sequence of goals as a list of points."""
    if _is_point(value):
        return [tuple(float(v) for v in value)]
    if isinstance(value, (list, tuple)) and value and all(_is_point(p) for p in value):
        return [tuple(float(v) for v in p) for p in value]
    raise ValueError(f"{where}: expected a goal [x, y] or a sequence of goals, got {value!r}")


class TaskGenerator:
    """The fleet (``agents`` / ``mapf_agents``) and the task stream (``release`` / ``finished``) of a tasks
    file. ``config``: the file's content; ``agent_types``: ``{type name: Agent}`` (None: point particles);
    ``rng``: a ``random.Random`` for the starts that are not given and for ``order: random``."""

    def __init__(self, config: dict, agent_types: dict | None = None, rng: random.Random | None = None):
        config = dict(config or {})
        unknown = set(config) - KEYS
        if unknown:
            raise ValueError(f"tasks: unknown keys {sorted(unknown)} (known: {sorted(KEYS)})")
        self.agent_types = dict(agent_types) if agent_types else None
        self.rng = rng or random.Random(0)
        self.fleet = self._fleet(config.get("agents"))
        self.pool = self._pool(config.get("tasks"))

        self.mode = config.get("mode", "singular")
        if self.mode not in MODES:
            raise ValueError(f"tasks: mode must be one of {MODES}, got {self.mode!r}")
        self.order = config.get("order", "sequential")
        if self.order not in ORDERS:
            raise ValueError(f"tasks: order must be one of {ORDERS}, got {self.order!r}")
        self.count = config.get("count")
        if self.mode in ("repeated", "achieved"):
            if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
                raise ValueError(f"tasks: mode {self.mode!r} needs count (an integer >= 1), got {self.count!r}")
        elif self.count is not None:
            raise ValueError(f"tasks: count is used by the repeated and achieved modes only, not {self.mode!r}")
        self.frequency = config.get("frequency")
        if self.frequency is not None and self.frequency <= 0:
            raise ValueError(f"tasks: frequency must be > 0 releases per time unit, got {self.frequency!r}")
        if self.mode == "endless" and self.frequency is None:
            raise ValueError("tasks: mode 'endless' needs a frequency (how often new tasks are created)")
        self.per_release = int(config.get("per_release", 1))
        if self.per_release < 1:
            raise ValueError(f"tasks: per_release must be >= 1, got {self.per_release!r}")
        if self.mode != "singular" and not self.pool:
            raise ValueError(f"tasks: mode {self.mode!r} needs tasks (under `tasks`, or goals on the agents)")

        # How many tasks finish the run (None: never) and how many are created in all (None: no end).
        self.target = {"singular": len(self.pool), "repeated": len(self.pool) * (self.count or 0),
                       "achieved": self.count, "endless": None}[self.mode]
        self._supply = None if self.mode == "endless" or (self.mode == "achieved" and self.frequency) else self.target
        self.released = 0                    # tasks created so far
        self.created = []                    # ... and the tasks themselves, as released
        self._round = []                     # the pool entries still to draw in this pass over the pool
        self._to_frame = lambda points: [tuple(p) for p in points]   # world -> roadmap frame, once a map is given
        self._placed = None                  # the fleet with starts, in the roadmap frame

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_yaml(cls, tasks_path: str, agents_path: str | None = None, seed: int = 0) -> "TaskGenerator":
        """From a tasks file and, for agents with bodies, an agent-types file."""
        with open(tasks_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return cls(config, load_agent_types(agents_path) if agents_path else None, random.Random(seed))

    @classmethod
    def from_list(cls, tasks: list) -> "TaskGenerator":
        """A stream over tasks made beforehand (``task_name``, ``start_time`` and ``waypoints``, in the
        roadmap frame): each is released at its ``start_time`` and the run is finished when all are done."""
        stream = cls({})
        stream._fixed = sorted((dict(task) for task in tasks), key=lambda task: int(task["start_time"]))
        stream.target = stream._supply = len(stream._fixed)
        return stream

    _fixed = None                            # (from_list) the tasks, by start time

    def _fleet(self, spec) -> list:
        """The agents as ``{name, type, [start], [goals], [yaw], [pitch]}``."""
        types = self.agent_types
        if spec is None:
            return []
        if isinstance(spec, int) and not isinstance(spec, bool):
            if types:
                raise ValueError(f"tasks: with agent types, give the number of agents per type "
                                 f"(agents: {{{next(iter(types))}: {spec}}}), not a single number")
            entries = [{} for _ in range(spec)]
        elif isinstance(spec, dict):
            entries = [{"type": kind} for kind, n in spec.items() for _ in range(int(n))]
        elif isinstance(spec, list):
            entries = [dict(entry) for entry in spec]
        else:
            raise ValueError(f"tasks: agents must be a number, {{type: number}} or a list of agents, got {spec!r}")
        fleet, names = [], set()
        for k, entry in enumerate(entries):
            unknown = set(entry) - AGENT_KEYS
            if unknown:
                raise ValueError(f"tasks: agent {k} has unknown keys {sorted(unknown)} (known: {sorted(AGENT_KEYS)})")
            name = str(entry.get("name", f"agent{k}"))
            if name in names:
                raise ValueError(f"tasks: two agents are called {name!r}")
            names.add(name)
            kind = entry.get("type")
            if types is None and kind is not None:
                raise ValueError(f"tasks: agent {name!r} has type {kind!r} but no agent types were given (agents.yaml)")
            if types is not None and kind not in types:
                raise ValueError(f"tasks: agent {name!r} has type {kind!r}; the agent types are {sorted(types)}")
            agent = {"name": name, "type": kind}
            if "goal" in entry and "goals" in entry:
                raise ValueError(f"tasks: agent {name!r} has both goal and goals")
            if "goal" in entry or "goals" in entry:
                agent["goals"] = _goals(entry.get("goal", entry.get("goals")), f"tasks: agent {name!r}")
            if "start" in entry:
                if not _is_point(entry["start"]):
                    raise ValueError(f"tasks: agent {name!r} start must be a point, got {entry['start']!r}")
                agent["start"] = tuple(float(v) for v in entry["start"])
            agent.update({key: entry[key] for key in ("yaw", "pitch") if key in entry})
            fleet.append(agent)
        if not fleet:
            raise ValueError("tasks: no agents")
        return fleet

    def _pool(self, spec) -> list:
        """The tasks as ``{name, goals, types (None: any), agent (None: any)}``: the agents' own goals,
        then the ``tasks`` entries."""
        pool = [{"name": f"{agent['name']}_task", "goals": agent["goals"], "types": None, "agent": agent["name"]}
                for agent in self.fleet if "goals" in agent]
        if spec is None:
            return pool
        if not isinstance(spec, dict) or not spec:
            raise ValueError(f"tasks: tasks must be {{name: goal(s)}}, got {spec!r}")
        fleet_types = {agent["type"] for agent in self.fleet}
        for name, entry in spec.items():
            where = f"tasks: task {name!r}"
            allowed = None
            if isinstance(entry, dict):
                unknown = set(entry) - {"goal", "goals", "types"}
                if unknown or ("goal" in entry) == ("goals" in entry):
                    raise ValueError(f"{where} takes goal or goals, and optionally types; got {sorted(entry)}")
                goals = _goals(entry.get("goal", entry.get("goals")), where)
                if entry.get("types") is not None:
                    allowed = [entry["types"]] if isinstance(entry["types"], str) else list(entry["types"])
                    if self.agent_types is None:
                        raise ValueError(f"{where} names types {allowed} but no agent types were given (agents.yaml)")
                    missing = [kind for kind in allowed if kind not in self.agent_types]
                    if missing:
                        raise ValueError(f"{where}: unknown types {missing}; the agent types are {sorted(self.agent_types)}")
                    if not fleet_types & set(allowed):
                        raise ValueError(f"{where} is for {allowed}, but the fleet has only {sorted(fleet_types)}")
            else:
                goals = _goals(entry, where)
            pool.append({"name": str(name), "goals": goals, "types": allowed, "agent": None})
        return pool

    # ------------------------------------------------------------------ the fleet
    def agents(self, graph_map) -> list:
        """The fleet for ``graph_map`` as agent dicts in the roadmap frame: ``name``, ``type``, ``start``
        and, with agent types, ``agent`` (the body); ``yaw`` / ``pitch`` when given. A start that is not
        given is drawn among the map's free positions (its roadmap nodes when it has them), clear of
        the other starts and of every goal by the bodies' reach. A given start or goal outside the
        map or inside an obstacle is an error. Also sets the frame the tasks are released in."""
        if self._placed is not None:
            return [dict(agent) for agent in self._placed]
        self._to_frame = lambda points: points_to_roadmap_frame(graph_map, points)
        placed = []
        for entry in self.fleet:
            agent = {"name": entry["name"], "type": entry["type"]}
            if self.agent_types is not None:
                agent["agent"] = self.agent_types[entry["type"]]
            if "start" in entry:
                agent["start"] = self._to_frame([entry["start"]])[0]
            agent.update({key: entry[key] for key in ("yaw", "pitch") if key in entry})
            placed.append(agent)
        # (checked in world coordinates: on a grid, a point off the map would be moved to its border cell)
        for what, points in ([(f"the start of agent {entry['name']!r}", [entry["start"]]) for entry in self.fleet if "start" in entry]
                             + [(f"a goal of task {task['name']!r}", task["goals"]) for task in self.pool]):
            for p, placed_p in zip(points, self._to_frame(points)):
                if not all(lo <= c <= hi for c, (lo, hi) in zip(p, graph_map.bounds)):
                    raise ValueError(f"tasks: {what} is outside the map: {p}")
                if graph_map.in_collision(placed_p):
                    raise ValueError(f"tasks: {what} is inside an obstacle (inflated by the fleet's size): {p}")
        self._draw_starts(graph_map, placed)
        self._placed = placed
        return [dict(agent) for agent in placed]

    def mapf_agents(self, graph_map) -> list:
        """The fleet for a one-shot solver (``solve_mapf``): ``agents`` with a ``goal`` each. Needs every
        agent listed with exactly one goal, no ``tasks`` and the ``singular`` mode."""
        if self.mode != "singular" or self.frequency is not None:
            raise ValueError(f"tasks: mode {self.mode!r} / a frequency: {ONE_SHOT}")
        if any(task["agent"] is None for task in self.pool):
            raise ValueError(f"tasks: a pool of tasks: {ONE_SHOT}")
        agents = self.agents(graph_map)
        for agent, entry in zip(agents, self.fleet):
            if len(entry.get("goals", ())) != 1:
                raise ValueError(f"tasks: agent {entry['name']!r} has {len(entry.get('goals', ()))} goals: {ONE_SHOT}")
            agent["goal"] = self._to_frame(entry["goals"])[0]
        return agents

    def goal_points(self) -> list:
        """Every goal of every task, in the frame the tasks are released in (see ``agents``)."""
        return list(dict.fromkeys(goal for task in self.pool for goal in self._to_frame(task["goals"])))

    def reach(self, name=None) -> float:
        """An agent's bounding radius + clearance (no name: the largest; 0 for point particles)."""
        if self.agent_types is None:
            return 0.0
        kinds = [agent["type"] for agent in self.fleet if name in (None, agent["name"])]
        return max(self.agent_types[kind].reach for kind in kinds)

    def _draw_starts(self, graph_map, placed):
        """Give every agent without a start one, in place."""
        todo = [agent for agent in placed if "start" not in agent]
        if not todo:
            return
        largest = self.reach()
        # (position, reach) of what a new start has to stay clear of: the given starts and every goal.
        taken = [(agent["start"], self.reach(agent["name"])) for agent in placed if "start" in agent]
        taken += [(goal, largest) for task in self.pool for goal in self._to_frame(task["goals"])]
        nodes = [tuple(float(c) for c in node.current) for node in getattr(graph_map, "nodes", None) or []]
        if nodes:
            candidates = [p for p in nodes if not graph_map.in_collision(p)]
            self.rng.shuffle(candidates)
        else:                                # no roadmap yet: free positions of the map
            bounds = [tuple(b) for b in graph_map.bounds]
            draws = [[self.rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(200 * len(todo) + 2000)]
            candidates = [p for p in dict.fromkeys(self._to_frame(draws)) if not graph_map.in_collision(p)]
        for agent in todo:
            reach = self.reach(agent["name"])
            for p in candidates:
                if all(math.dist(p, q) > reach + r + 1e-9 for q, r in taken) and p not in [q for q, _ in taken]:
                    agent["start"] = p
                    taken.append((p, reach))
                    break
            else:
                raise ValueError(f"tasks: no free start left for agent {agent['name']!r} "
                                 f"({len(candidates)} candidate positions)")

    # ------------------------------------------------------------------ the task stream
    def _draw(self) -> dict:
        """The next pool entry: pass after pass over the pool, in order or shuffled (``order``)."""
        if not self._round:
            self._round = list(self.pool)
            if self.order == "random":
                self.rng.shuffle(self._round)
        return self._round.pop(0)

    def _due(self, t) -> int:
        """How many tasks are to exist by time ``t``."""
        if self.frequency is None:
            return self._supply
        due = (int(math.floor(t * self.frequency + 1e-9)) + 1) * self.per_release     # a release at t = 0, then every 1 / frequency
        return due if self._supply is None else min(due, self._supply)

    def release(self, t) -> list:
        """The tasks created at time ``t`` (call it with every time step, in order): dicts with
        ``task_name``, ``start_time`` and ``waypoints`` (the goals, in the roadmap frame), and, when the
        task is limited, ``types`` (the agent types that may take it) or ``agent`` (the one agent)."""
        if self._fixed is not None:
            out = []
            while self.released < len(self._fixed) and int(self._fixed[self.released]["start_time"]) <= t:
                out.append(self._fixed[self.released])
                self.released += 1
            self.created += out
            return out
        out = []
        while self.released < self._due(t):
            entry = self._draw()
            turn = self.released // len(self.pool)           # which pass over the pool this is
            task = {"task_name": entry["name"] if self.mode == "singular" else f"{entry['name']}#{turn + 1}",
                    "start_time": t, "waypoints": self._to_frame(entry["goals"])}
            if entry["types"] is not None:
                task["types"] = list(entry["types"])
            if entry["agent"] is not None:
                task["agent"] = entry["agent"]
            out.append(task)
            self.released += 1
        self.created += out
        return out

    def finished(self, completed: int) -> bool:
        """Whether ``completed`` tasks finish the run (never in the ``endless`` mode)."""
        return self.target is not None and completed >= self.target
