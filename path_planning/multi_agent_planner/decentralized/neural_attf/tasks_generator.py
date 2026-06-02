"""
Random task-stream generator for Neural-ATTF on arbitrary graphs.

Two entry points:

``sample_task_endpoints`` draws disjoint pickup / delivery / parking node
points uniformly from a ``GraphSampler``'s node set, avoiding obstacle nodes
and agent start positions.

``gen_tasks`` produces a chronologically ordered list of pickup-and-delivery
tasks with arrival times spaced by ``1 / task_freq`` (or sampled from an
exponential distribution when ``distribution='poisson'``). Each task is a
dict suitable for ``NeuralATTF.update_tasks``.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple


def _node_point(node) -> Tuple[float, ...]:
    return tuple(float(c) for c in getattr(node, "current"))


def sample_task_endpoints(
    graph_map,
    n_pickups: int,
    n_deliveries: int,
    n_parking: int,
    rng: random.Random | None = None,
    exclude_points: Sequence[Sequence[float]] = (),
) -> Tuple[List[Tuple[float, ...]], List[Tuple[float, ...]], List[Tuple[float, ...]]]:
    rng = rng or random.Random()
    excluded = {tuple(p) for p in exclude_points}
    candidates: List[Tuple[float, ...]] = []
    for node in graph_map.nodes:
        pt = _node_point(node)
        if pt in excluded:
            continue
        if graph_map.in_collision_point(pt):
            continue
        candidates.append(pt)
    total = n_pickups + n_deliveries + n_parking
    if len(candidates) < total:
        raise ValueError(
            f"Not enough free nodes ({len(candidates)}) to sample "
            f"{n_pickups} pickups + {n_deliveries} deliveries + {n_parking} parking endpoints."
        )
    rng.shuffle(candidates)
    pickups = candidates[:n_pickups]
    deliveries = candidates[n_pickups : n_pickups + n_deliveries]
    parking = candidates[n_pickups + n_deliveries : total]
    return pickups, deliveries, parking


def gen_tasks(
    pickup_pts: Sequence[Sequence[float]],
    delivery_pts: Sequence[Sequence[float]],
    n_tasks: int,
    task_freq: float,
    rng: random.Random | None = None,
    distribution: str = "uniform",
    start_time: int = 1,
) -> List[dict]:
    if task_freq <= 0:
        raise ValueError("task_freq must be > 0")
    if not pickup_pts or not delivery_pts:
        raise ValueError("pickup_pts and delivery_pts must be non-empty")
    rng = rng or random.Random()
    inter = 1.0 / float(task_freq)
    arrival = float(start_time)
    tasks: List[dict] = []
    for i in range(n_tasks):
        if distribution == "poisson":
            arrival += rng.expovariate(task_freq)
        else:
            arrival += inter
        start = tuple(rng.choice(list(pickup_pts)))
        goal = tuple(rng.choice(list(delivery_pts)))
        tasks.append(
            {
                "task_name": f"task{i}",
                "start_time": max(int(start_time), int(round(arrival))),
                "start": start,
                "goal": goal,
            }
        )
    return tasks


def gen_mapd_tasks(
    start_location_pairs: Sequence[Sequence[Sequence[float]]],
    goal_location_pairs: Sequence[Sequence[Sequence[float]]],
    n_tasks: int,
    task_freq: float,
    rng: random.Random | None = None,
    distribution: str = "uniform",
    start_time: int = 1,
) -> List[dict]:
    """
    Randomly generate life-long MAPD tasks from grouped start/goal location pairs.

    Each task draws one random ``goal_locations`` pair and one random
    ``start_locations`` pair (each pair is ``[[x, y], [x, y]]``, as stored under
    ``map.start_locations`` / ``map.goal_locations`` of a MAPD YAML) and follows the
    route ``start_pair[0] -> goal_pair[0] -> goal_pair[1] -> start_pair[1]``. The
    resulting 4-waypoint task is a single multi-leg route, converted whole to a
    NeuralATTF task by :func:`convert_mapd_tasks`. Arrival times use the same
    spacing as :func:`gen_tasks` (``inter_arrival = 1/task_freq``, or exponential
    when ``distribution='poisson'``).

    Args:
        start_location_pairs: Iterable of ``[point, point]`` start-location pairs.
        goal_location_pairs: Iterable of ``[point, point]`` goal-location pairs.
        n_tasks: Number of tasks to generate.
        task_freq: Task arrival rate; inter-arrival time is ``1/task_freq``.
        rng: Optional RNG for reproducibility.
        distribution: ``"uniform"`` (fixed spacing) or ``"poisson"`` (exponential).
        start_time: Earliest task arrival time.

    Returns:
        List of raw task dicts ``{task_name, start_time, waypoints}`` (world coords),
        suitable for :func:`convert_mapd_tasks`.
    """
    if task_freq <= 0:
        raise ValueError("task_freq must be > 0")
    if not start_location_pairs or not goal_location_pairs:
        raise ValueError("start_location_pairs and goal_location_pairs must be non-empty")
    rng = rng or random.Random()
    inter = 1.0 / float(task_freq)
    arrival = float(start_time)
    start_pairs = list(start_location_pairs)
    goal_pairs = list(goal_location_pairs)
    tasks: List[dict] = []
    for i in range(n_tasks):
        if distribution == "poisson":
            arrival += rng.expovariate(task_freq)
        else:
            arrival += inter
        gp = rng.choice(goal_pairs)
        sp = rng.choice(start_pairs)
        # Route: start_pair[0] -> goal_pair[0] -> goal_pair[1] -> start_pair[1].
        waypoints = [
            [float(c) for c in sp[0]],
            [float(c) for c in gp[0]],
            [float(c) for c in gp[1]],
            [float(c) for c in sp[1]],
        ]
        tasks.append(
            {
                "task_name": f"task{i}",
                "start_time": max(int(start_time), int(round(arrival))),
                "waypoints": waypoints,
            }
        )
    return tasks


def convert_mapd_tasks(
    raw_tasks: Sequence[dict],
    point_fn: Optional[Callable[[Sequence[float]], Tuple[float, ...]]] = None,
) -> List[dict]:
    """
    Convert raw life-long MAPD tasks into NeuralATTF tasks, keeping each task WHOLE.

    Each task's ``waypoints`` is an ordered list of points
    (pickup, delivery, pickup, delivery, ...) describing a single multi-leg route.
    The whole route is preserved under the original ``task_name`` so NeuralATTF
    assigns the entire task to one agent that visits every waypoint in order (see
    ``NeuralATTF.update_tasks``); the task completes only at the final waypoint.

    Args:
        raw_tasks: Dicts with ``task_name``, ``start_time`` and ``waypoints`` (as
            returned under the ``tasks`` key of ``read_mapd_from_yaml`` or by
            ``gen_mapd_tasks``).
        point_fn: Optional per-waypoint converter (e.g. world -> roadmap frame). If
            omitted, waypoints are coerced to plain float tuples unchanged.

    Returns:
        List of task dicts ``{task_name, start_time, waypoints}`` (>=2 waypoints),
        preserving the input task ordering.
    """
    point_fn = point_fn or (lambda p: tuple(float(c) for c in p))
    out: List[dict] = []
    for ti, task in enumerate(raw_tasks):
        waypoints = [point_fn(p) for p in task.get("waypoints", [])]
        if len(waypoints) < 2:
            continue
        out.append(
            {
                "task_name": task.get("task_name", f"task{ti}"),
                "start_time": int(task.get("start_time", 0)),
                "waypoints": waypoints,
            }
        )
    return out
