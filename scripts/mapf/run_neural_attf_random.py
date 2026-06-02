"""
Run Neural-ATTF on a life-long MAPD instance with a *randomly generated* task
stream. Reads the map, agents, parking endpoints and the
``start_locations`` / ``goal_locations`` / ``n_tasks`` / ``task_freq`` /
``n_delays_per_agent`` config from ``2d_mapd.yaml``, but ignores that file's
predefined ``tasks``.

Each generated task draws one random ``goal_locations`` pair and one random
``start_locations`` pair, following the route
``start_pair[0] -> goal_pair[0] -> goal_pair[1] -> start_pair[1]`` (two
pickup->delivery legs after expansion). Arrival times are spaced by
``1/task_freq``; ``n_delays_per_agent`` maps to a global per-step
``delay_probability = n_delays_per_agent / horizon``.

Usage:
    python scripts/mapf/run_neural_attf_random.py

Outputs:
    figs/neural_attf/neural_attf_random_<mode>.png
    figs/neural_attf/neural_attf_random_<mode>.gif
    path_planning/maps/2d/neural_attf/solution_random_<mode>.yaml
"""

from __future__ import annotations

import os
import random
import sys

# Make the source tree importable when the package is installed non-editably, and
# allow importing the shared helpers from the sibling ``run_neural_attf`` script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from path_planning.multi_agent_planner.decentralized.neural_attf.tasks_generator import (
    convert_mapd_tasks,
    gen_mapd_tasks,
)
from path_planning.utils.util import points_to_roadmap_frame, set_global_seed
from run_neural_attf import _make_map_mapd, _solve_and_save


def _run_random(
    map_yaml: str,
    use_discrete_space: bool = True,
    horizon: int = 2000,
    seed: int = 42,
    # Obstacle inflation accounts for the agent footprint (see _make_map_mapd):
    # the ESDF is grid-quantized, so agent_radius must be >= ~0.293 for any
    # inflation to take effect (radius 0 inflates nothing). Match _run_mapd's 1.0.
    agent_radius: float = 1.0,
    out_dir_figs: str = "figs/neural_attf",
    out_dir_yaml: str = "path_planning/maps/2d/neural_attf",
    make_gif: bool = True,
):
    """
    Run Neural-ATTF on randomly generated MAPD tasks drawn from the instance's
    ``start_locations`` / ``goal_locations`` pairs (the file's predefined ``tasks``
    are ignored).
    """
    mode = "discrete" if use_discrete_space else "continuous"
    tag = f"random_{mode}"
    print(f"\n=== Neural-ATTF (random MAPD) | {os.path.basename(map_yaml)} | mode={mode} ===")

    rng = random.Random(seed)
    map_, agents, mapd, nodes, road_map = _make_map_mapd(map_yaml, use_discrete_space, agent_radius)

    # Scenario config from the YAML. ``task_freq`` is stored as a 1-element list.
    n_tasks = int(mapd.get("n_tasks", 0))
    task_freq = mapd.get("task_freq", 1.0)
    if isinstance(task_freq, (list, tuple)):
        task_freq = task_freq[0]
    n_delays_per_agent = int(mapd.get("n_delays_per_agent", 0))
    delay_probability = (n_delays_per_agent / horizon) if horizon > 0 else 0.0

    # Generate random 4-waypoint tasks and convert each whole (multi-leg route
    # under one task name) to the roadmap frame (same pipeline as _run_mapd).
    raw_tasks = gen_mapd_tasks(
        start_location_pairs=mapd["start_locations"],
        goal_location_pairs=mapd["goal_locations"],
        n_tasks=n_tasks,
        task_freq=task_freq,
        rng=rng,
    )
    tasks = convert_mapd_tasks(
        raw_tasks, point_fn=lambda p: points_to_roadmap_frame(map_, [p])[0]
    )
    n_waypoints = sum(len(t["waypoints"]) for t in tasks)
    print(
        f"  {len(agents)} agents | {len(tasks)} tasks ({n_waypoints} waypoints) "
        f"| task_freq={task_freq} | delay_probability={delay_probability:g}"
    )

    # Surface pickups/deliveries/parking in the figures/GIF (cosmetic only):
    # within each task's waypoints, even indices are pickups, odd are deliveries.
    parking_pts = points_to_roadmap_frame(map_, mapd["non_task_endpoints"])
    pickups, deliveries = [], []
    for t in tasks:
        wps = t["waypoints"]
        pickups += list(wps[0::2])
        deliveries += list(wps[1::2])
    map_.set_endpoints(pickups=pickups, deliveries=deliveries, parking=parking_pts)

    _solve_and_save(
        map_=map_,
        agents=agents,
        tasks=tasks,
        parking_pts=parking_pts,
        nodes=nodes,
        road_map=road_map,
        tag=tag,
        use_discrete_space=use_discrete_space,
        agent_radius=agent_radius,
        horizon=horizon,
        rng=rng,
        out_dir_figs=out_dir_figs,
        out_dir_yaml=out_dir_yaml,
        make_gif=make_gif,
        delay_probability=delay_probability,
        extra_summary={
            "map_yaml": map_yaml,
            "n_tasks": n_tasks,
            "task_freq": task_freq,
            "n_delays_per_agent": n_delays_per_agent,
            "delay_probability": delay_probability,
        },
    )


def main():
    set_global_seed(42)
    _run_random(
        map_yaml="path_planning/maps/2d/2d_mapd.yaml",
        use_discrete_space=True,
        horizon=2000,
        seed=42,
        make_gif=True,
    )


if __name__ == "__main__":
    main()
