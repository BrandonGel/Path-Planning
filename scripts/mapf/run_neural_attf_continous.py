"""
Run Neural-ATTF on a *continuous* (PRM) roadmap with a non-zero agent radius.

Reads the map, agents and ``start_locations`` / ``goal_locations`` /
``non_task_endpoints`` / ``n_tasks`` / ``task_freq`` config from ``2d_mapd.yaml``
(its predefined ``tasks`` are ignored), samples a continuous PRM roadmap, and runs
the token-passing planner with a radius-``agent_radius`` disk footprint. Agent–agent
clearance is computed as the set of graph nodes each disk covers via the map's CGAL
swept-collision query (see ``NeuralATTF._covered_nodes``); static clearance comes
from obstacle inflation.

Because PRM nodes are sampled (not on a grid), the task/parking/agent endpoints are
registered as graph nodes up front (``register_task_endpoints=True``) so A* can
reach them at their exact coordinates.

Usage:
    python scripts/mapf/run_neural_attf_continous.py

Outputs:
    figs/neural_attf/neural_attf_continuous.png
    figs/neural_attf/neural_attf_continuous.gif
    path_planning/maps/2d/neural_attf/solution_continuous.yaml
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


def _run_continuous(
    map_yaml: str,
    horizon: int = 2000,
    seed: int = 42,
    agent_radius: float = 0.5,
    velocity: float = 1.0,
    roadmap_type: str = "prm",
    sweep_backend: str = "auto",
    out_dir_figs: str = "figs/neural_attf",
    out_dir_yaml: str = "path_planning/maps/2d/neural_attf",
    make_gif: bool = True,
):
    """
    Run Neural-ATTF on a continuous roadmap with randomly generated MAPD tasks and a
    non-zero agent radius. ``roadmap_type`` selects the roadmap construction
    (``"prm"``, ``"rrg"``, ``"cdt"``, ``"voronoi"``, ``"halton"``, ...). For fixed
    MAPD endpoints, ``prm``/``halton`` keep them reachable (KNN connection);
    topology-defining types (cdt/voronoi/rrg) may not route through them.
    """
    tag = f"continuous_{roadmap_type}"
    print(f"\n=== Neural-ATTF (continuous {roadmap_type}) | {os.path.basename(map_yaml)} | radius={agent_radius} ===")

    rng = random.Random(seed)
    # Continuous roadmap; register the candidate task/parking endpoints as nodes so
    # the sampled roadmap can reach them at their exact coordinates.
    map_, agents, mapd, nodes, road_map = _make_map_mapd(
        map_yaml, use_discrete_space=False, agent_radius=agent_radius,
        register_task_endpoints=True, roadmap_type=roadmap_type, sweep_backend=sweep_backend,
    )

    # Scenario config from the YAML. ``task_freq`` is stored as a 1-element list.
    n_tasks = int(mapd.get("n_tasks", 0))
    task_freq = mapd.get("task_freq", 1.0)
    if isinstance(task_freq, (list, tuple)):
        task_freq = task_freq[0]

    # Generate random multi-leg tasks; in continuous mode points_to_roadmap_frame
    # passes world coords through — they coincide with the registered endpoint nodes.
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
        f"| {len(map_.nodes)} nodes | task_freq={task_freq} | velocity={velocity}"
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
        use_discrete_space=False,
        agent_radius=agent_radius,
        horizon=horizon,
        rng=rng,
        out_dir_figs=out_dir_figs,
        out_dir_yaml=out_dir_yaml,
        make_gif=make_gif,
        velocity=velocity,
        low_level="sipp",
        extra_summary={
            "map_yaml": map_yaml,
            "n_tasks": n_tasks,
            "task_freq": task_freq,
            "agent_radius": agent_radius,
            "velocity": velocity,
        },
    )


def main():
    # CLI: `python run_neural_attf_continous.py [roadmap_type] [sweep_backend]`
    # roadmap_type: prm (default), rrg, cdt, voronoi, halton, midpoints, centroids, dt
    # sweep_backend: auto (default; shapely in 2D, cgal in 3D+), or force cgal/shapely
    roadmap_type = sys.argv[1] if len(sys.argv) > 1 else "halton"
    sweep_backend = sys.argv[2] if len(sys.argv) > 2 else "auto"
    set_global_seed(42)
    _run_continuous(
        map_yaml="path_planning/maps/2d/2d_mapd.yaml",
        horizon=2000,
        seed=42,
        agent_radius=0.5,
        velocity=1.0,
        roadmap_type=roadmap_type,
        sweep_backend=sweep_backend,
        make_gif=True,  # flip to True to also render the (slower) GIF
    )


if __name__ == "__main__":
    main()
