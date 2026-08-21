"""
Run Neural-ATTF (multi-agent pickup-and-delivery via token passing + Neural
STA*) across several random task frequencies, in both discrete (grid) and
continuous (PRM) graph modes.

Usage:
    python scripts/mapf/run_neural_attf.py

Outputs:
    figs/neural_attf/neural_attf_<mode>_freq<f>.png
    figs/neural_attf/neural_attf_<mode>_freq<f>.gif
    path_planning/maps/2d/neural_attf/solution_<mode>_freq<f>.yaml
"""

from __future__ import annotations

import os
import random
import sys
import time
from copy import deepcopy

# Make the source tree importable when the package is installed non-editably:
# the new `decentralized/neural_attf/` modules may only exist in the working
# tree, not in site-packages.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import NeuralATTF
from path_planning.multi_agent_planner.decentralized.neural_attf.simulation import Simulation
from path_planning.multi_agent_planner.decentralized.neural_attf.tasks_generator import (
    convert_mapd_tasks,
)
from path_planning.utils.util import (
    _to_native_yaml,
    mapd_agents_to_roadmap_frame,
    points_to_roadmap_frame,
    read_graph_sampler_from_yaml,
    read_mapd_from_yaml,
    set_global_seed,
    write_to_yaml,
)


def _make_map_mapd(map_yaml: str, use_discrete_space: bool, agent_radius: float,
                   register_task_endpoints: bool = False, roadmap_type: str | None = None,
                   sweep_backend: str = "auto"):
    """
    Build the roadmap for a life-long MAPD instance (e.g. ``2d_mapd.yaml``).

    Agents carry only a ``start`` (no goal) and the parking/task endpoints come from
    the file rather than random sampling. Returns the raw MAPD payload alongside the
    map so the caller can convert the predefined tasks. In discrete mode every free
    cell is a roadmap node, so the fixed task pickup/delivery endpoints are
    guaranteed reachable. In continuous mode they are NOT sampled nodes, so set
    ``register_task_endpoints=True`` to add the candidate task/parking points to the
    graph (with their exact coords) before the roadmap is built.

    ``roadmap_type`` selects the roadmap (``GraphSampler.generate_map``):
    ``"grid"``, ``"prm"``, ``"rrg"``, ``"cdt"``, ``"voronoi"``, ``"midpoints"``,
    ``"centroids"``, ``"dt"``, or ``"halton..."``. Defaults to ``"grid"`` in discrete
    mode and ``"prm"`` in continuous mode. NOTE for MAPD: ``prm`` connects the
    registered endpoints via KNN so they stay reachable; topology-defining types
    (``cdt``/``voronoi``/``rrg``) rebuild the node set and may not route through
    arbitrary registered endpoints — prefer ``prm``/``halton`` when endpoints are
    fixed.
    """
    if roadmap_type is None:
        roadmap_type = "grid" if use_discrete_space else "prm"
    print(f"roadmap_type: {roadmap_type}")
    map_ = read_graph_sampler_from_yaml(map_yaml, use_discrete_space=use_discrete_space, sweep_backend=sweep_backend)
    mapd = read_mapd_from_yaml(map_yaml)
    # Inflate obstacles by the agent footprint. NOTE: the ESDF is quantized to grid
    # steps, so the nearest free cell to an obstacle sits at distance 1.0 — an
    # inflation radius < 1.0 marks no cells at all. With the +sqrt(2)/2 buffer this
    # means ``agent_radius`` must be >= ~0.293 for any inflation to appear; a point
    # agent (agent_radius=0 -> radius 0.707) intentionally inflates nothing.
    map_.inflate_obstacles(radius=agent_radius + np.sqrt(2) / 2)

    if use_discrete_space:
        map_.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
    else:
        # Sparser PRM with shorter edges: the SIPP low-level planner rebuilds its
        # per-node/edge interval graph each call, so fewer nodes/edges keeps it fast
        # while staying dense enough for radius-aware connectivity.
        map_.set_parameters(sample_num=400, num_neighbors=8.0, min_edge_len=0.0, max_edge_len=3.0)

    # World-frame starts so generateRandomNodes registers them as roadmap nodes.
    agents_rt = mapd_agents_to_roadmap_frame(map_, mapd["agents"])
    map_.set_start([a["start"] for a in agents_rt])

    if register_task_endpoints:
        # Continuous roadmaps: candidate task/parking endpoints aren't sampled nodes,
        # so register them (exact coords) via set_goal so generateRandomNodes adds
        # them and generate_map connects them. Cleared after build for a clean figure.
        endpoint_world = (
            [p for pair in mapd["goal_locations"] for p in pair]
            + [p for pair in mapd["start_locations"] for p in pair]
            + list(mapd["non_task_endpoints"])
        )
        map_.set_goal(points_to_roadmap_frame(map_, endpoint_world))
    else:
        map_.set_goal([])

    # Sample nodes for the requested roadmap, then connect them via the matching
    # builder. generate_map dispatches per type; rrg/planar rebuild the node set, so
    # read the authoritative nodes/road_map back from the map afterwards.
    nodes = map_.generateRandomNodes(generate_grid_nodes=use_discrete_space, roadmap_type=roadmap_type)
    map_.generate_map(roadmap_type, nodes)
    nodes = map_.nodes
    road_map = map_.road_map

    if register_task_endpoints:
        map_.set_goal([])  # nodes persist in node_index_dict; keep the figure clean

    return map_, agents_rt, mapd, nodes, road_map


def _run_mapd(
    map_yaml: str,
    use_discrete_space: bool = True,
    horizon: int = 2000,
    seed: int = 42,
    agent_radius: float = 1.0,
    out_dir_figs: str = "figs/neural_attf",
    out_dir_yaml: str = "path_planning/maps/2d/neural_attf",
    make_gif: bool = True,
):
    """
    Run Neural-ATTF on a life-long MAPD instance whose agents, parking endpoints
    and multi-leg tasks are all read from ``map_yaml`` (e.g. ``2d_mapd.yaml``).
    """
    mode = "discrete" if use_discrete_space else "continuous"
    tag = f"mapd_{mode}"
    print(f"\n=== Neural-ATTF (MAPD) | {os.path.basename(map_yaml)} | mode={mode} ===")

    rng = random.Random(seed)
    map_, agents, mapd, nodes, road_map = _make_map_mapd(map_yaml, use_discrete_space, agent_radius,roadmap_type='halton')

    # Convert predefined endpoints/tasks from world coords to the roadmap frame,
    # keeping each task whole (multi-leg route under one task name).
    parking_pts = points_to_roadmap_frame(map_, mapd["non_task_endpoints"])
    tasks = convert_mapd_tasks(
        mapd["tasks"], point_fn=lambda p: points_to_roadmap_frame(map_, [p])[0]
    )
    n_waypoints = sum(len(t["waypoints"]) for t in tasks)
    print(f"  {len(agents)} agents | {len(tasks)} tasks ({n_waypoints} waypoints)")

    # Surface pickups/deliveries/parking in the figures/GIF (cosmetic only):
    # within each task's waypoints, even indices are pickups, odd are deliveries.
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
        extra_summary={"map_yaml": map_yaml, "n_mapd_tasks": len(mapd["tasks"])},
    )


def _solve_and_save(
    *,
    map_,
    agents,
    tasks,
    parking_pts,
    nodes,
    road_map,
    tag: str,
    use_discrete_space: bool,
    agent_radius: float,
    horizon: int,
    rng,
    out_dir_figs: str,
    out_dir_yaml: str,
    make_gif: bool,
    velocity: float = 0.0,
    low_level: str = "grid",
    delay_probability: float = 0.0,
    extra_summary: dict | None = None,
):
    """
    Shared tail for the Neural-ATTF runners: simulate the token-passing planner
    over the given task stream, write the YAML solution summary, and render the
    static PNG (+ optional GIF). Used by :func:`_run_mapd` (predefined MAPD tasks)
    and the random/continuous runners. ``agent_radius``/``velocity`` give the planner
    a radius-aware (CGAL swept) agent–agent footprint; ``delay_probability`` is the
    per-step probability that an agent is held in place by ``Simulation``.
    """
    planner = NeuralATTF(
        graph_map=map_,
        agents=agents,
        non_task_endpoints=parking_pts,
        a_star_max_iter=4000,
        encoder=None,
        grid_overlay=None,
        alpha=0.001,
        heuristic_type="manhattan" if use_discrete_space else "euclidean",
        agent_radius=agent_radius,
        velocity=velocity,
        low_level=low_level,
    )
    sim = Simulation(tasks=[], agents=agents, delay_probability=delay_probability, rng=rng)

    target = len(tasks)
    t0 = time.time()
    while planner.get_completed_tasks() < target and sim.get_time() < horizon:
        new_tasks = [t for t in tasks if int(t["start_time"]) == sim.get_time()]
        if new_tasks:
            planner.update_tasks(sim.get_time(), new_tasks)
        sim.time_forward(planner)
    wall = time.time() - t0

    completed = planner.get_completed_tasks()
    print(
        f"  completed {completed}/{target} tasks in {sim.get_time()} steps "
        f"(planner: {sim.get_algo_time():.2f}s, wall: {wall:.2f}s)"
    )

    summary = {
        "mapf_solver_name": "neural_attf",
        "schedule": sim.get_actual_paths(),
        "completed_tasks": completed,
        "n_tasks": target,
        "n_replans": planner.get_n_replans(),
        "completed_tasks_times": planner.get_completed_tasks_times(),
        "tasks": tasks,
        "horizon": horizon,
        "use_discrete_space": use_discrete_space,
        "agent_radius": agent_radius,
        "num_nodes": len(map_.nodes),
        "num_edges": len(map_.edges),
        "runtime": wall,
        "algo_time": sim.get_algo_time(),
    }
    if extra_summary:
        summary.update(extra_summary)

    os.makedirs(out_dir_figs, exist_ok=True)
    os.makedirs(out_dir_yaml, exist_ok=True)
    write_to_yaml(_to_native_yaml(summary), os.path.join(out_dir_yaml, f"solution_{tag}.yaml"))

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_, nodes, road_map, map_frame=False)
    for _, trajectory in summary["schedule"].items():
        path = np.array([[p["x"], p["y"]] for p in trajectory])
        vis.plot_path(path, map_frame=False)
    png_path = os.path.join(out_dir_figs, f"neural_attf_{tag}.png")
    vis.savefig(png_path)
    print(f"  saved static image: {png_path}")
    vis.close()

    if make_gif:
        schedule = {"schedule": deepcopy(summary["schedule"])}
        gif_path = os.path.join(out_dir_figs, f"neural_attf_{tag}.gif")
        vis = Visualizer2D()
        vis.animate(
            gif_path,
            map_,
            schedule,
            road_map=road_map,
            skip_frames=1,
            intermediate_frames=1,
            speed=3,
            radius=agent_radius,
            map_frame=False,
        )
        vis.close()
        print(f"  saved animation: {gif_path}")


def main():
    set_global_seed(42)

    # Life-long MAPD instance: agents, parking endpoints and multi-leg tasks are
    # all read from the YAML. Discrete mode so every free cell is a roadmap node
    # and the fixed task endpoints are reachable.
    # NOTE: agent_radius must stay small relative to the warehouse's width-1
    # corridors. The footprint clearance is 2*agent_radius, so a radius >= 0.5
    # makes two agents on adjacent cells (1.0 apart) mutually blocking and
    # deadlocks the corridors. 0.293 -> inflation 1.0 (one cell), still passable.
    _run_mapd(
        map_yaml="path_planning/maps/2d/2d_mapd.yaml",
        use_discrete_space=True,
        horizon=2000,
        seed=42,
        agent_radius=0.293,
        make_gif=False,
    )


if __name__ == "__main__":
    main()
