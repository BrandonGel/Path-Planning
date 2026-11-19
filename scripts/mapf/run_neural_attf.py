"""
Run Neural-ATTF (multi-agent pickup-and-delivery via token passing + Neural
STA*) on one life-long MAPD instance: agents, parking endpoints and tasks are
read from the map YAML. run_neural_attf_random.py and
run_neural_attf_continous.py reuse the helpers defined here.

Usage:
    python scripts/mapf/run_neural_attf.py [--map path/to/mapd.yaml]
    python scripts/mapf/run_neural_attf.py --tasks config/tasks.yaml [--agents config/agents.yaml]

With ``--tasks`` the fleet and the tasks come from that file (and the agents' bodies from
``--agents``; without it: point particles) instead of the map YAML, which still gives the map and
its parking endpoints. See ``path_planning/tasks/task_generator.py`` for the format.

Outputs:
    figs/neural_attf/neural_attf_mapd_<mode>.png
    path_planning/maps/2d/neural_attf/solution_mapd_<mode>.yaml
"""

from __future__ import annotations

import os
import random
import sys
from copy import deepcopy

# Make the source tree importable when the package is installed non-editably:
# the new `decentralized/neural_attf/` modules may only exist in the working
# tree, not in site-packages.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from path_planning.agent import Agent
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.multi_agent_planner.mapf_solver import solve_mapd
from path_planning.multi_agent_planner.decentralized.neural_attf.tasks_generator import (
    convert_mapd_tasks,
)
from path_planning.tasks import TaskGenerator
from path_planning.utils.util import (
    _to_native_yaml,
    mapd_agents_to_roadmap_frame,
    points_to_roadmap_frame,
    read_graph_sampler_from_yaml,
    read_mapd_from_yaml,
    set_global_seed,
    write_to_yaml,
)


DEFAULT_MAP_YAML = "path_planning/maps/2d/2d_mapd.yaml"


def _option_from_argv(flag: str, default=None):
    """Pop an optional ``<flag> VALUE`` (or ``<flag>=VALUE``) from ``sys.argv``."""
    argv = sys.argv
    for i, a in enumerate(argv[1:], start=1):
        if a == flag and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if a.startswith(flag + "="):
            del argv[i]
            return a.split("=", 1)[1]
    return default


def _map_yaml_from_argv(default: str = DEFAULT_MAP_YAML) -> str:
    """Pop an optional ``--map PATH`` (or ``--map=PATH``) from ``sys.argv``.

    Shared by the Neural-ATTF runners so any MAPD YAML (e.g.
    ``2d_mapd_8agents.yaml``) can be run without editing the script; the
    positional arguments of ``run_neural_attf_continous.py`` are untouched.
    """
    return _option_from_argv("--map", default)


def _task_files_from_argv():
    """Pop the optional ``--tasks PATH`` and ``--agents PATH`` from ``sys.argv``: ``(tasks, agents)``."""
    return _option_from_argv("--tasks"), _option_from_argv("--agents")


def _map_tag(map_yaml: str) -> str:
    """Output-name suffix for non-default maps (keeps the 2d_mapd fixtures stable)."""
    stem = os.path.splitext(os.path.basename(map_yaml))[0]
    return "" if stem == "2d_mapd" else f"_{stem}"


def _make_map_mapd(map_yaml: str, use_discrete_space: bool, agent_radius: float,
                   register_task_endpoints: bool = False, roadmap_type: str | None = None,
                   sweep_backend: str = "auto", generator: TaskGenerator | None = None):
    """
    Build the roadmap for a life-long MAPD instance (e.g. ``2d_mapd.yaml``).

    Agents carry only a ``start`` (no goal) and the parking/task endpoints come from
    the file rather than random sampling. Returns the raw MAPD payload alongside the
    map so the caller can convert the predefined tasks. In discrete mode every free
    cell is a roadmap node, so the fixed task pickup/delivery endpoints are
    guaranteed reachable. In continuous mode they are NOT sampled nodes, so set
    ``register_task_endpoints=True`` to add the candidate task/parking points to the
    graph (with their exact coords) before the roadmap is built.

    ``roadmap_type`` selects the roadmap (``GraphSampler.generate_roadmap``):
    ``"grid"``, ``"prm"``, ``"rrg"``, ``"cdt"``, ``"voronoi"``, ``"midpoints"``,
    ``"centroids"``, ``"dt"``, or ``"halton..."``. Defaults to ``"grid"`` in discrete
    mode and ``"prm"`` in continuous mode. NOTE for MAPD: ``prm`` connects the
    registered endpoints via KNN so they stay reachable; topology-defining types
    (``cdt``/``voronoi``/``rrg``) rebuild the node set and may not route through
    arbitrary registered endpoints — prefer ``prm``/``halton`` when endpoints are
    fixed.

    ``generator``: a ``TaskGenerator`` whose fleet replaces the file's agents and whose
    task goals are the endpoints to register (``agent_radius`` is then the fleet's
    largest reach).
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
        map_.set_roadmap_config(sample_num=0, num_neighbors=4.0, min_edge_length=0.0, max_edge_length=1.1)
    else:
        # Sparser PRM with shorter edges: the SIPP low-level planner rebuilds its
        # per-node/edge interval graph each call, so fewer nodes/edges keeps it fast
        # while staying dense enough for radius-aware connectivity.
        map_.set_roadmap_config(sample_num=400, num_neighbors=8.0, min_edge_length=0.0, max_edge_length=3.0)

    # World-frame starts so generateRandomNodes registers them as roadmap nodes.
    if generator is not None:
        agents_rt = generator.agents(map_)       # (starts that are not given are drawn on the free map)
    else:
        agents_rt = mapd_agents_to_roadmap_frame(map_, mapd["agents"])
    map_.set_start([a["start"] for a in agents_rt])

    if register_task_endpoints:
        # Continuous roadmaps: candidate task/parking endpoints aren't sampled nodes,
        # so register them (exact coords) via set_goal so generateRandomNodes adds
        # them and generate_map connects them. Cleared after build for a clean figure.
        if generator is not None:
            endpoints = generator.goal_points() + points_to_roadmap_frame(map_, mapd["non_task_endpoints"])
        else:
            endpoints = points_to_roadmap_frame(map_, (
                [p for pair in mapd["goal_locations"] for p in pair]
                + [p for pair in mapd["start_locations"] for p in pair]
                + list(mapd["non_task_endpoints"])
            ))
        map_.set_goal(endpoints)
    else:
        map_.set_goal([])

    # Sample nodes for the requested roadmap, then connect them via the matching
    # builder. generate_roadmap dispatches per type; rrg/planar rebuild the node set, so
    # read the authoritative nodes/road_map back from the map afterwards.
    nodes = map_.generateRandomNodes(generate_grid_nodes=use_discrete_space, roadmap_type=roadmap_type)
    map_.generate_roadmap(roadmap_type)
    nodes = map_.nodes
    road_map = map_.road_map

    if register_task_endpoints:
        map_.set_goal([])  # nodes persist in pos_index_dict; keep the figure clean

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
    tag = f"mapd_{mode}{_map_tag(map_yaml)}"
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


def _run_tasks(
    map_yaml: str,
    tasks_yaml: str,
    agents_yaml: str | None = None,
    use_discrete_space: bool = True,
    horizon: int = 2000,
    seed: int = 42,
    roadmap_type: str | None = None,
    sweep_backend: str = "auto",
    time_limit: float | None = None,
    out_dir_figs: str = "figs/neural_attf",
    out_dir_yaml: str = "path_planning/maps/2d/neural_attf",
    make_gif: bool = True,
):
    """
    Run Neural-ATTF with the fleet and the task stream of ``tasks_yaml`` (agent
    types from ``agents_yaml``; none: point particles) on the map of ``map_yaml``,
    whose ``non_task_endpoints`` are the parking spots. Agents with bodies are
    planned with the SIPP low level, point particles on the grid.
    """
    mode = "discrete" if use_discrete_space else "continuous"
    stem = os.path.splitext(os.path.basename(tasks_yaml))[0]
    tag = f"{stem}_{mode}{_map_tag(map_yaml)}"
    print(f"\n=== Neural-ATTF ({os.path.basename(tasks_yaml)}) | {os.path.basename(map_yaml)} | mode={mode} ===")

    rng = random.Random(seed)
    generator = TaskGenerator.from_yaml(tasks_yaml, agents_yaml, seed=seed)
    map_, agents, mapd, nodes, road_map = _make_map_mapd(
        map_yaml, use_discrete_space, generator.reach(), register_task_endpoints=not use_discrete_space,
        roadmap_type=roadmap_type, sweep_backend=sweep_backend, generator=generator)
    parking_pts = points_to_roadmap_frame(map_, mapd["non_task_endpoints"])
    fleet = {}
    for a in agents:
        fleet[a["type"]] = fleet.get(a["type"], 0) + 1
    print(f"  {len(agents)} agents {fleet} | {len(generator.pool)} tasks in the pool | mode={generator.mode}"
          f" | target={generator.target} | frequency={generator.frequency}")
    map_.set_endpoints(pickups=generator.goal_points(), deliveries=[], parking=parking_pts)

    bodied = generator.agent_types is not None
    _solve_and_save(
        map_=map_,
        agents=agents,
        tasks=generator,
        parking_pts=parking_pts,
        nodes=nodes,
        road_map=road_map,
        tag=tag,
        use_discrete_space=use_discrete_space,
        agent_radius=generator.reach(),
        horizon=horizon,
        rng=rng,
        out_dir_figs=out_dir_figs,
        out_dir_yaml=out_dir_yaml,
        make_gif=make_gif,
        low_level="sipp" if bodied else "grid",
        time_limit=time_limit,
        extra_summary={"map_yaml": map_yaml, "tasks_yaml": tasks_yaml, "agents_yaml": agents_yaml,
                       "task_mode": generator.mode, "agent_types": {a["name"]: a["type"] for a in agents}},
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
    agent: Agent | None = None,
    time_limit: float | None = None,
):
    """
    Shared tail for the Neural-ATTF runners: simulate the token-passing planner
    over the given task stream, write the YAML solution summary, and render the
    static PNG (+ optional GIF). Used by :func:`_run_mapd` (predefined MAPD tasks)
    and the random/continuous runners. Every agent gets the body ``agent`` (an
    ``Agent``, any shape), by default a disc of ``agent_radius`` moving at
    ``velocity`` (0: one time unit per move); ``agent_radius`` 0 and no ``agent``:
    point particles. ``delay_probability`` is the per-step probability that an agent
    is held in place by ``Simulation``. ``tasks``: a list of tasks or a
    ``TaskGenerator``; agents that carry their own ``"agent"`` keep it.
    ``time_limit``: wall seconds (None: the horizon only).
    """
    config = {
        "agent": agent, "agent_radius": agent_radius, "agent_velocity": velocity,
        "heuristic_type": "manhattan" if use_discrete_space else "euclidean",
        "low_level": low_level, "horizon": horizon, "delay_probability": delay_probability,
        "time_limit": time_limit,
    }
    summary = solve_mapd(map_, agents, tasks, parking_pts, config, rng=rng)
    print(
        f"  completed {summary['completed_tasks']}/{summary['n_tasks']} tasks in {summary['steps']} steps "
        f"(planner: {summary['algo_time']:.2f}s, wall: {summary['runtime']:.2f}s)"
        + (" | COLLISION" if summary["collision"] else "")
    )
    summary["tasks"] = tasks.created if isinstance(tasks, TaskGenerator) else tasks
    summary["use_discrete_space"] = use_discrete_space
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

    # if make_gif:
    #     schedule = {"schedule": deepcopy(summary["schedule"])}
    #     gif_path = os.path.join(out_dir_figs, f"neural_attf_{tag}.gif")
    #     vis = Visualizer2D()
    #     vis.animate(
    #         gif_path,
    #         map_,
    #         schedule,
    #         road_map=road_map,
    #         skip_frames=1,
    #         intermediate_frames=1,
    #         speed=3,
    #         radius=agent_radius,
    #         map_frame=False,
    #     )
    #     vis.close()
    #     print(f"  saved animation: {gif_path}")


def main():
    set_global_seed(42)

    # Life-long MAPD instance: agents, parking endpoints and multi-leg tasks are
    # all read from the YAML. Discrete mode so every free cell is a roadmap node
    # and the fixed task endpoints are reachable.
    # NOTE: agent_radius must stay small relative to the warehouse's width-1
    # corridors. The footprint clearance is 2*agent_radius, so a radius >= 0.5
    # makes two agents on adjacent cells (1.0 apart) mutually blocking and
    # deadlocks the corridors. 0.293 -> inflation 1.0 (one cell), still passable.
    # CLI: `python run_neural_attf.py [--map path/to/mapd.yaml] [--tasks tasks.yaml [--agents agents.yaml]]`
    map_yaml = _map_yaml_from_argv()
    tasks_yaml, agents_yaml = _task_files_from_argv()
    if tasks_yaml:
        _run_tasks(map_yaml, tasks_yaml, agents_yaml, use_discrete_space=True, horizon=2000, seed=42, make_gif=False)
        return
    _run_mapd(
        map_yaml=map_yaml,
        use_discrete_space=True,
        horizon=2000,
        seed=42,
        agent_radius=0.293,
        make_gif=False,
    )


if __name__ == "__main__":
    main()
