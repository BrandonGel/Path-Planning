"""
Benchmark the CGAL vs Shapely collision-sweep backends across MAPF/MAPD algorithms.

For each sweep-using algorithm (CBS, SIPP, Neural-ATTF), build the SAME instance
twice — once per backend (``sweep_backend="cgal"`` / ``"shapely"``) — run the
planner, and report planner runtime, success, and the number of distinct sweep
queries issued (from the backend's caches; 0 ⇒ the backend was never exercised).

LaCAM is intentionally excluded: it never calls the constraint sweep.

Usage:
    python scripts/mapf/benchmark_sweep_algorithms.py [radius] [time_limit_s]
"""

from __future__ import annotations

import os
import random
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.multi_agent_planner.decentralized.neural_attf.neural_attf import NeuralATTF
from path_planning.multi_agent_planner.decentralized.neural_attf.simulation import Simulation
from path_planning.multi_agent_planner.decentralized.neural_attf.tasks_generator import (
    convert_mapd_tasks,
    gen_mapd_tasks,
)
from path_planning.utils.util import (
    points_to_roadmap_frame,
    read_agents_from_yaml,
    read_graph_sampler_from_yaml,
    set_global_seed,
)
from run_neural_attf import _make_map_mapd

MAPF_YAML = "path_planning/maps/2d/2d.yaml"
MAPD_YAML = "path_planning/maps/2d/2d_mapd.yaml"
BACKENDS = ["cgal", "shapely"]


def _sweep_query_count(map_) -> int:
    """Number of distinct sweep queries cached by the backend (0 ⇒ never used)."""
    cs = getattr(map_, "constraint_sweep", None)
    if cs is None:
        return 0
    return len(getattr(cs, "overlapping_sweep", {})) + len(getattr(cs, "overlapping_interval_sweep", {}))


def _build_mapf_map(backend: str, agent_radius: float):
    """Continuous-PRM map + agents for 2d.yaml, with the chosen sweep backend."""
    set_global_seed(42)
    map_ = read_graph_sampler_from_yaml(MAPF_YAML, use_discrete_space=False, sweep_backend=backend)
    agents = read_agents_from_yaml(MAPF_YAML)
    map_.inflate_obstacles(radius=agent_radius + np.sqrt(2) / 2)
    map_.set_parameters(sample_num=800, num_neighbors=10.0, min_edge_len=0.0, max_edge_len=4.0)
    map_.set_start([a["start"] for a in agents])
    map_.set_goal([a["goal"] for a in agents])
    nodes = map_.generateRandomNodes(generate_grid_nodes=False)
    map_.generate_roadmap(nodes)
    start = [s.current for s in map_.get_start_nodes()]
    goal = [g.current for g in map_.get_goal_nodes()]
    agents = [{"start": start[i], "name": a["name"], "goal": goal[i]} for i, a in enumerate(agents)]
    map_.set_constraint_sweep()
    return map_, agents


def run_solve_mapf(solver_name: str, backend: str, agent_radius: float, velocity: float, time_limit: float):
    map_, agents = _build_mapf_map(backend, agent_radius)
    cfg = {
        "mapf_solver_name": solver_name,
        "time_limit": time_limit,
        "agent_radius": agent_radius,
        "agent_velocity": velocity,
        "max_iterations": 10000,
        "heuristic_type": "euclidean",
    }
    t0 = time.perf_counter()
    summary = solve_mapf(map_, agents, cfg)
    wall = time.perf_counter() - t0
    return {
        "runtime": float(summary.get("runtime", wall)),
        "success": bool(summary.get("success", False)),
        "sweeps": _sweep_query_count(map_),
        "nodes": len(map_.nodes),
    }


def run_neural_attf(backend: str, agent_radius: float, velocity: float, horizon: int = 400):
    set_global_seed(42)
    rng = random.Random(42)
    map_, agents, mapd, nodes, road_map = _make_map_mapd(
        MAPD_YAML, use_discrete_space=False, agent_radius=agent_radius,
        register_task_endpoints=True, roadmap_type="prm", sweep_backend=backend,
    )
    tf = mapd.get("task_freq", 1.0)
    if isinstance(tf, (list, tuple)):
        tf = tf[0]
    tasks = convert_mapd_tasks(
        gen_mapd_tasks(mapd["start_locations"], mapd["goal_locations"], int(mapd["n_tasks"]), tf, rng=rng),
        point_fn=lambda p: points_to_roadmap_frame(map_, [p])[0],
    )
    parking = points_to_roadmap_frame(map_, mapd["non_task_endpoints"])
    planner = NeuralATTF(
        graph_map=map_, agents=agents, non_task_endpoints=parking, a_star_max_iter=4000,
        alpha=0.001, heuristic_type="euclidean", agent_radius=agent_radius, velocity=velocity,
        low_level="sipp",
    )
    sim = Simulation(tasks=[], agents=agents, delay_probability=0.0, rng=rng)
    target = len(tasks)
    while planner.get_completed_tasks() < target and sim.get_time() < horizon:
        new = [t for t in tasks if int(t["start_time"]) == sim.get_time()]
        if new:
            planner.update_tasks(sim.get_time(), new)
        sim.time_forward(planner)
    return {
        "runtime": float(sim.get_algo_time()),
        "success": planner.get_completed_tasks() == target,
        "sweeps": _sweep_query_count(map_),
        "nodes": len(map_.nodes),
    }


def main():
    radius = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    na_radius = 0.5  # Neural-ATTF: smaller disc keeps the continuous instance feasible

    runners = [
        ("cbs",         lambda be: run_solve_mapf("cbs", be, radius, 0.0, time_limit)),
        ("sipp",        lambda be: run_solve_mapf("sipp", be, radius, 1.0, time_limit)),
        ("neural_attf", lambda be: run_neural_attf(be, na_radius, 1.0)),
    ]

    print(f"=== CGAL vs Shapely sweep backend | MAPF radius={radius} | Neural-ATTF radius={na_radius} ===")
    results = {}
    for name, fn in runners:
        results[name] = {}
        for be in BACKENDS:
            print(f"  running {name:12s} [{be}] ...", flush=True)
            try:
                results[name][be] = fn(be)
            except Exception as e:
                results[name][be] = {"runtime": float("nan"), "success": False, "sweeps": -1, "nodes": 0, "error": repr(e)}

    print("\n{:<12} {:>8} {:>10} {:>9} {:>10} {:>9}".format(
        "algorithm", "backend", "runtime(s)", "success", "#sweeps", "nodes"))
    print("-" * 62)
    for name, _ in runners:
        for be in BACKENDS:
            r = results[name][be]
            print("{:<12} {:>8} {:>10.3f} {:>9} {:>10} {:>9}".format(
                name, be, r["runtime"], str(r["success"]), r["sweeps"], r["nodes"]))
            if "error" in r:
                print(f"             error: {r['error']}")
        c, s = results[name]["cgal"], results[name]["shapely"]
        if c["runtime"] > 0 and s["runtime"] > 0 and not np.isnan(c["runtime"]) and not np.isnan(s["runtime"]):
            print(f"{'':<12} {'speedup':>8} {c['runtime']/s['runtime']:>9.2f}x  (cgal/shapely)")
        print()


if __name__ == "__main__":
    main()
