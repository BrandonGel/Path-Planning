"""The demo shared by run_cbs / run_icbs / run_ccbs / run_sipp / run_lacam / run_lacam_random: solve the
2d.yaml instance with one solver over a list of (velocity, radius) bodies, in discrete and/or continuous
space, and save each solution (YAML), its plot (PNG) and its animation (GIF).

Each of those scripts also takes ``--tasks tasks.yaml [--agents agents.yaml]``: the fleet (its bodies,
starts and goals) then comes from those files instead of the map's agents and the script's cases, one
run per space. The tasks file has to list the agents with one goal each (see
``path_planning/tasks/task_generator.py``); without ``--agents`` they are point particles."""
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # this repo, ahead of any installed copy

from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.tasks import TaskGenerator
from path_planning.utils.util import (_to_native_yaml, agents_yaml_to_roadmap_frame, read_agents_from_yaml,
                                      read_graph_sampler_from_yaml, set_global_seed, write_to_yaml)

MAP_YAML = "path_planning/maps/2d/2d.yaml"


def run_demo(solver, cases, spaces=(True, False), inflate=lambda radius: radius + np.sqrt(2) / 2,
             max_iterations=10000, to_roadmap_frame=False):
    """Run ``solver`` on ``MAP_YAML`` for every ``(velocity, radius)`` in ``cases`` and every space in
    ``spaces`` (True: the grid, False: a 1000-node PRM). ``inflate(radius)`` is the obstacle inflation
    (``None``: no inflation). ``to_roadmap_frame``: convert the YAML starts and goals with
    ``agents_yaml_to_roadmap_frame`` (cell centres on the grid) before placing them. Outputs go to ``path_planning/maps/2d/<solver>/`` and ``figs/<solver>/``; the
    velocity is part of the file names when any case has one."""
    parser = argparse.ArgumentParser(description=f"{solver} on {MAP_YAML}")
    parser.add_argument("--tasks", help="tasks file: the fleet, with one goal per agent (replaces the built-in cases)")
    parser.add_argument("--agents", help="agent types file for --tasks (left out: point particles)")
    files = parser.parse_args()
    if files.agents and not files.tasks:
        parser.error("--agents needs --tasks")
    os.makedirs(f"figs/{solver}", exist_ok=True)
    os.makedirs(f"path_planning/maps/2d/{solver}", exist_ok=True)
    if files.tasks:
        cases = [(0.0, None)]                # one run per space: the fleet's own bodies
    timed = any(velocity > 0 for velocity, _ in cases)

    set_global_seed(42)
    for use_discrete_space in spaces:
        for agent_velocity, agent_radius in cases:
            map_ = read_graph_sampler_from_yaml(MAP_YAML, use_discrete_space=use_discrete_space)
            fleet = TaskGenerator.from_yaml(files.tasks, files.agents, seed=42) if files.tasks else None
            if fleet is not None:
                agent_radius = fleet.reach()              # the largest body, for the inflation and the summary
            if inflate is not None:
                map_.inflate_obstacles(radius=inflate(agent_radius))
            # A fleet: world -> roadmap frame, and starts that are not given are drawn on the free map.
            agents = fleet.mapf_agents(map_) if fleet is not None else read_agents_from_yaml(MAP_YAML)
            if use_discrete_space:
                map_.set_roadmap_config(sample_num=0, num_neighbors=4.0, min_edge_length=0.0, max_edge_length=1.1)
            else:
                map_.set_roadmap_config(sample_num=1000, num_neighbors=13.0, min_edge_length=0.0, max_edge_length=5.1)

            placed = agents_yaml_to_roadmap_frame(map_, agents) if to_roadmap_frame and fleet is None else agents
            map_.set_start([agent["start"] for agent in placed])
            map_.set_goal([agent["goal"] for agent in placed])
            nodes = map_.generateRandomNodes(generate_grid_nodes=use_discrete_space)
            map_.generate_roadmap("prm")
            road_map = map_.road_map            # per-node neighbour lists, as the plots read them
            agents = [{**agent, "start": s.current, "goal": g.current}      # (a fleet's agents keep their bodies)
                      for agent, s, g in zip(agents, map_.start_nodes, map_.goal_nodes)]
            map_.set_collision_sweep()

            label = f"radius {agent_radius}" + (f", velocity {agent_velocity}" if timed else "")
            if fleet is not None:
                label = f"the fleet of {os.path.basename(files.tasks)}"
            solution_summary = solve_mapf(map_, agents, {
                "mapf_solver_name": solver,
                "time_limit": 60,
                "agent_radius": agent_radius,
                "agent_velocity": agent_velocity,
                "max_iterations": max_iterations,
                "heuristic_type": "euclidean",
            })
            print(f"Time taken to search with {label}: {solution_summary['runtime']} seconds")
            if not solution_summary["success"]:
                print(f"Solution not found for {label}")
                continue

            tag = (f"radius{agent_radius}" + (f"_velocity{agent_velocity}" if timed else "")
                   + ("_discrete" if use_discrete_space else "_continuous"))
            if fleet is not None:
                tag = os.path.splitext(os.path.basename(files.tasks))[0] + ("_discrete" if use_discrete_space else "_continuous")
            write_to_yaml(_to_native_yaml(solution_summary), f"path_planning/maps/2d/{solver}/solution_{tag}.yaml")

            solution = solution_summary["schedule"]
            vis = Visualizer2D()
            vis.plot_grid_map(map_)
            vis.plot_road_map(map_, nodes, road_map, map_frame=use_discrete_space)
            for trajectory in solution.values():
                path = np.array([[point["x"], point["y"]] for point in trajectory])
                vis.plot_path(path, map_frame=use_discrete_space)
            vis.savefig(f"figs/{solver}/{solver}_2d_{tag}.png")
            vis.close()

            gif_filename = f"figs/{solver}/{solver}_2d_{tag}.gif"
            vis = Visualizer2D()
            vis.animate(gif_filename, map_, {"schedule": deepcopy(solution)}, road_map=road_map, skip_frames=1,
                        intermediate_frames=3, speed=3, map_frame=use_discrete_space,
                        radius=agent_radius if fleet is None else {a["name"]: fleet.reach(a["name"]) for a in agents})
            print(f"Animation saved to: {gif_filename}")
            vis.close()
