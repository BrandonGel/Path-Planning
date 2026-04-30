"""
Run LaCAM for 2D maps.
python scripts/mapf/run_lacam_random.py
"""

from path_planning.utils.util import set_global_seed
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.utils.util import write_to_yaml
from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.utils.util import _to_native_yaml
import numpy as np
import os
from copy import deepcopy

if __name__ == "__main__":
    os.makedirs("figs/lacam_random", exist_ok=True)
    os.makedirs("path_planning/maps/2d/lacam_random", exist_ok=True)

    set_global_seed(42)
    use_discrete_space = True
    for agent_radius in [0.0]:
        map_ = read_graph_sampler_from_yaml("path_planning/maps/2d/2d.yaml",use_discrete_space=use_discrete_space)
        agents = read_agents_from_yaml("path_planning/maps/2d/2d.yaml")
        map_.inflate_obstacles(radius=1)

        if use_discrete_space:
            map_.set_parameters(
                sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1
            )
        else:
            map_.set_parameters(
                sample_num=1000, num_neighbors=13.0, min_edge_len=0.0, max_edge_len=5.1
            )

        start = [agent["start"] for agent in agents]
        goal = [agent["goal"] for agent in agents]
        map_.set_start(start)
        map_.set_goal(goal)
        nodes = map_.generateRandomNodes(generate_grid_nodes=use_discrete_space)
        road_map = map_.generate_roadmap(nodes)

        start = [s.current for s in map_.get_start_nodes()]
        goal = [g.current for g in map_.get_goal_nodes()]
        agents = [
            {"start": start[i], "name": agent["name"], "goal": goal[i]}
            for i, agent in enumerate(agents)
        ]

        # Searching
        mapf_solver_config = {
            "mapf_solver_name":  "lacam_random",
            "time_limit": 60,
            "agent_radius": agent_radius,
            "agent_velocity": 0.0,
            "max_iterations": None,
            "heuristic_type": "euclidean",
        }
        solution_summary = solve_mapf(map_, agents, mapf_solver_config)
        print(f"Time taken to search with radius {agent_radius}: {solution_summary['runtime']} seconds")
        success = solution_summary['success']
        if not success:
            print(f"Solution not found for radius {agent_radius}")
            continue
        # Write to output file
        if use_discrete_space:
            write_to_yaml(_to_native_yaml(solution_summary), f"path_planning/maps/2d/lacam_random/solution_radius{agent_radius}_discrete.yaml")
        else:
            write_to_yaml(_to_native_yaml(solution_summary), f"path_planning/maps/2d/lacam_random/solution_radius{agent_radius}_continuous.yaml")

        vis = Visualizer2D()
        vis.plot_grid_map(map_)
        vis.plot_road_map(map_, nodes, road_map,map_frame=use_discrete_space)

        # Plot each agent's path
        solution = solution_summary["schedule"]
        for agent_name, trajectory in solution.items():
            path = np.array([([point["x"], point["y"]]) for point in trajectory])
            vis.plot_path(path,map_frame=use_discrete_space)
        if use_discrete_space:
            vis.savefig(f"figs/lacam_random/lacam_random_2d_radius{agent_radius}_discrete.png")
        else:
            vis.savefig(f"figs/lacam_random/lacam_random_2d_radius{agent_radius}_continuous.png")
        vis.show()
        vis.close()

        # # Create animation
        schedule = {"schedule": deepcopy(solution)}
        if use_discrete_space:
            gif_filename = f"figs/lacam_random/lacam_random_2d_radius{agent_radius}_discrete.gif"
        else:
            gif_filename = f"figs/lacam_random/lacam_random_2d_radius{agent_radius}_continuous.gif"
        vis = Visualizer2D()
        vis.animate(
            gif_filename,
            map_,
            schedule,
            road_map=road_map,
            skip_frames=1,
            intermediate_frames=3,
            speed=3,
            radius=agent_radius,
            map_frame=use_discrete_space,
        )
        print(f"Animation saved to: {gif_filename}")
        vis.show()
        vis.close()