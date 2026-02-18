"""
Run LaCAM for 2D maps.
python scripts/run_lacam.py
"""

from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.common.visualizer.visualizer_3d import Visualizer3D
from path_planning.utils.util import write_to_yaml
from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
from path_planning.multi_agent_planner.centralized.lacam.lacam import LaCAM
from path_planning.multi_agent_planner.centralized.lacam.utility import set_starts_goals_config, is_valid_mapf_solution
from path_planning.utils.util import set_global_seed
from copy import deepcopy
import numpy as np
import os
import time

if __name__ == "__main__":
    os.makedirs("figs/lacam", exist_ok=True)
    os.makedirs("path_planning/maps/2d/lacam", exist_ok=True)

    set_global_seed(42)
    discrete_space = True
    map_ =read_graph_sampler_from_yaml('path_planning/maps/2d/2d.yaml',discrete_space)
    agents = read_agents_from_yaml('path_planning/maps/2d/2d.yaml')
    map_.inflate_obstacles(radius=1)
    map_.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.1, max_edge_len=1.1)
    start = [agent['start'] for agent in agents]
    goal = [agent['goal'] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    nodes = map_.generateRandomNodes(generate_grid_nodes = True)
    road_map = map_.generate_roadmap(nodes)

    # Searching
    starts,goals = set_starts_goals_config(start,goal)
    planner = LaCAM()
    st = time.time()
    solution_config = planner.solve(map_, starts, goals,seed=0,time_limit_ms=30000,verbose=1)
    if not is_valid_mapf_solution(map_, starts, goals, solution_config):
        print("Solution is not valid")
    ft = time.time()
    solution = planner.get_solution_dict(solution_config)
    cost = planner.compute_solution_cost(solution)

    output = dict()
    output["schedule"] = solution
    output["cost"] = cost
    output["runtime"] = ft - st
    write_to_yaml(output, f"path_planning/maps/2d/lacam/solution.yaml")

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_, nodes, road_map)

    # Plot each agent's path
    for agent_name, trajectory in solution.items():
        path = np.array([([point["x"], point["y"]]) for point in trajectory])
        vis.plot_path(path)
    vis.savefig(f"figs/lacam/lacam_2d.png")
    vis.show()
    vis.close()

    # Create animation
    schedule = {"schedule": deepcopy(solution)}
    gif_filename = f"figs/lacam/lacam_2d.gif"
    vis = Visualizer2D()
    vis.animate(
        gif_filename,
        map_,
        schedule,
        road_map=road_map,
        skip_frames=1,
        intermediate_frames=3,
        speed=3,
        radius=0,
        map_frame=True,
    )
    print(f"Animation saved to: {gif_filename}")
    vis.show()
    vis.close()

