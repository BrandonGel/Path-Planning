import random

random.seed(0)

import numpy as np

np.random.seed(0)

from path_planning.utils.util import read_grid_from_yaml

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.common.visualizer.visualizer_3d import Visualizer3D
from path_planning.utils.util import write_to_yaml
from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
from path_planning.multi_agent_planner.centralized.icbs.icbs import IEnvironment, ICBS
from python_motion_planning.common import TYPES
import os
import time
import numpy as np
from copy import deepcopy

if __name__ == "__main__":
    map_ = read_graph_sampler_from_yaml("path_planning/maps/2d/2d.yaml")
    agents = read_agents_from_yaml("path_planning/maps/2d/2d.yaml")
    map_.inflate_obstacles(radius=1)
    map_.set_parameters(
        sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1
    )

    start = [agent["start"] for agent in agents]
    goal = [agent["goal"] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    nodes = map_.generateRandomNodes(generate_grid_nodes=True)
    road_map = map_.generate_roadmap(nodes)

    start = [s.current for s in map_.get_start_nodes()]
    goal = [g.current for g in map_.get_goal_nodes()]
    agents = [
        {"start": start[i], "name": agent["name"], "goal": goal[i]}
        for i, agent in enumerate(agents)
    ]

    map_.set_constraint_sweep()
    env = IEnvironment(
        map_, agents, radius=1.0, velocity=0.0, use_constraint_sweep=True
    )

    # Searching
    st = time.time()
    icbs = ICBS(env)
    solution = icbs.search()
    print(f"Time taken to search: {time.time() - st} seconds")
    if not solution:
        print(" Solution not found")

    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    write_to_yaml(output, "path_planning/maps/2d/output.yaml")

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_, nodes, road_map)

    # Plot each agent's path
    for agent_name, trajectory in solution.items():
        path = np.array([([point["x"], point["y"]]) for point in trajectory])
        vis.plot_path(path)

    vis.show()

    # from path_planning.common.visualizer.visualizer_2d import Visualizer2D
    # import numpy as np
    # from copy import deepcopy

    # schedule = {"schedule": deepcopy(solution)}

    # # Create animation
    # temp_filename = f"temp.gif"
    # vis = Visualizer2D()
    # vis.animate(
    #     temp_filename,
    #     map_,
    #     schedule,
    #     road_map=road_map,
    #     skip_frames=1,
    #     intermediate_frames=3,
    #     speed=3,
    #     radius=0.9 * 2.0,
    #     map_frame=False,
    # )
    # print(f"Animation saved to: {temp_filename}")
    # vis.show()
