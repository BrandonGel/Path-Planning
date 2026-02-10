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
from path_planning.multi_agent_planner.centralized.icbs.icbs import Environment, ICBS
from python_motion_planning.common import TYPES
import os
import time

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

    env = Environment(map_, agents)

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
