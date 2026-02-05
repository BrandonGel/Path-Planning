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
from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from python_motion_planning.common import TYPES
from path_planning.multi_agent_planner.centralized.lacam.lacam import LaCAM
from path_planning.multi_agent_planner.centralized.lacam.utility import set_starts_goals_config, validate_mapf_solution
import os
import time

if __name__ == "__main__":
    discrete_space = True
    map_ =read_graph_sampler_from_yaml('path_planning/maps/2d/2d.yaml',discrete_space)
    agents = read_agents_from_yaml('path_planning/maps/2d/2d.yaml')
    map_.inflate_obstacles(radius=1)
    map_.set_parameters(sample_num=0, num_neighbors=8.0, min_edge_len=0.1, max_edge_len=1.1)
    start = [agent['start'] for agent in agents]
    goal = [agent['goal'] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    nodes = map_.generateRandomNodes(generate_grid_nodes = True)
    # road_map = map_.generate_roadmap(nodes)
    road_map = map_.generate_planar_map(nodes)

    # Searching
    starts,goals = set_starts_goals_config(start,goal)
    planner = LaCAM()
    solution = planner.solve(map_, starts, goals,seed=0,time_limit_ms=1000,verbose=1)
    validate_mapf_solution(map_, starts, goals, solution)
    print(solution)

