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
from natsort import os_sorted
import os
import time

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

if __name__ == "__main__":
    # Get folders in natural (Windows-like) order
    map_folders = os.listdir('benchmark/maps')
    sort_nicely(map_folders)

    for map_folder in map_folders:
        print(f"Running CBS on {map_folder}...")

        # Get files in each folder in natural (Windows-like) order
        files = os.listdir(f'benchmark/maps/{map_folder}')
        sort_nicely(files)

        for fname in files:
            print(f"Running CBS on {fname}...")
            map_ =read_graph_sampler_from_yaml(f'benchmark/maps/{map_folder}/{fname}')
            agents = read_agents_from_yaml(f'benchmark/maps/{map_folder}/{fname}')

            map_.inflate_obstacles(radius=0)
            map_.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)

            start = [agent['start'] for agent in agents]
            goal = [agent['goal'] for agent in agents]
            map_.set_start(start)
            map_.set_goal(goal)
            nodes = map_.generateRandomNodes(generate_grid_nodes = True)
            road_map = map_.generate_roadmap(nodes)

            start = [s.current for s in map_.get_start_nodes()]
            goal = [g.current for g in map_.get_goal_nodes()]
            agents ={
                agent['name']: {
                    "start": start[i],
                    "goal": goal[i]
                }
                for i, agent in enumerate(agents)
            }

            env = Environment(map_, agents)

            # Searching
            st = time.time()
            cbs = CBS(env)
            solution = cbs.search()
            print(f"Time taken to search: {time.time() - st} seconds")
            if not solution:
                print(" Solution not found" )

            # Write to output file
            output = dict()
            output["schedule"] = solution
            output["cost"] = env.compute_solution_cost(solution)
            write_to_yaml(output, f"benchmark/solutions/{map_folder}/{fname.replace('.yaml', '_output.yaml')}")

            break