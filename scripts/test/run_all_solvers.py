"""
2D Scenario
Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
Save the dataset in the benchmark/train folder.

Only provide the save path
python scripts/test/run_all_solvers.py -s /home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test -n 4 -p 4 -c 2

Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_ground_truth.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 100

"""

from path_planning.data_generation.dataset_ground_truth_solve import (
    create_solutions,
    create_path_parameter_directory,
)
from path_planning.data_generation.dataset_ground_truth_map import create_maps
import argparse
from pathlib import Path
import yaml
from multiprocessing import cpu_count
import random
import numpy as np
from math import sqrt   
from path_planning.utils.util import set_map_config
# Set seeding so that randomness is deterministic
random.seed(0)
np.random.seed(0)


if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,64.0,0,64.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.05, help="number of obstacles or obstacle density")
    parser.add_argument("-p","--nb_permutations",type=int, default=10, help="number of permutations")
    parser.add_argument("-pt","--nb_permutations_tries",type=int, default=10, help="number of permutations tries")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-rmt","--road_map_types",type=str, nargs='+', default=['grid', 'prm', 'cdt'], help="road map type")
    parser.add_argument("-ar","--agent_radii",nargs='+',type=float, default=[0.0, round(sqrt(2)/4, 3), 1.0], help="agent radius")
    parser.add_argument("-av","--agent_velocities",nargs='+',type=float, default=[0.0, 1.0], help="agent velocity")
    parser.add_argument("-mapf","--mapf_solver_names",type=str, nargs='+', default=["cbs", "icbs", "lacam", "sipp"], choices=["cbs", "icbs", "lacam", "lacam_random", "sipp"], help="MAPF solver to use")
    parser.add_argument("-c","--num_cases",type=int, default=25, help="number of cases to generate")
    parser.add_argument("-y","--config",type=str, default='', help="config file to use")
    parser.add_argument("-t","--time_limit",type=int, default=60, help="time_limit for the solver in seconds")
    parser.add_argument("-m","--max_iterations",type=int, default=10000, help="max iterations for the solver")
    parser.add_argument("-ds","--use_discrete_space",action="store_true",help="use discrete space",)
    parser.add_argument("-dp","--delete_failed_path",action="store_true", help="delete failed path")
    parser.add_argument("-gng","--generate_new_graph",action="store_true", help="generate new graph")
    parser.add_argument("-rs","--resolve_solution",dest="resolve_solution",action="store_true", help="resolve solution")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="config file")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        map_config = yaml.load(f,Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config,args=args)
    num_workers=map_config['num_workers']
    base_path = map_config['path']

    discrete_config = {
            'use_discrete_space': True,
            'sample_num': 0,
            'num_neighbors': 4.0,
            'min_edge_len': 0.1,
            'max_edge_len': 1.1,
            'heuristic_type': 'manhattan',
    }
    continuous_config = {
            'use_discrete_space': False,
            'sample_num': 2000,
            'num_neighbors': 15.0,
            'min_edge_len': 0.1,
            'max_edge_len': 5.1,
            'heuristic_type': 'euclidean',
    }
    agent_radii=args.agent_radii
    road_map_types=args.road_map_types
    mapf_solver_names=args.mapf_solver_names
    agent_velocities=args.agent_velocities
    num_agents = map_config['nb_agents']
    
    # For each road map type: first create maps and permutations, then run all solvers on the same maps
    for road_map_type in road_map_types:  
        for agent_radius in agent_radii:
            # Phase 1: generate map + permutations once (same map for all solvers)
            map_config['agent_radius'] = agent_radius
            map_config['road_map_type'] = road_map_type
            if road_map_type == "grid":
                map_config.update(discrete_config)
            else:
                map_config.update(continuous_config)

            path = create_path_parameter_directory(base_path, map_config)
            print(f"Phase 1: Generating maps for num_agents={num_agents}, road_map_type={road_map_type}, agent_radius={agent_radius}")
            create_maps(path, args.num_cases, map_config, num_workers=num_workers, generate_new_graph= args.generate_new_graph)

            # Phase 2: run each solver on the saved maps
            for solver in mapf_solver_names:
                for agent_velocity in agent_velocities:
                    if 'lacam' in solver and (agent_radius != 0.0 or agent_velocity != 0.0 or road_map_type != "grid"):
                        continue
                    if (solver == 'cbs' or solver == 'icbs') and agent_velocity != 0.0:
                        continue
                    if solver == 'ccbs' and (agent_radius == 0.0 or agent_velocity == 0.0):
                        continue
                    map_config['mapf_solver_name'] = solver
                    map_config['agent_velocity'] = agent_velocity
                    map_config['delete_failed_path'] = args.delete_failed_path
                    map_config['solve_till_success'] = True
                    map_config['resolve_solution'] = True

                    path = create_path_parameter_directory(base_path, map_config)
                    create_solutions(path, args.num_cases, map_config, num_workers=num_workers,verbose=map_config['verbose'])
