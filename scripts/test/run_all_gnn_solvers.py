"""
2D Scenario
Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
Save the dataset in the benchmark/train folder.

Only provide the save path
python scripts/test/run_all_solvers.py -s /home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test -n 4 -p 4 -c 2

Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_ground_truth.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 100

"""

import sys
from pathlib import Path

# Ensure local workspace package is preferred over installed site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from path_planning.data_generation.dataset_ground_truth_solve import (
    create_solutions,
    create_path_parameter_directory,
)
from path_planning.data_generation.dataset_ground_truth_map import create_maps
import argparse
import yaml
import random
import numpy as np
from math import sqrt   
# Set seeding so that randomness is deterministic
random.seed(0)
np.random.seed(0)
from path_planning.gnn.dataset_prune import create_prune_mechanism, get_prune_mechanism_folder, get_gnn_paths
from path_planning.utils.util import set_map_config
from path_planning.gnn.dataset_prune import read_prune_mechanism_from_yaml
from path_planning.data_generation.dataset_visualize_graph import collect_graph_tasks
from path_planning.gnn.dataset_prune import create_gnn_map_tasks
from path_planning.data_generation.dataset_util import get_graph_file_path, generate_gnn_sampler_path, generate_base_case_path

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

    parser.add_argument("-prune_config","--prune_config",type=str, default='config/prune_config.yaml', help="prune config")
    parser.add_argument("-rf","--run_folder",type=str, default="logs/gatv2_compile/wandb/run-20260217_062327-z447mk1j", help="run folder")
    parser.add_argument("-cp_path","--checkpoint_path",type=str, default=None, help="checkpoint path")
    parser.add_argument("-train","--train_config",type=str, default='config/train.yaml', help="train config file")
    parser.add_argument("-run_id","--run_id",type=str, default=None, help="run_id")
    parser.add_argument("-gng","--generate_new_graph",action="store_true", help="generate new graph")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="config file")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        map_config = yaml.load(f,Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config,args=args)
    num_workers=map_config['num_workers']
    base_path = map_config['path']

    # Array with all algorithms to run
    prune_mechanism = read_prune_mechanism_from_yaml(args.prune_config)
    agent_radii=args.agent_radii
    road_map_types=args.road_map_types
    mapf_solver_names=args.mapf_solver_names
    agent_velocities=args.agent_velocities

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
            'sample_num': 1500,
            'num_neighbors': 13.0,
            'min_edge_len': 0.1,
            'max_edge_len': 5.1,
            'heuristic_type': 'euclidean',
    }
    _, _, _, gnn_folder_name = get_gnn_paths(
        run_folder=args.run_folder,
        checkpoint_path=args.checkpoint_path,
        config_path=args.train_config,
        run_id=args.run_id,
    )
    prune_name = get_prune_mechanism_folder(prune_mechanism)
    pruned_graph_name = f"graph_sampler_{prune_name}.pkl" if len(prune_name) > 0 else "graph_map.pkl"


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

            # Graph files to use for MAPF solving.
            graph_files = []
            for case_id in range(args.num_cases):
                _,map_path = generate_base_case_path(path,case_id,road_map_type)
                gnn_dir = generate_gnn_sampler_path(map_path, gnn_folder_name)
                graph_files.append(get_graph_file_path(gnn_dir, pruned_graph_name))

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
                    create_solutions(
                        path,
                        args.num_cases,
                        map_config,
                        num_workers=num_workers,
                        verbose=False,
                        graph_files=graph_files,
                    )
