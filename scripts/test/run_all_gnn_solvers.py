"""
2D Scenario
Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
Save the dataset in the benchmark/train folder.

Only provide the save path
python scripts/test/run_all_solvers.py -s /home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test -n 4 -p 4 -c 2

Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_ground_truth.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 100

"""

from path_planning.data_generation.dataset_ground_truth import (
    create_solutions,
    create_maps,
    create_path_parameter_directory,
)
import argparse
from pathlib import Path
import yaml
from multiprocessing import cpu_count
import random
import numpy as np
from math import sqrt   
# Set seeding so that randomness is deterministic
random.seed(0)
np.random.seed(0)
from path_planning.gnn.dataset_prune import create_prune_mechanism, get_prune_mechanism_folder, get_gnn_paths

if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,32.0,0,32.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.1, help="number of obstacles or obstacle density")
    parser.add_argument("-p","--nb_permutations",type=int, default=10, help="number of permutations")
    parser.add_argument("-pt","--nb_permutations_tries",type=int, default=128, help="number of permutations tries")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-c","--num_cases",type=int, default=25, help="number of cases to generate")
    parser.add_argument("-y","--config",type=str, default='', help="config file to use")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("-t","--time_limit",type=int, default=60, help="time_limit for the solver in seconds")
    parser.add_argument("-m","--max_iterations",type=int, default=10000, help="max iterations for the solver")
    parser.add_argument("-ds","--use_discrete_space",type=bool,default=False,help="use discrete space",)
    parser.add_argument("-dp","--delete_failed_path",type=bool, default=False, help="delete failed path")

    parser.add_argument("-prune_mode","--prune_mode",type=str, default='mean', help="prune mode")
    parser.add_argument("-prune_std_scale","--prune_std_scale",type=float, default=0.5, help="prune std scale")
    parser.add_argument("-prune_value","--prune_value",type=float, default=0.5, help="prune value")
    parser.add_argument("-rf","--run_folder",type=str, default="logs/gatv2_compile/wandb/run-20260217_062327-z447mk1j", help="run folder")
    parser.add_argument("-cp_path","--checkpoint_path",type=str, default=None, help="checkpoint path")
    parser.add_argument("-cfg_path","--config_path",type=str, default=None, help="config path")
    parser.add_argument("-run_id","--run_id",type=str, default=None, help="run_id")
    args = parser.parse_args()

    # Convert bounds from flat list to nested list format
    if isinstance(args.bounds, list):
        if len(args.bounds) == 2:
            bounds = [[0, args.bounds[1]], [0, args.bounds[0]]]
        elif len(args.bounds) == 3:
            bounds = [[0, args.bounds[1]], [0, args.bounds[0]], [0, args.bounds[2]]]
        elif len(args.bounds) == 4:
            bounds = [
                [args.bounds[0], args.bounds[1]],
                [args.bounds[2], args.bounds[3]],
            ]
        elif len(args.bounds) == 6:
            bounds = [
                [args.bounds[0], args.bounds[1]],
                [args.bounds[2], args.bounds[3]],
                [args.bounds[4], args.bounds[5]],
            ]
        else:
            raise ValueError(f"Invalid bounds: {args.bounds}")
    else:
        bounds = args.bounds  # Use default or from config
    base_path = Path(args.path)
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()

    # Array with all algorithms to run
    mapf_solvers = ["cbs", "icbs","lacam","lacam_random", "sipp"]
    # mapf_solvers = ["sipp"]
    road_map_types = ["grid","prm", "planar"]    
    # road_map_types = ["prm"]    
    agent_radii = [0, sqrt(2)/4]
    # agent_radii = [sqrt(2)/4]
    agent_velocities = [0, 1.0]
    # agent_velocities = [1.0]

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
    prune_mechanism = create_prune_mechanism(args.prune_mode, args.prune_std_scale, args.prune_value)
    prune_name = get_prune_mechanism_folder(prune_mechanism)
    run_folder = Path(args.run_folder) if args.run_folder is not None else None
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path is not None else None
    config_path = Path(args.config_path) if args.config_path is not None else None
    run_id = args.run_id if args.run_id is not None else None
    _,_,_,gnn_folder_name = get_gnn_paths(run_folder, checkpoint_path, config_path, run_id)

    # For each road map type: first create maps and permutations, then run all solvers on the same maps
    for road_map_type in road_map_types:
        # Phase 1: generate map + permutations once (same map for all solvers)
        map_config = {
            "seed": args.seed,
            "bounds": bounds,
            "resolution": args.resolution,
            "nb_agents": args.nb_agents,
            "nb_obstacles": args.nb_obstacles,
            "nb_permutations": args.nb_permutations,
            "nb_permutations_tries": args.nb_permutations_tries,
            "time_limit": args.time_limit,
            "max_iterations": args.max_iterations,
            "road_map_type": road_map_type,
            "delete_failed_path": args.delete_failed_path,
        }
        if road_map_type == "grid":
            map_config.update(discrete_config)
        else:
            map_config.update(continuous_config)

        path = create_path_parameter_directory(base_path, map_config)
        graph_files = [path / f"case_{case_id}" / road_map_type / "gnn" /gnn_folder_name/ f"graph_sampler_{prune_name}.pkl" for case_id in range(args.num_cases)]
        # Phase 2: run each solver on the saved maps
        for solver in mapf_solvers:
            for agent_radius in agent_radii:
                for agent_velocity in agent_velocities:
                    if 'lacam' in solver and (agent_radius != 0.0 or agent_velocity != 0.0 or road_map_type != "grid"):
                        continue
                    if (solver == 'cbs' or solver == 'icbs') and agent_velocity != 0.0:
                        continue
                    if solver == 'ccbs' and (agent_radius == 0.0 or agent_velocity == 0.0):
                        continue
                    solver_config = {
                        "seed": args.seed,
                        "bounds": bounds,
                        "resolution": args.resolution,
                        "nb_agents": args.nb_agents,
                        "nb_obstacles": args.nb_obstacles,
                        "nb_permutations": args.nb_permutations,
                        "nb_permutations_tries": args.nb_permutations_tries,
                        "time_limit": args.time_limit,
                        "max_iterations": args.max_iterations,
                        "mapf_solver_name": solver,
                        "road_map_type": road_map_type,
                        "discrete_space": args.use_discrete_space,
                        "delete_failed_path": args.delete_failed_path,
                        "agent_radius": agent_radius,
                        "agent_velocity": agent_velocity,
                    }
                    if road_map_type == "grid":
                        solver_config.update(discrete_config)
                    else:
                        solver_config.update(continuous_config)

                    path = create_path_parameter_directory(base_path, solver_config)
                    create_solutions(path, args.num_cases, solver_config, all=False, num_workers=num_workers,graph_files=graph_files,verbose=False)
