
from path_planning.utils.checker import check_time_anomaly, check_velocity_anomaly, check_collision
from pathlib import Path

from path_planning.gnn.dataset_prune import create_gnn_maps, create_prune_mechanism
import argparse
from multiprocessing import cpu_count
from math import sqrt
from path_planning.data_generation.dataset_ground_truth import create_maps, create_solutions, create_path_parameter_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,32.0,0,32.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=2, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.1, help="number of obstacles or obstacle density")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("-t","--time_limit",type=int, default=60, help="time_limit for the solver in seconds")
    parser.add_argument("-m","--max_iterations",type=int, default=10000, help="max iterations for the solver")

    parser.add_argument("-rf","--run_folder",type=str, default="logs/gatv2_compile/wandb/run-20260217_062327-z447mk1j", help="run folder")
    parser.add_argument("-cp_path","--checkpoint_path",type=str, help="checkpoint path")
    parser.add_argument("-cfg_path","--config_path",type=str, help="config path")
    parser.add_argument("-run_id","--run_id",type=str, help="run_id")
    parser.add_argument("-c","--num_cases",type=int, default=None, help="number of cases to generate")
    parser.add_argument("-v","--verbose",type=bool, default=True, help="verbose")
    parser.add_argument("-prune_mode","--prune_mode",type=str, default='mean', help="prune mode")
    parser.add_argument("-prune_std_scale","--prune_std_scale",type=float, default=1.0, help="prune std scale")
    parser.add_argument("-prune_value","--prune_value",type=float, default=0.5, help="prune value")
    
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
    mapf_solvers = ["cbs", "icbs","lacam","lacam_random", "sipp", "ccbs"]
    road_map_types = ["grid", "prm", "planar"]
    heuristic_types = ["manhattan", "euclidean"]
    agent_radii = [0, sqrt(2)/2, 1.0]
    agent_velocities = [0, 1.0, 2.0]
    prune_mechanism = create_prune_mechanism(args.prune_mode, args.prune_std_scale, args.prune_value)

    # For each road map type: first create maps and permutations, then run all solvers on the same maps
    for road_map_type in road_map_types:
        map_config = {
            "bounds": bounds,
            "resolution": args.resolution,
            "nb_agents": args.nb_agents,
            "nb_obstacles": args.nb_obstacles,
            "road_map_type": road_map_type,
        }
        path = create_path_parameter_directory(base_path, map_config)
        num_cases = len(list(path.glob("case_*"))) if path.exists() else 0
        print(f"Running Road Map Type: {road_map_type} with {num_cases} cases")
        create_gnn_maps(path, num_cases, map_config, run_folder=args.run_folder, checkpoint_path=args.checkpoint_path, config_path=args.config_path, run_id=args.run_id, road_map_type=road_map_type, verbose=args.verbose, prune_mechanism=prune_mechanism)
