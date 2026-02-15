'''
2D Scenario
Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
Save the dataset in the benchmark/train folder.
python scripts/run_dataset_gen.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 100

3D Scenario
Generate a dataset of MAPF instances and their solutions for 8 agents in a 32x32x32 grid with 0.1 obstacles and 64 permutations.
Save the dataset in the benchmark/train folder.
python scripts/run_dataset_gen.py -s benchmark/train -b 0 32.0 0 32.0 0 32.0 -n 8 -o 0.1 -p 64 -r 1.0 -c 100
'''

from path_planning.data_generation.dataset_gen import create_solutions,create_path_parameter_directory
import argparse
from pathlib import Path
import yaml
from multiprocessing import cpu_count

if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='benchmark/train', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,32.0,0,32.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.1, help="number of obstacles or obstacle density")
    parser.add_argument("-p","--nb_permutations",type=int, default=64, help="number of permutations")
    parser.add_argument("-pt","--nb_permutations_tries",type=int, default=128, help="number of permutations tries")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-c","--num_cases",type=int, default=1, help="number of cases to generate")
    parser.add_argument("-y","--config",type=str, default='', help="config file to use")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("-t","--timeout",type=int, default=60, help="timeout for the solver in seconds")
    parser.add_argument("-m","--max_attempts",type=int, default=10000, help="max attempts for the solver")
    args = parser.parse_args()
    
    # Convert bounds from flat list to nested list format
    if isinstance(args.bounds, list):
        if len(args.bounds) == 2:
            bounds = [[0, args.bounds[1]], [0, args.bounds[0]]]
        elif len(args.bounds) == 3:
            bounds = [[0, args.bounds[1]], [0, args.bounds[0]], [0, args.bounds[2]]]
        elif len(args.bounds) == 4:
            bounds = [[args.bounds[0], args.bounds[1]], [args.bounds[2], args.bounds[3]]]
        elif len(args.bounds) == 6:
            bounds = [[args.bounds[0], args.bounds[1]], [args.bounds[2], args.bounds[3]], [args.bounds[4], args.bounds[5]]]
        else:
            raise ValueError(f"Invalid bounds: {args.bounds}")
    else:
        bounds = args.bounds  # Use default or from config
    base_path = Path(args.path)
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    
    if args.config != '':
        with open(args.config, 'r') as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        # Override worker settings if specified via command line
        if args.num_workers is not None:
            config["num_workers"] = args.num_workers
        elif "num_workers" not in config:
            config["num_workers"] = cpu_count()
    else:
        config = {
            "bounds": bounds,
            "resolution": args.resolution,
            "nb_agents": args.nb_agents,
            "nb_obstacles": args.nb_obstacles,
            "nb_permutations": args.nb_permutations,
            "num_workers": num_workers,
            "timeout": args.timeout,
            "max_attempts": args.max_attempts,
        }
    path = create_path_parameter_directory(base_path, config)
    create_solutions(path, args.num_cases, config)

