'''
This script will generate a dataset of MAPF instances and their solutions, parse the trajectories, and generate the target space.
- Using the dataset_gen.py script to generate the dataset of MAPF instances and their solutions
- Using the trajectory_parser.py script to parse the trajectories
- Using the target_gen.py script to generate the target space

2D Scenario
Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
Save the dataset in the benchmark/train folder.
Only provide the save path
python scripts/generate/run_all.py -s benchmark/train

Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_all.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 1

Generate & Visualize a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_all.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 1 -v

Generate & Target a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_all.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 100 -ds True -ns 1000 -nn 13.0 -min_el 1e-10 -max_el 5.0000001 -ngs 1 -rmt prm -ts convolution_binary -gn True -isg True -ws True -spfr 0.5

Generate & Visualize & Target a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_all.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 100 -v -ds True -ns 1000 -nn 13.0 -min_el 1e-10 -max_el 5.0000001 -ngs 1 -rmt prm -ts convolution_binary -gn True -isg True -ws True -spfr 0.5
'''

from path_planning.data_generation.dataset_gen import create_solutions,create_path_parameter_directory
import argparse
from pathlib import Path
import yaml
from multiprocessing import cpu_count
from path_planning.data_generation.trajectory_parser import parse_dataset_trajectories
from path_planning.data_generation.target_gen import generate_graph_samples

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
    parser.add_argument("-v","--visualize",action='store_true', help="visualize the density map")
    parser.add_argument("-ds","--use_discrete_space",type=bool, default=False, help="use discrete space")
    parser.add_argument("-ns","--num_samples",type=int, default=1000, help="number of samples")
    parser.add_argument("-nn","--num_neighbors",type=float, default=13.0, help="number of neighbors")
    parser.add_argument("-min_el","--min_edge_len",type=float, default=1e-10, help="minimum edge length")
    parser.add_argument("-max_el","--max_edge_len",type=float, default=5+1e-10, help="maximum edge length")
    parser.add_argument("-ngs","--num_graph_samples",type=int, default=1, help="number of graph samples")
    parser.add_argument("-rmt","--road_map_type",type=str, default='prm', help="road map type")
    parser.add_argument("-ts","--target_space",type=str, default='convolution_binary', help="target space")
    parser.add_argument("-gn","--generate_new_graph",type=bool, default=True, help="generate new graph")
    parser.add_argument("-isg","--is_start_goal_discrete",type=bool, default=True, help="use discrete space for start and goal")
    parser.add_argument("-ws","--weighted_sampling",type=bool, default=True, help="weighted sampling")
    parser.add_argument("-spfr","--samp_from_prob_map_ratio",type=float, default=0.5, help="sample from prob map ratio")
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
            data_gen_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    else:
        data_gen_config = {
            "bounds": bounds,
            "resolution": args.resolution,
            "nb_agents": args.nb_agents,
            "nb_obstacles": args.nb_obstacles,
            "nb_permutations": args.nb_permutations,
            "timeout": args.timeout,
            "max_attempts": args.max_attempts,
        }
    path = create_path_parameter_directory(base_path, data_gen_config)
    create_solutions(path, args.num_cases, data_gen_config,num_workers=num_workers)
    parse_dataset_trajectories(path, args.visualize, num_workers=num_workers)


    target_gen_config = {
            "use_discrete_space": args.use_discrete_space,
            "num_samples": args.num_samples,
            "num_neighbors": args.num_neighbors,
            "min_edge_len": args.min_edge_len,
            "max_edge_len": args.max_edge_len,
            "num_graph_samples": args.num_graph_samples,
            "road_map_type": args.road_map_type,
            "target_space": args.target_space,
            "generate_new_graph": args.generate_new_graph,
            "is_start_goal_discrete": args.is_start_goal_discrete,
            "weighted_sampling": args.weighted_sampling,
            "samp_from_prob_map_ratio": args.samp_from_prob_map_ratio,
    }
    generate_graph_samples(path,target_gen_config,num_workers=num_workers)
