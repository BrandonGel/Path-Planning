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

Generate a dataset of MAPF instances with default settings using timeout (120 seconds) or max attempts (20000).
python scripts/generate/run_all.py -s benchmark/train -t 120 
python scripts/generate/run_all.py -s benchmark/train -m 20000 

Generate a dataset of MAPF instances with default settings using cbs/icbs/lacam/lacam_random algorithm.
python scripts/generate/run_all.py -s benchmark/train -mapf cbs
python scripts/generate/run_all.py -s benchmark/train -mapf icbs
python scripts/generate/run_all.py -s benchmark/train -mapf lacam
python scripts/generate/run_all.py -s benchmark/train -mapf lacam_random

Generate & Target a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_all.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 100 -ds True -ns 1000 -nn 13.0 -min_el 1e-10 -max_el 5.0000001 -ngs 1 -rmt prm -ts convolution_binary -gn True -isg True -ws True -spfr 0.5

Generate & Visualize & Target a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_all.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 64 -r 1.0 -c 100 -v -ds True -ns 1000 -nn 13.0 -min_el 1e-10 -max_el 5.0000001 -ngs 1 -rmt prm -ts convolution_binary -gn True -isg True -ws True -spfr 0.5
'''


import argparse
from pathlib import Path
import yaml
from multiprocessing import cpu_count
from path_planning.data_generation.dataset_ground_truth_solve import create_solutions,create_path_parameter_directory
from path_planning.data_generation.dataset_label import label_dataset
from path_planning.data_generation.dataset_generate import generate_graph_samples
from path_planning.utils.util import set_global_seed
from path_planning.utils.util import set_map_config
from path_planning.data_generation.dataset_ground_truth_map import create_maps

if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-seed","--seed",type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, default='benchmark/train', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,32.0,0,32.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.1, help="number of obstacles or obstacle density")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-ar","--agent_radius",type=float, default=0.0, help="agent radius")
    parser.add_argument("-c","--num_cases",type=int, default=10, help="number of cases to generate")
    parser.add_argument("-p","--nb_permutations",type=int, default=16, help="number of permutations")
    parser.add_argument("-pt","--nb_permutations_tries",type=int, default=32, help="number of permutations tries")
    parser.add_argument("-t","--time_limit",type=float, default=60, help="time_limit for the solver in seconds")
    parser.add_argument("-m","--max_iterations",type=int, default=10000, help="max iterations for the solver")
    parser.add_argument("-mapf","--mapf_solver_name",type=str, default="cbs", choices=["cbs", "icbs", "lacam", "lacam_random"], help="MAPF solver to use")
    parser.add_argument("-gng","--generate_new_graph",type=bool, default=False, help="generate new graph")
    parser.add_argument("-v","--visualize",action='store_true', help="visualize the density map")
    parser.add_argument("-ds","--use_discrete_space",type=bool, default=False, help="use discrete space")
    parser.add_argument("-ns","--num_samples",type=int, default=1000, help="number of samples")
    parser.add_argument("-nn","--num_neighbors",type=float, default=13.0, help="number of neighbors")
    parser.add_argument("-min_el","--min_edge_len",type=float, default=1e-10, help="minimum edge length")
    parser.add_argument("-max_el","--max_edge_len",type=float, default=5+1e-10, help="maximum edge length")
    parser.add_argument("-ngs","--num_graph_samples",type=int, default=1, help="number of graph samples")
    parser.add_argument("-rmt","--road_map_type",type=str, default='prm', help="road map type")
    parser.add_argument("-ts","--target_space",type=str, default='convolution_binary', help="target space")
    parser.add_argument("-isg","--is_start_goal_discrete",type=bool, default=True, help="use discrete space for start and goal")
    parser.add_argument("-ws","--weighted_sampling",type=bool, default=True, help="weighted sampling")
    parser.add_argument("-spfr","--samp_from_prob_map_ratio",type=float, default=0.5, help="sample from prob map ratio")
    parser.add_argument("-w","--num_workers",type=int, default=1, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("-verbose","--verbose",type=bool, default=True, help="verbose")
    args = parser.parse_args()
    

    with open('config/map.yaml', 'r') as f:
        map_config = yaml.load(f,Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config,args=args)
    map_config['agent_velocity'] = 0.0
    map_config['road_map_type'] = "grid"
    map_config['use_discrete_space'] = True
    map_config['solve_till_success'] = True
    map_config['num_samples'] = 0
    map_config['num_neighbors'] = 4.0
    map_config['min_edge_len'] = 0.1
    map_config['max_edge_len'] = 1.1
    map_config['heuristic_type'] = "manhattan"
    base_path = map_config['path']
    seed = map_config['seed']
    bounds = map_config['bounds']
    resolution = map_config['resolution']
    nb_agents = map_config['nb_agents']
    nb_obstacles = map_config['nb_obstacles']
    nb_permutations = map_config['nb_permutations']
    nb_permutations_tries = map_config['nb_permutations_tries']
    time_limit = map_config['time_limit']
    max_iterations = map_config['max_iterations']
    mapf_solver_name = map_config['mapf_solver_name']
    road_map_type = map_config['road_map_type']
    generate_new_graph = map_config['generate_new_graph']
    agent_radius = map_config['agent_radius']
    roadmap_type = map_config['road_map_type']
    num_cases = map_config['num_cases']
    road_map_types = map_config['road_map_type']
    num_workers = map_config['num_workers']
    verbose = map_config['verbose']


    path = create_path_parameter_directory(base_path, map_config)
    print(f"Phase 1: Generating maps for num_agents={nb_agents}, road_map_type={road_map_types}, agent_radius={agent_radius}")
    print("generate_new_graph is ", generate_new_graph, args.generate_new_graph)
    create_maps(path, num_cases, map_config, num_workers=num_workers, generate_new_graph= generate_new_graph, verbose=verbose)

    print(f"Phase 2: MAPF Solver for num_agents={nb_agents}, road_map_type={road_map_types}, agent_radius={agent_radius}")
    path = create_path_parameter_directory(base_path, map_config)
    create_solutions(path, num_cases, map_config, num_workers=num_workers, verbose=verbose)


    label_dataset(path, None, mapf_solver_name, roadmap_type, 0.0, args.visualize, num_workers=num_workers)


    target_gen_config = {
            "seed": seed,
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
