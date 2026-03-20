'''
2D Scenario
Visualize a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 16 permutations for the first 3 cases with all workers.
python scripts/generate/run_visualize_gt.py -s benchmark/train

Save the dataset in the benchmark/train folder if path not
python scripts/generate/run_visualize_gt.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 16 -r 1.0 

Visualize a dataset of MAPF instances and their solutions with a config file.
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml

Visualize a dataset of MAPF instances and their solutions with all cases, first_n, range, and specific modes.
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm all
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm first_n -cn 3
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm range -cr 3 4
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm specific -cs 0 5 10

Visualize a dataset of MAPF instances and their solutions with first_n, range, and specific modes for casese and permutations.
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm first_n -cn 3 -pv all
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm first_n -cn 3 -pv first_n -pn 3
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm range -cr 3 4 -pv range -pr 3 4
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm specific -cs 5 -pv specific -ps 0 5 10

Visualize a dataset of MAPF instances and their solutions with static visualization and animation.
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -ss True 
python scripts/generate/run_visualize_gt.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm specific -cs 0 -sa True

'''

from path_planning.data_generation.dataset_visualize_ground_truth import load_and_visualize_case
import argparse
from pathlib import Path
import yaml
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from path_planning.utils.util import set_map_config
from path_planning.data_generation.dataset_util import *
from path_planning.data_generation.dataset_visualize_ground_truth import visualize_gt


def get_all_cases(path: Path, case_mode: str = "first_n", num_cases: int = 3, case_range: list[int] = [0, 16], specific_cases: list[int] = [0, 5, 10]):
    # Get all case directories
    case_dirs = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_") and len(os.listdir(d)) > 0],
                    key=lambda x: int(x.name.split("_")[1]))
    print(f"Found {len(case_dirs)} cases to visualize")
    if len(case_dirs) == 0:
        print(f"No cases found in {path}")
        exit(1)

    # User can modify this to visualize specific cases
    # Options: "all", "first_n", "range", "specific"
    viz_mode = case_mode  # Change this to control which cases to visualize
    n_cases = num_cases  # Number of cases if using "first_n"
    case_range = np.array(case_range).astype(int).flatten()  # Range if using "range"
    specific_cases = np.array(specific_cases).astype(int).flatten()  # Specific case numbers if using "specific"

    # Select cases to visualize
    if viz_mode == "all":
        cases_to_viz = case_dirs
    elif viz_mode == "first_n":
        cases_to_viz = case_dirs[:n_cases]
    elif viz_mode == "range":
        cases_to_viz = case_dirs[case_range[0]:case_range[1]]
    elif viz_mode == "specific":
        cases_to_viz = [case_dirs[i] for i in specific_cases if i < len(case_dirs)]
    else:
        cases_to_viz = case_dirs[:3]  # Default to first 3
    return cases_to_viz

def get_all_permutations(cases_to_viz: Path, permutation_mode: str = "first_n", num_permutations: int = 4, permutations_range: list[int] = [0, 16], specific_permutations: list[int] = [0, 5, 10],graph_file_name: str = None):
    perm_mode = permutation_mode  # Change this to control which permutations to visualize
    n_permutations = num_permutations
    permutations_range = np.array(permutations_range).astype(int).flatten()  # Range if using "range"
    specific_permutations = np.array(specific_permutations).astype(int).flatten()  # Specific permutation numbers if using "specific"

    # Collect all tasks (case + permutation combinations)
    tasks = []
    for case_path in cases_to_viz:
        perm_path = generate_perm_base_path(case_path)
        perm_dirs = sorted([d for d in perm_path.iterdir() if d.is_dir() and d.name.startswith("perm_") and len(os.listdir(d)) > 0],
                           key=lambda x: int(x.name.split("_")[1]))
        if perm_mode == "all":
            permutations_to_viz = perm_dirs
        elif perm_mode == "first_n":
            permutations_to_viz = perm_dirs[:n_permutations]
        elif perm_mode == "range":
            permutations_to_viz = perm_dirs[permutations_range[0]:permutations_range[1]]
        elif perm_mode == "specific":
            permutations_to_viz = [perm_dirs[i] for i in specific_permutations if i < len(perm_dirs)]
        else:
            permutations_to_viz = perm_dirs[:3]  # Default to first 3

        map_path = generate_map_path(case_path,road_map_type)
        graph_file = get_graph_file_path(map_path,graph_file_name)
        if not graph_file.exists():
            raise Exception(f"Graph file {graph_file} does not exist")

        for perm_dir in permutations_to_viz:
            tasks.append([case_path,perm_dir, graph_file])
    return tasks
    
if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='benchmark/train', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,32.0,0,32.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.1, help="number of obstacles or obstacle density")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-ar","--agent_radius",type=float, default=0.0, help="agent radius")
    parser.add_argument("-ms","--mapf_solver_name",type=str, default="cbs", help="mapf solver name")
    parser.add_argument("-rmt","--road_map_type",type=str, default="grid", help="road map type")
    parser.add_argument("-av","--agent_velocity",type=float, default=0.0, help="agent velocity")
    parser.add_argument("-cm","--case_mode",type=str, default='first_n', choices=['all', 'first_n', 'range', 'specific'], help="mode to visualize the dataset")
    parser.add_argument("-cn","--num_cases",type=int, default=3, help="number of cases to visualize")
    parser.add_argument("-cr","--case_range",type=int, nargs=2, default=[0, 16], help="range of cases to visualize")
    parser.add_argument("-cs","--specific_cases",type=int, nargs='+', default=[0, 5, 10], help="specific cases to visualize")
    parser.add_argument("-pv","--permutation_mode",type=str, default='first_n', choices=['all', 'first_n', 'range', 'specific'], help="mode to visualize the dataset")
    parser.add_argument("-pn","--num_permutations",type=int, default=4, help="number of permutations to visualize")
    parser.add_argument("-pr","--permutations_range",type=int, nargs=2, default=[0, 16], help="range of permutations to visualize")
    parser.add_argument("-ps","--specific_permutations",type=int, nargs='+', default=[0, 5, 10], help="specific permutations to visualize")
    parser.add_argument("-ss","--show_static",type=bool, default=True, help="show static paths")
    parser.add_argument("-sa","--show_animation",type=bool, default=False, help="show animation")
    parser.add_argument("-w", "--num_workers", type=int, default=None,
                        help="number of parallel workers (default: auto-detect CPU cores)")
    parser.add_argument("-v","--verbose",type=bool, default=False, help="verbose")
    args = parser.parse_args()
    
    with open('config/map.yaml', 'r') as f:
        map_config = yaml.load(f,Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config,args=args)
    base_path = map_config['path']
    path = create_path_parameter_directory(base_path, map_config,dump_config=False)
    mapf_solver_name = map_config['mapf_solver_name']
    road_map_type = map_config['road_map_type']
    agent_velocity = map_config['agent_velocity']
    num_workers = map_config['num_workers'] 
    verbose = map_config['verbose']
    show_static = args.show_static
    show_animation = args.show_animation
    assert os.path.exists(path), f"Path {path} does not exist"

    cases_to_viz = get_all_cases(path, case_mode=args.case_mode, num_cases=args.num_cases, case_range=args.case_range, specific_cases=args.specific_cases)
    tasks = get_all_permutations(cases_to_viz, permutation_mode=args.permutation_mode, num_permutations=args.num_permutations, permutations_range=args.permutations_range, specific_permutations=args.specific_permutations)

    for task in tasks:
        task.extend([mapf_solver_name, road_map_type, agent_velocity, show_static, show_animation, verbose])

    visualize_gt(tasks, num_workers=num_workers)
