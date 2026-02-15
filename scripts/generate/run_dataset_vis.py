'''
2D Scenario
Visualize a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 16 permutations for the first 3 cases with all workers.
python scripts/generate/run_dataset_vis.py -s benchmark/train

Save the dataset in the benchmark/train folder if path not
python scripts/generate/run_dataset_vis.py -s benchmark/train -b 0 32.0 0 32.0 -n 4 -o 0.1 -p 16 -r 1.0 

Visualize a dataset of MAPF instances and their solutions with a config file.
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml

Visualize a dataset of MAPF instances and their solutions with all cases, first_n, range, and specific modes.
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm all
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm first_n -cn 3
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm range -cr 3 4
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm specific -cs 0 5 10

Visualize a dataset of MAPF instances and their solutions with first_n, range, and specific modes for casese and permutations.
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm first_n -cn 3 -pv all
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm first_n -cn 3 -pv first_n -pn 3
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm range -cr 3 4 -pv range -pr 3 4
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm specific -cs 5 -pv specific -ps 0 5 10

Visualize a dataset of MAPF instances and their solutions with static visualization and animation.
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -ss True 
python scripts/generate/run_dataset_vis.py -s benchmark/train -y benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/config.yaml -cm specific -cs 0 -sa True

'''

from path_planning.data_generation.dataset_gen import create_path_parameter_directory
from path_planning.data_generation.dataset_visualize import load_and_visualize_case
import argparse
from pathlib import Path
import yaml
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np


def visualize_case_worker(args):
    """Worker function for parallel visualization."""
    perm_dir, show_static, show_animation, case_name = args
    try:
        load_and_visualize_case(perm_dir, show_static=show_static, show_animation=show_animation)
        return True, case_name, perm_dir.name
    except Exception as e:
        return False, case_name, f"{perm_dir.name}: {str(e)}"


if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='benchmark/train', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,32.0,0,32.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.1, help="number of obstacles or obstacle density")
    parser.add_argument("-p","--nb_permutations",type=int, default=16, help="number of permutations")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-y","--config",type=str, default='', help="config file to use")
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
    if args.config != '':
        with open(args.config, 'r') as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    else:
        config = {
            "bounds": bounds,
            "resolution": args.resolution,
            "nb_agents": args.nb_agents,
            "nb_obstacles": args.nb_obstacles,
            "nb_permutations": args.nb_permutations,
        }
    path = create_path_parameter_directory(base_path, config,dump_config=False)
    assert os.path.exists(path), f"Path {path} does not exist"

    # Get all case directories
    case_dirs = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_") and len(os.listdir(d)) > 0],
                    key=lambda x: int(x.name.split("_")[1]))
    print(f"Found {len(case_dirs)} cases to visualize")
    if len(case_dirs) == 0:
        print(f"No cases found in {path}")
        exit(1)

    # User can modify this to visualize specific cases
    # Options: "all", "first_n", "range", "specific"
    viz_mode = args.case_mode  # Change this to control which cases to visualize
    n_cases = args.num_cases  # Number of cases if using "first_n"
    case_range = np.array(args.case_range).astype(int).flatten()  # Range if using "range"
    specific_cases = np.array(args.specific_cases).astype(int).flatten()  # Specific case numbers if using "specific"

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

    perm_mode = args.permutation_mode  # Change this to control which permutations to visualize
    n_permutations = args.num_permutations
    permutations_range = np.array(args.permutations_range).astype(int).flatten()  # Range if using "range"
    specific_permutations = np.array(args.specific_permutations).astype(int).flatten()  # Specific permutation numbers if using "specific"

    # Collect all tasks (case + permutation combinations)
    tasks = []
    for case_path in cases_to_viz:
        root = case_path / "ground_truth" if (case_path / "ground_truth").exists() else case_path
        perm_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("perm_") and len(os.listdir(d)) > 0],
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
        for perm_dir in permutations_to_viz:
            tasks.append((perm_dir, args.show_static, args.show_animation, case_path.name))

    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    successful = 0
    failed = 0
    print(f"\nVisualizing {len(tasks)} permutations across {len(cases_to_viz)} cases with {num_workers} workers...")

    if num_workers > 1 and len(tasks) > 1:
        with Pool(processes=num_workers) as pool:
            for success, case_name, result in tqdm(
                pool.imap_unordered(visualize_case_worker, tasks),
                total=len(tasks),
                desc="Visualizing cases",
            ):
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"Error in {case_name}: {result}")
    else:
        for task in tqdm(tasks, desc="Visualizing cases"):
            success, case_name, result = visualize_case_worker(task)
            if success:
                successful += 1
            else:
                failed += 1
                print(f"Error in {case_name}: {result}")

    print("\n" + "="*60)
    print(f"Visualization complete: {successful} succeeded, {failed} failed")
    print("="*60)
