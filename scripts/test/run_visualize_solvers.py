"""
Visualize agent paths from valid solution files produced by run_all_solvers.py.
Directory structure: base_path / map.../ agents.../ case_{id} / {road_map_type} / ground_truth / perm_{id} / {solver} / solution_radius*_velocity*.yaml
Saves paths.png (and optionally animation.gif) per solver directory.

Examples:
  python scripts/test/run_visualize_solvers.py -s benchmark/test
  python scripts/test/run_visualize_solvers.py -s benchmark/test -y path/to/config.yaml -cm first_n -cn 2 -rmt grid -solver lacam
"""
from path_planning.data_generation.dataset_ground_truth_solve import (
    create_path_parameter_directory,
    create_map,
)
from path_planning.data_generation.dataset_visualize_ground_truth import (
    load_and_visualize_solver_case,
)
import argparse
from pathlib import Path
import yaml
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np


def visualize_solver_worker(args):
    """Worker: load map from ground_truth, then visualize solver result."""
    solver_path, show_static, show_animation, show, map_frame, label = args
    try:
        ground_truth_path = solver_path.parent.parent
        with open(solver_path / "input.yaml", "r") as f:
            inpt = yaml.safe_load(f)
        map_ = create_map(inpt, ground_truth_path, load_graph_sampler=True)
        load_and_visualize_solver_case(
            solver_path, map_, show_static=show_static, show_animation=show_animation, show=show, map_frame=map_frame,
        )
        return True, label, solver_path.name
    except Exception as e:
        return False, label, f"{solver_path}: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--path", type=str, default="/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test", help="base path with results from run_all_solvers.py")
    parser.add_argument("-b", "--bounds", type=float, nargs="+", default=[0, 32.0, 0, 32.0], help="bounds of the map")
    parser.add_argument("-n", "--nb_agents", type=int, default=4, help="number of agents")
    parser.add_argument("-o", "--nb_obstacles", type=float, default=0.1, help="number of obstacles or density")
    parser.add_argument("-p", "--nb_permutations", type=int, default=16, help="number of permutations")
    parser.add_argument("-r", "--resolution", type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-y", "--config", type=str, default="", help="config file to use")
    parser.add_argument("-cm", "--case_mode", type=str, default="first_n", choices=["all", "first_n", "range", "specific"], help="which cases to visualize")
    parser.add_argument("-cn", "--num_cases", type=int, default=3, help="number of cases if case_mode=first_n")
    parser.add_argument("-cr", "--case_range", type=int, nargs=2, default=[0, 16], help="range of cases if case_mode=range")
    parser.add_argument("-cs", "--specific_cases", type=int, nargs="+", default=[0, 5, 10], help="specific case indices if case_mode=specific")
    parser.add_argument("-pv", "--permutation_mode", type=str, default="first_n", choices=["all", "first_n", "range", "specific"], help="which permutations to visualize")
    parser.add_argument("-pn", "--num_permutations", type=int, default=4, help="number of permutations if permutation_mode=first_n")
    parser.add_argument("-pr", "--permutations_range", type=int, nargs=2, default=[0, 16], help="range of permutations if permutation_mode=range")
    parser.add_argument("-ps", "--specific_permutations", type=int, nargs="+", default=[0, 5, 10], help="specific permutation indices if permutation_mode=specific")
    parser.add_argument("-rmt", "--road_map_type", type=str, default="all", choices=["all", "grid", "prm", "planar"], help="road map type to visualize (all = every type under case)")
    parser.add_argument("-solver", "--solver", type=str, default="all", choices=["all", "cbs", "icbs", "lacam", "lacam_random"], help="solver to visualize (all = every solver under perm)")
    parser.add_argument("-sh", "--show", action="store_true", help="show the matplotlib figure")
    parser.add_argument("-ss", "--show_static", dest="show_static", action="store_true", help="save static paths.png")
    parser.add_argument("--no-show-static", dest="show_static", action="store_false", help="disable static path output")
    parser.add_argument("-sa", "--show_animation", action="store_true", help="save animation.gif")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="config file")
    parser.add_argument("-w", "--num_workers", type=int, default=None, help="number of parallel workers")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    base_path = Path(config['path'])
    path = create_path_parameter_directory(base_path, config, dump_config=False)
    if not path.exists():
        print(f"Path {path} does not exist")
        exit(1)

    case_dirs = sorted(
        [d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_") and len(os.listdir(d)) > 0],
        key=lambda x: int(x.name.split("_")[1]),
    )
    print(f"Found {len(case_dirs)} cases")
    if not case_dirs:
        print(f"No cases in {path}")
        exit(1)

    viz_mode = args.case_mode
    n_cases = args.num_cases
    case_range = np.array(args.case_range).astype(int).flatten()
    specific_cases = np.array(args.specific_cases).astype(int).flatten()

    if viz_mode == "all":
        cases_to_viz = case_dirs
    elif viz_mode == "first_n":
        cases_to_viz = case_dirs[:n_cases]
    elif viz_mode == "range":
        cases_to_viz = case_dirs[case_range[0] : case_range[1]]
    elif viz_mode == "specific":
        cases_to_viz = [case_dirs[i] for i in specific_cases if i < len(case_dirs)]
    else:
        cases_to_viz = case_dirs[:3]

    perm_mode = args.permutation_mode
    n_permutations = args.num_permutations
    permutations_range = np.array(args.permutations_range).astype(int).flatten()
    specific_permutations = np.array(args.specific_permutations).astype(int).flatten()

    road_map_filter = None if args.road_map_type == "all" else args.road_map_type
    solver_filter = None if args.solver == "all" else args.solver

    tasks = []
    for case_path in cases_to_viz:
        for road_name in sorted(os.listdir(case_path)):
            road_path = case_path / road_name
            if not road_path.is_dir():
                continue
            if road_map_filter is not None and road_name != road_map_filter:
                continue
            gt_path = road_path / "ground_truth"
            if not gt_path.is_dir():
                continue
            perm_dirs = sorted(
                [d for d in gt_path.iterdir() if d.is_dir() and d.name.startswith("perm_") and len(os.listdir(d)) > 0],
                key=lambda x: int(x.name.split("_")[1]),
            )
            if perm_mode == "all":
                perms_to_viz = perm_dirs
            elif perm_mode == "first_n":
                perms_to_viz = perm_dirs[:n_permutations]
            elif perm_mode == "range":
                perms_to_viz = perm_dirs[permutations_range[0] : permutations_range[1]]
            elif perm_mode == "specific":
                perms_to_viz = [perm_dirs[i] for i in specific_permutations if i < len(perm_dirs)]
            else:
                perms_to_viz = perm_dirs[:3]
            map_frame = True if road_name == "grid" else False
            for perm_dir in perms_to_viz:
                for solver_name in sorted(os.listdir(perm_dir)):
                    solver_path = perm_dir / solver_name
                    if not solver_path.is_dir():
                        continue
                    if solver_filter is not None and solver_name != solver_filter:
                        continue
                    if not list(solver_path.glob("solution_radius*_velocity*.yaml")):
                        continue
                    label = f"{case_path.name}/{road_name}/{perm_dir.name}/{solver_name}"
                    tasks.append((solver_path, args.show_static, args.show_animation, args.show, map_frame, label))

    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    successful = 0
    failed = 0
    print(f"Visualizing {len(tasks)} solver results with {num_workers} workers...")

    if num_workers > 1 and len(tasks) > 1:
        with Pool(processes=num_workers) as pool:
            for success, label, result in tqdm(
                pool.imap_unordered(visualize_solver_worker, tasks),
                total=len(tasks),
                desc="Visualizing",
            ):
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"Error {label}: {result}")
    else:
        for task in tqdm(tasks, desc="Visualizing"):
            success, label, result = visualize_solver_worker(task)
            if success:
                successful += 1
            else:
                failed += 1
                print(f"Error {label}: {result}")

    print("=" * 60)
    print(f"Visualization complete: {successful} succeeded, {failed} failed")
    print("=" * 60)
