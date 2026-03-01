import os
import yaml
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
from path_planning.utils.checker import check_time_anomaly, check_velocity_anomaly, check_collision

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--path",
    type=str,
    default="/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test",
    help="file path with the results from run_all_solvers.py",
)
parser.add_argument(
    "-v",
    "--verbose",
    type=bool,
    default=False,
    help="verbose",
)
args = parser.parse_args()
base_path = Path(args.path)
verbose = args.verbose
# raw_data[map_section][agents_obst_section] = solver -> road_type -> radius -> velocity -> case -> perm -> metrics
# summary[map_section][agents_obst_section] = road_type/solver -> stats
raw_data_by_section = {}
summary_by_section = {}

# Loop through the two-level config folder structure
for config1 in tqdm(sorted(os.listdir(base_path)), desc=f"{base_path}", position=0, leave=True):
    config1_path = base_path / config1
    if not config1_path.is_dir():
        continue

    for config2 in tqdm(sorted(os.listdir(config1_path)), desc=f"{config1_path}", position=1, leave=True):
        config2_path = config1_path / config2
        if not config2_path.is_dir():
            continue

        raw_data = {}

        # config2_path / case_{id} / {road_map_type} / ground_truth / perm_{id} / {solver} / solution_radius*_velocity*.yaml
        for case_name in tqdm(sorted(os.listdir(config2_path)), desc="Case", position=2, leave=False):
            case_path = config2_path / case_name
            if not case_path.is_dir() or not case_name.startswith("case_"):
                continue

            for road_map_type in tqdm(sorted(os.listdir(case_path)), desc="Road", position=3, leave=False):
                road_path = case_path / road_map_type
                if not road_path.is_dir():
                    continue
                gt_path = road_path / "ground_truth"
                if not gt_path.is_dir():
                    continue

                for perm_name in tqdm(sorted(os.listdir(gt_path)), desc="Perm", position=4, leave=False):
                    perm_path = gt_path / perm_name
                    if not perm_path.is_dir() or not perm_name.startswith("perm_"):
                        continue

                    for solver in tqdm(sorted(os.listdir(perm_path)), desc="Solver", position=5, leave=False):
                        solver_path = perm_path / solver
                        if not solver_path.is_dir():
                            continue

                        sol_files = list(solver_path.glob("solution_radius*_velocity*.yaml"))
                        if not sol_files:
                            continue

                        for sol_file in tqdm(sol_files, desc="Files", position=6, leave=False):
                            radius = sol_file.name.split("radius")[1].split("_")[0]
                            velocity = sol_file.name.split("velocity")[1].split(".")[0]
                            with open(sol_file, "r") as f:
                                data = yaml.safe_load(f)
                            solution = data['schedule']

                            no_time_anomaly = check_time_anomaly(solution,verbose)
                            no_velocity_anomaly = check_velocity_anomaly(solution,is_using_constant_speed=(velocity != 0),verbose=verbose)
                            collisions = check_collision(solution, float(radius),verbose)
                            if not no_time_anomaly or not no_velocity_anomaly or not len(collisions) == 0:
                                print("--------------------------------")
                                print(f"Anomaly detected for {sol_file}")
                                print(f"Radius: {radius}, Velocity: {velocity}")
                                print(f"Time anomaly: { not no_time_anomaly}")
                                print(f"Velocity anomaly: { not no_velocity_anomaly}")
                                print(f"Collision: {collisions}")
                                print("--------------------------------")
                                collisions = check_collision(solution, float(radius),verbose)