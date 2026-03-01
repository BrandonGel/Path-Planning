import os
import yaml
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--path",
    type=str,
    default="benchmark/all",
    help="file path with the results from run_all_solvers.py",
)
args = parser.parse_args()
base_path = Path(args.path)

# raw_data[map_section][agents_obst_section] = solver -> road_type -> radius -> velocity -> case -> perm -> metrics
# summary[map_section][agents_obst_section] = road_type/solver -> stats
raw_data_by_section = {}
summary_by_section = {}

# Loop through the two-level config folder structure
for config1 in sorted(os.listdir(base_path)):
    config1_path = base_path / config1
    if not config1_path.is_dir():
        continue

    for config2 in sorted(os.listdir(config1_path)):
        config2_path = config1_path / config2
        if not config2_path.is_dir():
            continue

        raw_data = {}

        # config2_path / case_{id} / {road_map_type} / ground_truth / perm_{id} / {solver} / solution_radius*_velocity*.yaml
        for case_name in sorted(os.listdir(config2_path)):
            case_path = config2_path / case_name
            if not case_path.is_dir() or not case_name.startswith("case_"):
                continue

            for road_map_type in sorted(os.listdir(case_path)):
                road_path = case_path / road_map_type
                if not road_path.is_dir():
                    continue
                gt_path = road_path / "ground_truth"
                if not gt_path.is_dir():
                    continue

                for perm_name in sorted(os.listdir(gt_path)):
                    perm_path = gt_path / perm_name
                    if not perm_path.is_dir() or not perm_name.startswith("perm_"):
                        continue

                    for solver in sorted(os.listdir(perm_path)):
                        solver_path = perm_path / solver
                        if not solver_path.is_dir():
                            continue

                        sol_files = list(solver_path.glob("solution_radius*_velocity*.yaml"))
                        if not sol_files:
                            continue
                        sol_file = sol_files[0]

                        with open(sol_file, "r") as f:
                            data = yaml.safe_load(f)

                        path_length = data.get("path_length", None)
                        flowtime = data.get("flowtime", None)
                        wait_time = data.get("wait_time", None)
                        makespan = data.get("makespan", None)
                        runtime = data.get("runtime", None)
                        success = data.get("success", False)
                        agent_radius = data.get("agent_radius", 0.0)
                        agent_velocity = data.get("agent_velocity", 0.0)

                        metrics = {
                            "path_length": path_length,
                            "flowtime": flowtime,
                            "wait_time": wait_time,
                            "makespan": makespan,
                            "runtime": runtime,
                            "success": success,
                        }
                        radius_key = f"radius_{agent_radius}"
                        velocity_key = f"velocity_{agent_velocity}"
                        if solver not in raw_data:
                            raw_data[solver] = {}
                        if road_map_type not in raw_data[solver]:
                            raw_data[solver][road_map_type] = {}
                        if radius_key not in raw_data[solver][road_map_type]:
                            raw_data[solver][road_map_type][radius_key] = {}
                        if velocity_key not in raw_data[solver][road_map_type][radius_key]:
                            raw_data[solver][road_map_type][radius_key][velocity_key] = {}
                        if case_name not in raw_data[solver][road_map_type][radius_key][velocity_key]:
                            raw_data[solver][road_map_type][radius_key][velocity_key][case_name] = {}
                        raw_data[solver][road_map_type][radius_key][velocity_key][case_name][perm_name] = metrics

        if config1 not in raw_data_by_section:
            raw_data_by_section[config1] = {}
        raw_data_by_section[config1][config2] = raw_data

        by_solver = {}
        for solver, road_data in raw_data.items():
            for road_map_type, radius_data in road_data.items():
                key = (road_map_type, solver)
                if key not in by_solver:
                    by_solver[key] = []
                for velocity_data in radius_data.values():
                    for case_data in velocity_data.values():
                        for metrics in case_data.values():
                            by_solver[key].append(metrics)

        summary = {}
        for (road_map_type, solver), entries in sorted(by_solver.items()):
            path_lengths = [e["path_length"] for e in entries if e.get("path_length") is not None]
            flowtimes = [e["flowtime"] for e in entries if e.get("flowtime") is not None]
            makespans = [e["makespan"] for e in entries if e.get("makespan") is not None]
            runtimes = [e["runtime"] for e in entries if e.get("runtime") is not None]
            successes = [e.get("success", False) for e in entries]
            summary[f"{road_map_type}/{solver}"] = {
                "avg_path_length": float(np.mean(path_lengths)) if path_lengths else None,
                "avg_flowtime": float(np.mean(flowtimes)) if flowtimes else None,
                "avg_makespan": float(np.mean(makespans)) if makespans else None,
                "avg_runtime": float(np.mean(runtimes)) if runtimes else None,
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "num_runs": len(entries),
                "num_valid_path_length": len(path_lengths),
                "num_valid_flowtime": len(flowtimes),
                "num_valid_makespan": len(makespans),
                "num_valid_runtime": len(runtimes),
            }

        if config1 not in summary_by_section:
            summary_by_section[config1] = {}
        summary_by_section[config1][config2] = summary

# Write single raw_data.yaml and summary.yaml at base_path with map_section -> agents_obst_section
raw_file = base_path / "raw_data.yaml"
with open(raw_file, "w") as f:
    yaml.safe_dump({"raw_data": raw_data_by_section}, f, sort_keys=False)
print("Output file saved:", raw_file)

summary_file = base_path / "summary.yaml"
with open(summary_file, "w") as f:
    yaml.safe_dump({"summary": summary_by_section}, f, sort_keys=False)
print("Summary file saved:", summary_file)
