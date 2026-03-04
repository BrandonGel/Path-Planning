import os
import yaml
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _compute_summary(raw_data):
    """Build summary stats from raw_data for one section."""
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
    return summary


def _process_section(args_tuple):
    """Process one (config1, config2) section: scan solution files and return raw_data."""
    base_path, config1, config2 = args_tuple
    base_path = Path(base_path)
    config2_path = base_path / config1 / config2
    raw_data = {}

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

                    sol_files = list(solver_path.glob("*radius*_velocity*.yaml"))
                    if not sol_files:
                        continue
                    for sol_file in sol_files:
                        if 'std0.0' in sol_file.stem:
                            continue
                        if 'std1.0' in sol_file.stem:
                            continue
                        if 'radius1.0' in sol_file.stem:
                            continue
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
                        solution_name = sol_file.stem.split("_radius")[0]
                        total_iterations = data.get("total_iterations", None)
                        metrics = {
                            "path_length": path_length,
                            "flowtime": flowtime,
                            "wait_time": wait_time,
                            "makespan": makespan,
                            "runtime": runtime,
                            "success": success,
                            "total_iterations": total_iterations,

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
                        if perm_name not in raw_data[solver][road_map_type][radius_key][velocity_key][case_name]:
                            raw_data[solver][road_map_type][radius_key][velocity_key][case_name][perm_name] = {}
                        raw_data[solver][road_map_type][radius_key][velocity_key][case_name][perm_name][solution_name] = metrics

    return (config1, config2, raw_data)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--path",
    type=str,
    default="/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test",
    help="file path with the results from run_all_solvers.py",
)
parser.add_argument(
    "-w",
    "--num_workers",
    type=int,
    default=None,
    help="number of parallel workers (default: CPU count - 1)",
)
args = parser.parse_args()
base_path = Path(args.path)
num_workers = args.num_workers if args.num_workers is not None else max(1, cpu_count() - 1)

# Collect all (config1, config2) section pairs
sections = []
for config1 in sorted(os.listdir(base_path)):
    config1_path = base_path / config1
    if not config1_path.is_dir():
        continue
    for config2 in sorted(os.listdir(config1_path)):
        config2_path = config1_path / config2
        if not config2_path.is_dir():
            continue
        if '8' in config2 or '16' in config2 or '32' in config2:
            continue
        sections.append((str(base_path), config1, config2))

# raw_data[map_section][agents_obst_section] = solver -> road_type -> radius -> velocity -> case -> perm -> metrics
# summary[map_section][agents_obst_section] = road_type/solver -> stats
raw_data_by_section = {}
summary_by_section = {}

if sections:
    if num_workers <= 1:
        results = [_process_section(s) for s in tqdm(sections, desc="Sections")]
    else:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_section, sections),
                total=len(sections),
                desc="Sections",
            ))
    for config1, config2, raw_data in results:
        if config1 not in raw_data_by_section:
            raw_data_by_section[config1] = {}
        raw_data_by_section[config1][config2] = raw_data
        summary_by_section.setdefault(config1, {})[config2] = _compute_summary(raw_data)

# Write single raw_data.yaml and summary.yaml at base_path with map_section -> agents_obst_section
raw_file = base_path / "raw_data.yaml"
with open(raw_file, "w") as f:
    yaml.safe_dump({"raw_data": raw_data_by_section}, f, sort_keys=False)
print("Output file saved:", raw_file)

summary_file = base_path / "summary.yaml"
with open(summary_file, "w") as f:
    yaml.safe_dump({"summary": summary_by_section}, f, sort_keys=False)
print("Summary file saved:", summary_file)
