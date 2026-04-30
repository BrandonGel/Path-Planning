import os
import yaml
from pathlib import Path
import numpy as np
import argparse
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from path_planning.common.environment.map.graph_sampler import GraphSampler

def _deep_merge(target, source):
    """Recursively merge source into target. Prevents shallow update from overwriting nested dicts."""
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
            and not ("path_length" in value or "success" in value)  # metrics dict is leaf
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def compute_summary(raw_data):
    """
    Recursively summarize each solution_* node by averaging over all case/perm leaves.
    Also computes comparable averages (comp_*) on indices where all solutions succeed.
    """
    summary = {}

    def _is_metrics_dict(d):
        return isinstance(d, dict) and "success" in d and "path_length" in d

    def _collect_case_perm_entries(solution_dict):
        # Returns list[(case_key, perm_key, metrics)]
        rows = []
        for case_key, case_data in solution_dict.items():
            if not isinstance(case_data, dict):
                continue
            for perm_key, metrics in case_data.items():
                if _is_metrics_dict(metrics):
                    rows.append((case_key, perm_key, metrics))
        return rows

    def _mean_or_none(values):
        if len(values) == 0:
            return None
        return float(np.mean(values))

    def _std_or_none(values):
        if len(values) == 0:
            return None
        return float(np.std(values))

    def _summarize_solution_group(solution_group):
        # solution_group: {solution_*: {case_*: {perm_*: metrics}}}
        per_solution_rows = {
            solution_key: _collect_case_perm_entries(solution_dict)
            for solution_key, solution_dict in solution_group.items()
            if isinstance(solution_dict, dict)
        }
        if not per_solution_rows:
            return {}

        # Comparable mask: keep (case, perm) pairs that exist for all solutions and all succeed.
        common_keys = None
        success_sets = {}
        for solution_key, rows in per_solution_rows.items():
            key_set = {(case_key, perm_key) for case_key, perm_key, _ in rows}
            common_keys = key_set if common_keys is None else (common_keys & key_set)
            success_sets[solution_key] = {
                (case_key, perm_key)
                for case_key, perm_key, metrics in rows
                if bool(metrics.get("success", False))
            }
        common_success_keys = set(common_keys) if common_keys is not None else set()
        for solution_key in per_solution_rows.keys():
            common_success_keys &= success_sets[solution_key]

        sol_summary = {}
        for solution_key, rows in per_solution_rows.items():
            entries = [metrics for _, _, metrics in rows]
            successes = np.array([bool(e.get("success", False)) for e in entries], dtype=bool)
            comp_entries = [
                metrics
                for case_key, perm_key, metrics in rows
                if (case_key, perm_key) in common_success_keys
            ]

            # perm_success: fraction of cases where all perms succeeded
            case_to_perms = {}
            for case_key, perm_key, metrics in rows:
                case_to_perms.setdefault(case_key, []).append(bool(metrics.get("success", False)))
            perm_success_per_case = [all(perm_successes) for perm_successes in case_to_perms.values()]
            perm_success_rate = float(np.mean(perm_success_per_case)) if perm_success_per_case else 0.0

            def _collect(field, source):
                vals = [e.get(field, None) for e in source]
                return [v for v in vals if v is not None]

            success_entries = [e for e in entries if bool(e.get("success", False))]
            avg_path_length_entries = _collect("path_length", success_entries)
            comp_avg_path_length_entries = _collect("path_length", comp_entries)

            sol_summary[solution_key] = {
                "avg_path_length": _mean_or_none(avg_path_length_entries),
                "std_path_length": _std_or_none(avg_path_length_entries),
                "avg_flowtime": _mean_or_none(_collect("flowtime", success_entries)),
                "std_flowtime": _std_or_none(_collect("flowtime", success_entries)),
                "avg_wait_time": _mean_or_none(_collect("wait_time", success_entries)),
                "std_wait_time": _std_or_none(_collect("wait_time", success_entries)),
                "avg_makespan": _mean_or_none(_collect("makespan", success_entries)),
                "std_makespan": _std_or_none(_collect("makespan", success_entries)),
                "avg_runtime": _mean_or_none(_collect("runtime", success_entries)),
                "std_runtime": _std_or_none(_collect("runtime", success_entries)),
                "avg_total_iterations": _mean_or_none(_collect("total_iterations", success_entries)),
                "std_total_iterations": _std_or_none(_collect("total_iterations", success_entries)),
                "avg_num_nodes": _mean_or_none(_collect("num_nodes", success_entries)),
                "std_num_nodes": _std_or_none(_collect("num_nodes", success_entries)),
                "avg_num_edges": _mean_or_none(_collect("num_edges", success_entries)),
                "std_num_edges": _std_or_none(_collect("num_edges", success_entries)),
                "success_rate": float(np.mean(successes)) if len(successes) > 0 else 0.0,
                "perm_success": perm_success_rate,
                "avg_length": len(avg_path_length_entries),
                "comp_avg_path_length": _mean_or_none(comp_avg_path_length_entries),
                "comp_std_path_length": _std_or_none(comp_avg_path_length_entries),
                "comp_avg_flowtime": _mean_or_none(_collect("flowtime", comp_entries)),
                "comp_std_flowtime": _std_or_none(_collect("flowtime", comp_entries)),
                "comp_avg_wait_time": _mean_or_none(_collect("wait_time", comp_entries)),
                "comp_std_wait_time": _std_or_none(_collect("wait_time", comp_entries)),
                "comp_avg_makespan": _mean_or_none(_collect("makespan", comp_entries)),
                "comp_std_makespan": _std_or_none(_collect("makespan", comp_entries)),
                "comp_avg_runtime": _mean_or_none(_collect("runtime", comp_entries)),
                "comp_std_runtime": _std_or_none(_collect("runtime", comp_entries)),
                "comp_avg_total_iterations": _mean_or_none(_collect("total_iterations", comp_entries)),
                "comp_std_total_iterations": _std_or_none(_collect("total_iterations", comp_entries)),
                "comp_avg_num_nodes": _mean_or_none(_collect("num_nodes", comp_entries)),
                "comp_std_num_nodes": _std_or_none(_collect("num_nodes", comp_entries)),
                "comp_avg_num_edges": _mean_or_none(_collect("num_edges", comp_entries)),
                "comp_std_num_edges": _std_or_none(_collect("num_edges", comp_entries)),
                "comp_avg_length": len(comp_avg_path_length_entries),
            }
        return sol_summary

    def _walk(node, path):
        if not isinstance(node, dict):
            return
        solution_keys = [k for k in node.keys() if isinstance(k, str) and k.startswith("solution_")]
        if solution_keys:
            solution_group = {k: node[k] for k in solution_keys}
            section_key = "/".join(path) if path else "root"
            summary[section_key] = _summarize_solution_group(solution_group)
            return
        for k, v in node.items():
            _walk(v, path + [k])

    _walk(raw_data, [])
    return summary

def extract_keys(sol_file):
    solution_key,velocity = sol_file.stem.split('_velocity')
    velocity_key = 'velocity'+velocity
    road_map_key = sol_file.parent.name
    mapf_solver_key = sol_file.parent.parent.name
    perm_key = sol_file.parent.parent.parent.name
    case_key = sol_file.parent.parent.parent.parent.parent.name
    radius_folder = Path(str(sol_file).split(case_key)[0])
    radius_key = radius_folder.name
    agent_key, obs_key = radius_folder.parent.name.split('_')
    map_key,resolution_key = radius_folder.parent.parent.name.split('_')
    keys = [
        map_key,
        resolution_key,
        agent_key,
        obs_key,
        radius_key,
        mapf_solver_key,
        road_map_key,
        velocity_key,
        solution_key,
        case_key,
        perm_key]
    return keys

def _natural_key(value):
    """
    Build a key for natural sorting:
    case_2 < case_10, perm_3 < perm_12, while preserving regular string sorting.
    """
    if not isinstance(value, str):
        return (0, value)
    parts = re.split(r"(\d+)", value)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)

def _sort_nested_dict(obj):
    """Recursively sort nested dictionaries using natural key order."""
    if isinstance(obj, dict):
        return {k: _sort_nested_dict(v) for k, v in sorted(obj.items(), key=lambda kv: _natural_key(kv[0]))}
    return obj

def process_sol_file(sol_file):
    # Initialize the raw_data
    raw_data = {}
    with open(sol_file, "r") as f:
        data = yaml.safe_load(f)

    # Get keys for the raw_data
    keys = extract_keys(sol_file)

    # Get metrics for the raw_data
    path_length = data.get("path_length", None)
    flowtime = data.get("flowtime", None)
    wait_time = data.get("wait_time", None)
    makespan = data.get("makespan", None)
    runtime = data.get("runtime", None)
    success = data.get("success", False)
    total_iterations = data.get("total_iterations", None)
    num_nodes = data.get("num_nodes", None)
    num_edges = data.get("num_edges", None)
    metrics = {
        "path_length": path_length,
        "flowtime": flowtime,
        "wait_time": wait_time,
        "makespan": makespan,
        "runtime": runtime,
        "success": success,
        "total_iterations": total_iterations,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }
    
    # Update the raw_data with the metrics
    raw_data = {}
    temp_dict = raw_data
    for key in keys:
        if key not in temp_dict:
            temp_dict[key] = {}
        temp_dict = temp_dict[key]
    temp_dict.update(metrics)
    return raw_data

def get_sol_files(base_path, solver_filter):

    if not base_path.exists():
        raise FileNotFoundError(f"Base path {base_path} does not exist")

    # Collect all solution files in a single pass
    all_sol_files = list(base_path.rglob("solution_*.yaml"))
    if not all_sol_files:
        print(f"No solution files found under {base_path}")
        raise SystemExit(0)

    filtered_sol_files = []
    for f in all_sol_files:
        # Solver directory is the parent of the solution file
        solver_name = f.parent.parent.name
        if solver_filter is not None and solver_name != solver_filter:
            continue
        filtered_sol_files.append(f)

    if not filtered_sol_files:
        print(f"No solution files found for solver={args.solver} under {base_path}")
        raise SystemExit(0)
    return filtered_sol_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--path",
        type=str,
        default="/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test",
        help="file path with the results from run_all_solvers.py",
    )
    parser.add_argument(
        "-solver",
        "--solver",
        type=str,
        default="all",
        choices=["all", "cbs", "icbs", "lacam", "lacam_random", "sipp", "ccbs"],
        help="solver to check (all = every solver under perm directories)",
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
    solver_filter = None if args.solver == "all" else args.solver

    # Get all solution files
    filtered_sol_files = get_sol_files(base_path, solver_filter)

    # Process solution files
    raw_data_all = {}
    if num_workers > 1 and len(filtered_sol_files) > 1:
        with Pool(processes=num_workers) as pool:
            for raw_data in tqdm(
                pool.imap_unordered(process_sol_file, filtered_sol_files, chunksize=1),
                total=len(filtered_sol_files),
                desc="Summarizing filenames",
            ):
                _deep_merge(raw_data_all, raw_data)
    else:
        for sol_file in tqdm(filtered_sol_files, desc="Processing solution files"):
            raw_data = process_sol_file(sol_file)
            _deep_merge(raw_data_all, raw_data)

    # Write single raw_data.yaml for current section or leaf folder
    output_raw_file = base_path / f"raw_data_{solver_filter}.yaml"
    raw_data_all = _sort_nested_dict(raw_data_all)
    with open(output_raw_file, "w") as f:
        yaml.safe_dump(raw_data_all, f, sort_keys=False)
    print("Output file saved:", output_raw_file)

    summary = compute_summary(raw_data_all)
    summary_file = base_path / f"summary_{solver_filter}.yaml"
    with open(summary_file, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print("Summary file saved:", summary_file)