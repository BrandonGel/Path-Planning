import os
import yaml
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.data_generation.dataset_ground_truth import _to_native_yaml

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

def _process_filename(args_tuple):
    """Process one (config1, config2) section: scan solution files and return raw_data."""
    base_path, config1, config2 = args_tuple
    base_path = Path(base_path)
    config2_path = base_path / config1 / config2
    filenames_prop = [] #(filename,key)

    for case_name in sorted(os.listdir(config2_path)):
        case_path = config2_path / case_name
        if not case_path.is_dir() or not case_name.startswith("case_"):
            continue

        for road_map_type in sorted(os.listdir(case_path)):
            road_path = case_path / road_map_type
            if not road_path.is_dir():
                continue
                
            for radius_path in sorted(os.listdir(road_path)):
                gt_path = road_path/radius_path / "ground_truth"
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

                        sol_files = list(solver_path.glob("*_velocity*.yaml"))
                        if not sol_files:
                            continue
                        for sol_file in sol_files:
                            agent_velocity = sol_file.stem.split("velocity")[1].split(".yaml")[0]
                            velocity_key = f"velocity{agent_velocity}"
                            solution_name = sol_file.stem
                            
                            filename_key = (solver,road_map_type,radius_path,velocity_key,case_name,perm_name,solution_name)
                            filenames_prop.append((filename_key,sol_file))

    return filenames_prop

def _process_section_wrapper(item):
    """Unpack tuple for imap_unordered (enables real-time tqdm progress)."""
    return _process_section(item[0], item[1])


def _process_section(key,sol_file):
    raw_data = {}
    solver,road_map_type,radius_path,velocity_key,case_name,perm_name,solution_name = key
    with open(sol_file, "r") as f:
        data = yaml.safe_load(f)

    map_info = sol_file.parent.parent.parent.parent.parent.parent.parent.parent
    bounds = [[0,float(ii)] for ii in map_info.stem.split("_")[0].split("map")[1].split("x")]
    resolution = float(map_info.stem.split("_")[1].split('resolution')[-1])
    map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
    if 'solution_velocity' in solution_name:
        graph_file = sol_file.parent.parent.parent / 'graph_sampler.pkl'
    else:
        headfile = sol_file.stem.split('solution_')[-1].split("_velocity")[0] + '.pkl'
        graph_file = sol_file.parent.parent.parent.parent / 'gnn' / 'gatv2_z447mk1j' / headfile
    map_.load_graph_sampler(graph_file)

    path_length = data.get("path_length", None)
    flowtime = data.get("flowtime", None)
    wait_time = data.get("wait_time", None)
    makespan = data.get("makespan", None)
    runtime = data.get("runtime", None)
    success = data.get("success", False)
    total_iterations = data.get("total_iterations", None)
    num_nodes = len(map_.nodes)
    num_edges = len(map_.edges)
    data["num_nodes"] = len(map_.nodes)
    data["num_edges"] = len(map_.edges)
    with open(sol_file, "w") as f:
        yaml.safe_dump(_to_native_yaml(data), f)


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
    
    if solver not in raw_data:
        raw_data[solver] = {}
    if road_map_type not in raw_data[solver]:
        raw_data[solver][road_map_type] = {}
    if radius_path not in raw_data[solver][road_map_type]:
        raw_data[solver][road_map_type][radius_path] = {}
    if velocity_key not in raw_data[solver][road_map_type][radius_path]:
        raw_data[solver][road_map_type][radius_path][velocity_key] = {}
    if case_name not in raw_data[solver][road_map_type][radius_path][velocity_key]:
        raw_data[solver][road_map_type][radius_path][velocity_key][case_name] = {}
    if perm_name not in raw_data[solver][road_map_type][radius_path][velocity_key][case_name]:
        raw_data[solver][road_map_type][radius_path][velocity_key][case_name][perm_name] = {}
    raw_data[solver][road_map_type][radius_path][velocity_key][case_name][perm_name][solution_name] = metrics

    return raw_data

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
        "-w",
        "--num_workers",
        type=int,
        default=None,
        help="number of parallel workers (default: CPU count - 1)",
    )
    args = parser.parse_args()
    base_path = Path(args.path)
    num_workers = args.num_workers if args.num_workers is not None else max(1, cpu_count() - 1)
    num_workers = 20

    # Collect all (config1, config2) section pairs
    sections = []
    for config1 in sorted(os.listdir(base_path))[1:2]:
        config1_path = base_path / config1
        if not config1_path.is_dir():
            continue
        for config2 in sorted(os.listdir(config1_path), key=lambda x: int(x.split("_")[0].split('agents')[-1])):
            config2_path = config1_path / config2
            if not config2_path.is_dir():
                continue
            sections.append((str(base_path), config1, config2))


    # Collect all (config1, config2) section filenames
    print(sections)
    for section in sections:
        base_path, config1, config2 = section
        section_filenames_prop = _process_filename(section)
        raw_data = {}

        if num_workers > 1 and len(section_filenames_prop) > 1:
            with Pool(processes=num_workers) as pool:
                for raw_data_section in tqdm(
                    pool.imap_unordered(_process_section_wrapper, section_filenames_prop, chunksize=1),
                    total=len(section_filenames_prop),
                    desc="Summarizing filenames",
                ):
                    _deep_merge(raw_data, raw_data_section)
        else:
            for filename_prop in tqdm(section_filenames_prop, desc="Summarizing filenames"):
                raw_data_section = _process_section(filename_prop[0],filename_prop[1])
                _deep_merge(raw_data, raw_data_section)

        # Write single raw_data.yaml and summary.yaml at base_path with map_section -> agents_obst_section
        raw_file = os.path.join(base_path, config1, config2, "raw_data.yaml")
        with open(raw_file, "w") as f:
            yaml.safe_dump(raw_data, f, sort_keys=False)
        print("Output file saved:", raw_file)

    # summary_file = base_path / "summary.yaml"
    # with open(summary_file, "w") as f:
    #     yaml.safe_dump({"summary": summary_by_section}, f, sort_keys=False)
    # print("Summary file saved:", summary_file)