import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from path_planning.utils.checker import check_solution_full


def _parse_radius_velocity(sol_file: Path) -> tuple[float, float]:
    """Extract radius and velocity from a solution filename."""
    name = sol_file.name
    radius_str = name.split("radius")[1].split("_")[0]
    velocity_str = name.split("velocity")[1].split(".")[0]
    return float(radius_str), float(velocity_str)


def check_solution_worker(args):
    """
    Worker for multiprocessing.

    Parameters
    ----------
    args : tuple
        (sol_file: Path, radius: float, velocity: float, verbose: bool)

    Returns
    -------
    has_anomaly : bool
    sol_file : Path
    result : dict | None
        Result dict from check_solution_full, or {"error": str} on failure.
    """
    sol_file, radius, velocity, verbose = args
    try:
        with open(sol_file, "r") as f:
            data = yaml.safe_load(f)
        if not data or "schedule" not in data:
            return True, sol_file, {"error": "Missing 'schedule' in YAML"}

        solution = data["schedule"]
        result = check_solution_full(
            solution,
            r=radius,
            is_using_constant_speed=(velocity != 0),
            verbose=verbose,
        )

        has_anomaly = (
            not result["no_time_anomaly"]
            or not result["no_velocity_anomaly"]
            or bool(result["collisions"])
        )
        return has_anomaly, sol_file, result
    except Exception as e:
        return True, sol_file, {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--path",
        type=str,
        default="/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test",
        help="base path with the results from run_all_solvers.py",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=False,
        help="print detailed messages from individual checks",
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
        help="number of parallel workers (default: cpu_count())",
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    verbose = args.verbose

    if not base_path.exists():
        raise FileNotFoundError(f"Base path {base_path} does not exist")

    # Collect all solution files in a single pass
    all_sol_files = list(base_path.rglob("solution_radius*_velocity*.yaml"))
    if not all_sol_files:
        print(f"No solution files found under {base_path}")
        raise SystemExit(0)

    solver_filter = None if args.solver == "all" else args.solver

    filtered_sol_files = []
    for f in all_sol_files:
        # Solver directory is the parent of the solution file
        solver_name = f.parent.name
        if solver_filter is not None and solver_name != solver_filter:
            continue
        filtered_sol_files.append(f)

    if not filtered_sol_files:
        print(f"No solution files found for solver={args.solver} under {base_path}")
        raise SystemExit(0)

    # Build tasks for workers
    tasks = []
    for sol_file in filtered_sol_files:
        radius, velocity = _parse_radius_velocity(sol_file)
        tasks.append((sol_file, radius, velocity, verbose))

    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    num_workers = max(1, num_workers)

    print(
        f"Checking {len(tasks)} solution files "
        f"for solver={args.solver} with {num_workers} worker(s)..."
    )

    anomalies = []

    def handle_result(result, verbose: bool = False):
        has_anomaly, sol_file, info = result
        if not has_anomaly:
            return
        anomalies.append((sol_file, info))
        if verbose:
            print("--------------------------------")
            print(f"Anomaly detected for {sol_file}")
            if info is None:
                print("Unknown error (no info returned)")
            elif "error" in info:
                print(f"Error while checking solution: {info['error']}")
            else:
                # Interpret anomaly details
                if not info.get("no_time_anomaly", True):
                    print("Time anomaly detected")
                if not info.get("no_velocity_anomaly", True):
                    print("Velocity anomaly detected")
                collisions = info.get("collisions", {})
                if collisions:
                    print(f"Collision: {collisions}")
                    # If a collision was found, mark the solution as unsuccessful in-place.
                    try:
                        with open(sol_file, "r") as f:
                            data = yaml.safe_load(f) or {}
                        if isinstance(data, dict):
                            data["success"] = False
                            with open(sol_file, "w") as f:
                                yaml.safe_dump(data, f)
                    except Exception as e:
                        print(f"Warning: failed to update 'success' flag in {sol_file}: {e}")
            print("--------------------------------")

    if num_workers > 1 and len(tasks) > 1:
        with Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(check_solution_worker, tasks),
                total=len(tasks),
                desc="Checking solutions",
            ):
                handle_result(result, verbose=verbose)
    else:
        for task in tqdm(tasks, desc="Checking solutions"):
            result = check_solution_worker(task)
            handle_result(result, verbose=verbose)

    print(f"Finished checking. Total anomalies: {len(anomalies)}")