"""
Run benchmark for MAPF algorithms.
Run with the benchmarks with cbs algorithm with default settings (cbs, time limit of 60 seconds, all workers).
python scripts/mapf/run_benchmark.py 

Run with the benchmarks with cbs algorithm with 1 worker.
python scripts/mapf/run_benchmark.py -ca cbs -w 1

Run with the benchmarks with icbs algorithm.
python scripts/mapf/run_benchmark.py -ca icbs

Note: The benchmarks are stored in the benchmark/solutions folder.
Note: This will take a while to run.
"""

from path_planning.utils.util import write_to_yaml
from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
from path_planning.multi_agent_planner.centralized.get_centralized import get_centralized
from path_planning.utils.util import write_to_yaml, set_global_seed
import os
import time
import re
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def run_single_benchmark(args):
    """Run CBS on one map file. Designed for multiprocessing (single tuple arg)."""
    map_folder, fname, time_limit, centralized_alg_name = args
    map_path = f'benchmark/maps/{map_folder}/{fname}'
    try:
        map_ = read_graph_sampler_from_yaml(map_path)
        agents = read_agents_from_yaml(map_path)
    except Exception as e:
        return (map_folder, fname, False, 0.0, str(e))

    map_.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
    start = [agent['start'] for agent in agents]
    goal = [agent['goal'] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    nodes = map_.generateRandomNodes(generate_grid_nodes=True)
    map_.generate_roadmap(nodes)

    start = [s.current for s in map_.get_start_nodes()]
    goal = [g.current for g in map_.get_goal_nodes()]
    agents = [
        {
            "name": agent['name'],
            "start": start[i],
            "goal": goal[i]
        }
        for i, agent in enumerate(agents)
    ]
    CBS,Environment = get_centralized(centralized_alg_name)

    env = Environment(map_, agents, astar_max_iterations=-1)
    st = time.time()
    cbs = CBS(env, time_limit=time_limit)
    solution = cbs.search()
    ft = time.time()
    runtime = ft - st

    out_dir = f"benchmark/solutions/{centralized_alg_name}/{map_folder}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{fname.replace('.yaml', '_output.yaml')}"

    output = dict()
    output["schedule"] = solution if solution else {}
    output["cost"] = env.compute_solution_cost(solution) if solution else None
    output["runtime"] = runtime
    write_to_yaml(output, out_path)

    return (map_folder, fname, solution is not None, runtime, None)


if __name__ == "__main__":
    set_global_seed(42)
    parser = argparse.ArgumentParser(description="Run CBS benchmark (optionally in parallel).")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("--time-limit",type=float,default=60.0,help="CBS time limit per instance in seconds (default: 60)",)
    parser.add_argument("-ca","--centralized_alg_name",type=str, default='cbs', help="centralized algorithm name")
    args = parser.parse_args()
    n_workers = args.num_workers if args.num_workers is not None else cpu_count()
    centralized_alg_name = args.centralized_alg_name

    map_folders = os.listdir('benchmark/maps')
    sort_nicely(map_folders)

    tasks = []
    for map_folder in map_folders:
        folder_path = f'benchmark/maps/{map_folder}'
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        sort_nicely(files)
        for fname in files:
            if fname.endswith('.yaml'):
                tasks.append((map_folder, fname, args.time_limit,centralized_alg_name))

    if not tasks:
        print("No benchmark tasks found.")
        exit(0)

    print(f"Running {centralized_alg_name} on {len(tasks)} instance(s) with {n_workers} worker(s)...")
    results = []
    if n_workers <= 1:
        for t in tqdm(tasks, desc="Running benchmark"):
            results.append(run_single_benchmark(t))
    else:
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(run_single_benchmark, tasks), total=len(tasks), desc="Running benchmark"))

    for map_folder, fname, found, runtime, err in results:
        if err:
            print(f"{centralized_alg_name} on {map_folder}/{fname}: ERROR - {err}")
        elif not found:
            print(f"{centralized_alg_name} on {map_folder}/{fname}: Solution NOT FOUND ({round(runtime, 4)}s)")
        else:
            print(f"{centralized_alg_name} on {map_folder}/{fname}: Solution found in {round(runtime, 4)} seconds")

    n_ok = sum(1 for r in results if r[2])
    print(f"Done: {n_ok}/{len(results)} instances solved.")
