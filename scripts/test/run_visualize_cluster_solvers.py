"""
Visualize MAPF solutions solved on cluster-distilled roadmaps (campaign
layout: case_*/perm/perm_j/<solver>/<rmt>/solution_graph_map_<method>_velocity<v>.yaml).

Per selected case x perm x road map type x method, draws the agent paths on
the distilled roadmap (paths PNG + density heatmap, optional GIF animation)
via dataset_visualize_ground_truth.load_and_visualize_case, saving next to
the solution yaml (solution_..._path.png / _heatmap.png / _path.gif).

python scripts/test/run_visualize_cluster_solvers.py -s <root> -n 4 -rmt prm -cmm gnn -cn 1 -pn 1
python scripts/test/run_visualize_cluster_solvers.py -s <root> -n 64 -rmt grid -cmm gnn kmeans -cn 2 -pn 2 -sa
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm

from path_planning.data_generation.dataset_util import (
    create_path_parameter_directory,
    generate_base_case_path,
    generate_cluster_path,
    generate_input_perm_yaml_path,
    generate_mapf_path,
    generate_roadmap_path,
    get_graph_file_path,
    get_solution_file_path,
    get_solution_name_suffix,
)
from path_planning.data_generation.dataset_visualize_ground_truth import (
    load_and_visualize_case,
)

METHOD_GRAPHS = {
    "ctopprm": "graph_map_ctopprm.pkl",
    "kmeans": "graph_map_kmeans.pkl",
    "em": "graph_map_em.pkl",
    "gnn": "graph_map_gnn.pkl",
    # Full source roadmap, no clustering: solutions are the un-suffixed
    # solution_graph_map_velocity*.yaml from run_all_solvers.py.
    "none": "graph_map.pkl",
}


def worker(task):
    perm_path, graph_file, solver, road_map_type, velocity, show_static, show_animation, label = task
    try:
        load_and_visualize_case(
            perm_path, graph_file=graph_file, mapf_solver_name=solver,
            road_map_type=road_map_type, agent_velocity=velocity,
            show_static=show_static, show_animation=show_animation, verbose=False,
        )
        return True, label, ""
    except Exception as e:  # noqa: BLE001 - reported per task
        return False, label, str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test_samples1500', help="dataset root")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,64.0,0,64.0])
    parser.add_argument("-n","--nb_agents",type=int, default=4)
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.025)
    parser.add_argument("-r","--resolution",type=float, default=1.0)
    parser.add_argument("-ar","--agent_radius",type=float, default=0.5)
    parser.add_argument("-rmt","--road_map_types",type=str, nargs='+', default=['grid', 'prm', 'cdt'])
    parser.add_argument("-cmm","--cluster_methods",type=str, nargs='+', default=['none'], choices=['none'] + list(METHOD_GRAPHS))
    parser.add_argument("-solver","--mapf_solver_name",type=str, default="sipp")
    parser.add_argument("-av","--agent_velocity",type=float, default=1.0)
    parser.add_argument("-cm","--case_mode",type=str, default="first_n", choices=["all", "first_n", "specific"])
    parser.add_argument("-cn","--num_cases",type=int, default=1)
    parser.add_argument("-cs","--specific_cases",type=int, nargs='+', default=[0])
    parser.add_argument("-c","--total_cases",type=int, default=25)
    parser.add_argument("-pn","--num_permutations",type=int, default=1, help="first N perms per case")
    parser.add_argument("-ss","--show_static",dest="show_static",action="store_true", help="save paths PNG + heatmap (default on)")
    parser.add_argument("--no-show-static",dest="show_static",action="store_false")
    parser.add_argument("-sa","--show_animation",action="store_true", help="also save animation GIF (slow)")
    parser.add_argument("-w","--num_workers",type=int, default=None)
    parser.set_defaults(show_static=True)
    args = parser.parse_args()

    map_config = {
        "bounds": [[args.bounds[0], args.bounds[1]], [args.bounds[2], args.bounds[3]]],
        "resolution": args.resolution,
        "nb_agents": args.nb_agents,
        "nb_obstacles": args.nb_obstacles,
        "agent_radius": args.agent_radius,
    }
    path = create_path_parameter_directory(Path(args.path), map_config, dump_config=False)

    if args.case_mode == "all":
        case_ids = list(range(args.total_cases))
    elif args.case_mode == "specific":
        case_ids = args.specific_cases
    else:
        case_ids = list(range(min(args.num_cases, args.total_cases)))

    tasks = []
    for road_map_type in args.road_map_types:
        for case_id in case_ids:
            case_path, map_path = generate_base_case_path(path, case_id, road_map_type)
            cluster_dir = generate_cluster_path(map_path)
            for method in args.cluster_methods:
                pkl_dir = map_path if method == "none" else cluster_dir
                graph_file = get_graph_file_path(pkl_dir, METHOD_GRAPHS[method])
                if not graph_file.exists():
                    continue
                suffix = get_solution_name_suffix(graph_file)
                for perm_id in range(args.num_permutations):
                    perm_path, perm_file = generate_input_perm_yaml_path(case_path, perm_id)
                    if not perm_file.exists():
                        continue
                    roadmap_path = generate_roadmap_path(
                        generate_mapf_path(perm_path, args.mapf_solver_name), road_map_type)
                    sol = get_solution_file_path(roadmap_path, suffix, args.agent_velocity)
                    if not sol.exists():
                        continue
                    label = f"case_{case_id}/{road_map_type}/{method}/perm_{perm_id}"
                    tasks.append((perm_path, graph_file, args.mapf_solver_name,
                                  road_map_type, args.agent_velocity,
                                  args.show_static, args.show_animation, label))
                                  

    if not tasks:
        print("No matching solutions found — run the cluster/GNN solver scripts first.")
        raise SystemExit(0)

    num_workers = args.num_workers if args.num_workers is not None else min(4, cpu_count())
    ok, failed = 0, 0
    print(f"Visualizing {len(tasks)} solutions with {num_workers} workers...")
    if num_workers > 1 and len(tasks) > 1:
        with Pool(processes=num_workers) as pool:
            for success, label, err in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
                ok += success
                if not success:
                    failed += 1
                    print(f"Error {label}: {err}")
    else:
        for task in tqdm(tasks):
            success, label, err = worker(task)
            ok += success
            if not success:
                failed += 1
                print(f"Error {label}: {err}")
    print(f"done: {ok} succeeded, {failed} failed")
