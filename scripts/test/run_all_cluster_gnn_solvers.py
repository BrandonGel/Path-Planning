"""
Solve MAPF on the GNN-distilled cluster maps produced by
run_all_cluster_gnn_maps.py: for each road map type x agent radius, run each
solver x velocity on maps/<type>/cluster/graph_map_gnn.pkl; solutions land in
perm_j/<solver>/<type>/solution_graph_map_gnn_velocity<v>.yaml — the same
naming convention as the ctopprm/kmeans/em methods, so the analysis notebooks
read them as method name 'gnn'.

python scripts/test/run_all_cluster_gnn_solvers.py -s <dataset root> -n 4 -rmt prm -c 2 -mapf sipp -av 1.0
"""

import sys
from pathlib import Path

# Ensure local workspace package is preferred over installed site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import random

import numpy as np
import yaml

# Set seeding so that randomness is deterministic
random.seed(0)
np.random.seed(0)

from path_planning.cluster.gnn_cluster_map import GNN_GRAPH_NAME
from path_planning.data_generation.dataset_ground_truth_solve import (
    create_path_parameter_directory,
    create_solutions,
)
from path_planning.data_generation.dataset_util import (
    generate_base_case_path,
    generate_cluster_path,
    get_graph_file_path,
    read_gen_config_from_yaml,
)
from path_planning.utils.util import set_map_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test_samples1500', help="dataset root")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,64.0,0,64.0], help="bounds of the map as x_min x_max y_min y_max")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.025, help="number of obstacles or obstacle density")
    parser.add_argument("-p","--nb_permutations",type=int, default=4, help="number of permutations")
    parser.add_argument("-pt","--nb_permutations_tries",type=int, default=64, help="number of permutations tries")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-rmt","--road_map_types",type=str, nargs='+', default=['grid', 'prm', 'cdt', 'halton'], help="road map type")
    parser.add_argument("-ar","--agent_radii",nargs='+',type=float, default=[0.5], help="agent radius")
    parser.add_argument("-av","--agent_velocities",nargs='+',type=float, default=[1.0], help="agent velocity (cbs/icbs only run at 0.0)")
    parser.add_argument("-mapf","--mapf_solver_names",type=str, nargs='+', default=["sipp"], choices=["cbs", "icbs", "lacam", "lacam_random", "sipp"], help="MAPF solver to use")
    parser.add_argument("-sn","--sample_num",type=int, default=1500, help="number of sampled nodes for continuous road map types (ignored for grid)")
    parser.add_argument("-c","--num_cases",type=int, default=25, help="number of cases")
    parser.add_argument("-t","--time_limit",type=int, default=60, help="time_limit for the solver in seconds")
    parser.add_argument("-m","--max_iterations",type=int, default=10000, help="max iterations for the solver")
    parser.add_argument("-dp","--delete_failed_path",action="store_true", help="delete failed path")
    parser.add_argument("-gen_config","--gen_config",type=str, default='config/gen.yaml', help="start/goal placement config")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="config file")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases")
    parser.add_argument("-heurs","--heuristic_types",type=str, default='',choices=['manhattan', 'euclidean','dijkstra'], help="heuristic type")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        map_config = yaml.load(f,Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config,args=args)
    map_config['gen'] = read_gen_config_from_yaml(args.gen_config)
    num_workers=map_config['num_workers']
    base_path = map_config['path']

    discrete_config = {
            'use_discrete_space': True,
            'sample_num': 0,
            'num_neighbors': 4.0,
            'min_edge_len': 0.1,
            'max_edge_len': 1.1,
            'heuristic_type': 'manhattan' if args.heuristic_types == '' else args.heuristic_types,
    }
    continuous_config = {
            'use_discrete_space': False,
            'sample_num': args.sample_num,
            'num_neighbors': 13.0,
            'min_edge_len': 0.1,
            'max_edge_len': 5.1,
            'heuristic_type': 'euclidean' if args.heuristic_types == '' else args.heuristic_types,
    }

    for road_map_type in args.road_map_types:
        for agent_radius in args.agent_radii:
            map_config['agent_radius'] = agent_radius
            map_config['road_map_type'] = road_map_type
            if road_map_type == "grid":
                map_config.update(discrete_config)
            else:
                map_config.update(continuous_config)

            path = create_path_parameter_directory(base_path, map_config)

            graph_files = []
            missing = []
            for case_id in range(args.num_cases):
                _, map_path = generate_base_case_path(path, case_id, road_map_type)
                graph_file = get_graph_file_path(generate_cluster_path(map_path), GNN_GRAPH_NAME)
                graph_files.append(graph_file)
                # Without an existing pkl create_map would silently build a
                # fresh roadmap and save it under the cluster name.
                if not graph_file.exists():
                    missing.append(graph_file)
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)}/{args.num_cases} GNN cluster maps missing for "
                    f"road_map_type={road_map_type} — run run_all_cluster_gnn_maps.py "
                    f"first (e.g. {missing[0]})"
                )

            for solver in args.mapf_solver_names:
                for agent_velocity in args.agent_velocities:
                    if 'lacam' in solver and (agent_radius != 0.0 or agent_velocity != 0.0 or road_map_type != "grid"):
                        continue
                    if (solver == 'cbs' or solver == 'icbs') and agent_velocity != 0.0:
                        continue
                    map_config['mapf_solver_name'] = solver
                    map_config['agent_velocity'] = agent_velocity
                    map_config['delete_failed_path'] = args.delete_failed_path
                    map_config['solve_till_success'] = True
                    map_config['resolve_solution'] = True

                    path = create_path_parameter_directory(base_path, map_config)
                    create_solutions(
                        path,
                        args.num_cases,
                        map_config,
                        num_workers=num_workers,
                        verbose=False,
                        graph_files=graph_files,
                    )
