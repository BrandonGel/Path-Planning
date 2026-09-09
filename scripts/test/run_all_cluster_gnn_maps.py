"""
GNN cluster-map pipeline (cluster.md milestone 2, Method A): for each road map
type x agent radius,
(0) make sure the source maps/permutations exist (create_maps — resume-aware,
    existing campaign maps are never regenerated),
(1) distill every case's roadmap with the trained GNN encoder (embeddings ->
    K-means -> latent-space medoid seeds -> CTopPRM 'custom' reconstruction)
    and save maps/<type>/cluster/graph_map_gnn.pkl (+ runtime sidecar).

Solve afterwards with scripts/test/run_all_cluster_gnn_solvers.py.

python scripts/test/run_all_cluster_gnn_maps.py -s <dataset root> -n 4 -rmt prm -c 2 -cf 0.1 -sn 1500 -epoch 60
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

from path_planning.cluster.gnn_cluster_map import (
    create_gnn_cluster_maps,
    summarize_gnn_cluster_runtimes,
)
from path_planning.data_generation.dataset_ground_truth_map import create_maps
from path_planning.data_generation.dataset_ground_truth_solve import (
    create_path_parameter_directory,
)
from path_planning.data_generation.dataset_util import read_gen_config_from_yaml
from path_planning.utils.util import set_map_config

DEFAULT_RUN_FOLDER = ('/home/bho36/Documents/Path-Planning/logs/cluster/gatv2/'
                      'wandb/offline-run-20260908_030628-1wrpdbqy')

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
    parser.add_argument("-cf","--cluster_fraction",type=float, default=0.1, help="cluster count as a fraction of source roadmap nodes (K = max(#endpoints+2, ceil(fraction*num_nodes)))")
    parser.add_argument("-sn","--sample_num",type=int, default=1500, help="number of sampled nodes for continuous road map types (ignored for grid)")
    parser.add_argument("-rf","--run_folder",type=str, default=DEFAULT_RUN_FOLDER, help="trained cluster-GNN wandb run folder (train_cluster.py output)")
    parser.add_argument("-epoch","--epoch",type=int, default=None, help="checkpoint epoch (default: highest; 60 = best validation of the current run)")
    parser.add_argument("-ow","--overwrite_cluster",action="store_true", help="rebuild GNN cluster maps even if they already exist")
    parser.add_argument("-c","--num_cases",type=int, default=25, help="number of cases")
    parser.add_argument("-gng","--generate_new_graph",action="store_true", help="generate new source graph")
    parser.add_argument("-gen_config","--gen_config",type=str, default='config/gen.yaml', help="start/goal placement config")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="config file")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="parallel workers for source-map creation (GNN distillation is sequential)")
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
            map_config['solve_till_success'] = False
            if road_map_type == "grid":
                map_config.update(discrete_config)
            else:
                map_config.update(continuous_config)

            path = create_path_parameter_directory(base_path, map_config)

            # Phase 0: source maps + permutations (resumes if already present)
            create_maps(path, args.num_cases, map_config, num_workers=num_workers,
                        generate_new_graph=args.generate_new_graph)

            # Phase 1: GNN distillation (sequential; raises listing failed
            # cases so downstream graph_files stay complete)
            create_gnn_cluster_maps(path, args.num_cases, map_config,
                                    run_folder=args.run_folder, epoch=args.epoch,
                                    cluster_fraction=args.cluster_fraction,
                                    overwrite=args.overwrite_cluster)

            runtime_file = summarize_gnn_cluster_runtimes(
                path, args.num_cases, map_config)
            print(f"GNN clustering runtime summary saved: {runtime_file}")
