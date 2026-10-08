'''
Build a complete cluster-GNN training dataset in one go:

  Phase 0  create the case folders (obstacles, agents, permutations, input.yaml)
           with dataset_ground_truth_map.create_maps (resume-aware: existing
           cases are kept; -gng regenerates them with NEW obstacles/agents)
  Phase 1  for every requested roadmap type, sample a roadmap per case and
           write case_*/sample_cluster/<rmt>/graph_*_{0..3}/ (graph.npz,
           graph_map.pkl, target_cluster.npy) via cluster_dataset_generate

One set of cases is shared by all roadmap types. 'grid' is a true lattice
(-ns is ignored for it); prm/cdt/halton use -ns samples. No MAPF solving.

python scripts/generate/run_cluster_dataset.py \
  -s /home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train_cluster \
  -b 0 64.0 0 64.0 -n 12 -o 0.025 -ar 0.5 -r 1.0 -p 4 -c 100 \
  -rmt grid prm cdt halton -ns 1500 -gbn -bns 1.0 -ngs 1 -w 8
'''

import sys
from pathlib import Path

# Ensure we import the local `path_planning` package (this repo) instead of an
# unrelated installed version from site-packages.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
from multiprocessing import cpu_count

import yaml

from path_planning.data_generation.cluster_dataset_generate import generate_graph_samples
from path_planning.data_generation.dataset_ground_truth_map import create_maps
from path_planning.data_generation.dataset_ground_truth_solve import create_path_parameter_directory
from path_planning.data_generation.dataset_util import (
    generate_cluster_sample_base_path,
    generate_roadmap_path,
    read_gen_config_from_yaml,
)
from path_planning.utils.util import set_map_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed","--seed",type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/debug', help="dataset root; cases land in <root>/map..._resolution.../agents..._obst.../radius.../case_*")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,64.0,0,64.0], help="bounds of the map as x_min x_max y_min y_max")
    parser.add_argument("-n","--nb_agents",type=int, default=12, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.025, help="number of obstacles or obstacle density")
    parser.add_argument("-ar","--agent_radius",type=float, default=0.5, help="agent radius (inflation = radius + sqrt(2)/2 * resolution)")
    parser.add_argument("-r","--resolution",type=float, default=0.25, help="resolution of the map")
    parser.add_argument("-c","--num_cases",type=int, default=10, help="number of cases")
    parser.add_argument("-p","--nb_permutations",type=int, default=4, help="number of start/goal permutations per case (yaml only, not used for training)")
    parser.add_argument("-pt","--nb_permutations_tries",type=int, default=64, help="number of permutations tries")
    parser.add_argument("-rmt","--road_map_types",type=str, nargs='+', default=['grid', 'prm', 'cdt', 'halton'], help="road map types to generate samples for")
    parser.add_argument("-ns","--num_samples",type=int, default=1500, help="number of sampled nodes for prm/cdt/halton (ignored for grid)")
    parser.add_argument("-nn","--num_neighbors",type=float, default=13.0, help="KNN neighbours for the continuous roadmaps")
    parser.add_argument("-min_el","--min_edge_len",type=float, default=0.1, help="minimum edge length (continuous roadmaps)")
    parser.add_argument("-max_el","--max_edge_len",type=float, default=5.1, help="maximum edge length (continuous roadmaps)")
    parser.add_argument("-gbn","--generate_boundary_nodes",action="store_true", help="also register obstacle/map boundary vertices as roadmap nodes")
    parser.add_argument("-bns","--boundary_node_spacing",type=float, default=None, help="resample obstacle/map boundary nodes every this many world units (e.g. the resolution for one node per boundary cell); default: corners/junctions only")
    parser.add_argument("-ngs","--num_graph_samples",type=int, default=1, help="graph samples per case per roadmap type (each also gets 3 rotations)")
    parser.add_argument("-sps","--num_sp_sources",type=int, default=16, help="Dijkstra sources for shortest-path supervision pairs")
    parser.add_argument("-spp","--num_sp_pairs",type=int, default=2048, help="max shortest-path supervision pairs per sample")
    parser.add_argument("-gng","--generate_new_graph",action="store_true", help="regenerate cases (NEW obstacles/agents) and samples even if they exist")
    parser.add_argument("-w","--num_workers",type=int, default=1, help="parallel workers (default: all cores)")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="map config file (also provides sampling_dist_dict for halton)")
    parser.add_argument("-gen_config","--gen_config",type=str, default='config/gen.yaml', help="start/goal placement config")
    parser.add_argument("-verbose","--verbose",action="store_true", help="verbose")
    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers is not None else cpu_count()

    # set_map_config only honours -s when the directory already exists
    # (otherwise it silently falls back to map.yaml's path / benchmark/train).
    Path(args.path).mkdir(parents=True, exist_ok=True)

    with open(args.config, 'r') as f:
        map_config = yaml.load(f, Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config, args=args)
    map_config['gen'] = read_gen_config_from_yaml(args.gen_config)
    map_config['generate_boundary_nodes'] = args.generate_boundary_nodes
    map_config['boundary_node_spacing'] = args.boundary_node_spacing
    map_config['num_workers'] = num_workers
    map_config['agent_velocity'] = 0.0
    map_config['solve_till_success'] = False
    # The case's own roadmap (maps/<rmt>/graph_map.pkl) is not used by the
    # cluster generator, which re-samples from input.yaml; build it as a
    # continuous prm so input.yaml records the continuous settings.
    map_config['road_map_type'] = 'prm'
    map_config['use_discrete_space'] = False
    map_config['sample_num'] = args.num_samples
    map_config['num_neighbors'] = args.num_neighbors
    map_config['min_edge_len'] = args.min_edge_len
    map_config['max_edge_len'] = args.max_edge_len
    map_config['heuristic_type'] = 'euclidean'
    sampling_dist_dict = map_config.get('sampling_dist_dict', {}) or {}

    path = create_path_parameter_directory(map_config['path'], map_config)
    print(f"Phase 0: creating {args.num_cases} cases ({args.nb_agents} agents, obstacle density {args.nb_obstacles}, "
          f"radius {map_config['agent_radius']}) under {path}")
    create_maps(path, args.num_cases, map_config, num_workers=num_workers,
                generate_new_graph=args.generate_new_graph, verbose=args.verbose)

    sample_config = {
        "seed": args.seed,
        "use_discrete_space": False,
        "num_samples": args.num_samples,
        "num_neighbors": args.num_neighbors,
        "min_edge_len": args.min_edge_len,
        "max_edge_len": args.max_edge_len,
        "num_graph_samples": args.num_graph_samples,
        "target_space": "cluster",
        "generate_new_graph": args.generate_new_graph,
        "generate_boundary_nodes": args.generate_boundary_nodes,
        "boundary_node_spacing": args.boundary_node_spacing,
        "num_sp_sources": args.num_sp_sources,
        "num_sp_pairs": args.num_sp_pairs,
        "agent_radius": map_config['agent_radius'],
        "resolution": map_config['resolution'],
        "sampling_dist_dict": sampling_dist_dict,
        "weighted_sampling": False,
    }
    for road_map_type in args.road_map_types:
        print(f"Phase 1: generating cluster samples for road_map_type={road_map_type}")
        generate_graph_samples(path, {**sample_config, "road_map_type": road_map_type}, num_workers=num_workers)

    cases = sorted(d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_"))
    print(f"Done: {len(cases)} cases under {path}")
    for road_map_type in args.road_map_types:
        n = sum(len(list(generate_roadmap_path(generate_cluster_sample_base_path(c), road_map_type).glob("graph_*")))
                for c in cases if generate_roadmap_path(generate_cluster_sample_base_path(c), road_map_type).exists())
        print(f"  {road_map_type}: {n} sample dirs")
