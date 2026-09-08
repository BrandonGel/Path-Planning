'''
Generate self-supervised cluster-GNN training samples (cluster.md milestones 2-3).

Per case, writes graph.npz + target_cluster.npy under case_*/sample_cluster/<rmt>/:
  - node features [x, y, START/GOAL, FREE, BOUNDARY] (3-way exclusive one-hot)
  - ('node','to','node') roadmap edges, ('node','approx','node') S/G Dijkstra edges
  - ('node','boundary','node') self-loops (attr = distance to obstacle boundary)
  - sp_pair_index/sp_pair_dist: sampled shortest-path supervision pairs (L_SP)
  - y = d_i^B per node (obstacle-boundary distance; no solver labels needed)

python scripts/generate/run_cluster_generate.py -s /home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map64.0x64.0_resolution1.0/agents4_obst0.025/radius1.0 -ns 1500 -ngs 1 -w 4
'''

import argparse
import sys
from pathlib import Path

# Ensure we import the local `path_planning` package (this repo) instead of an
# unrelated installed version from site-packages.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from path_planning.data_generation.cluster_dataset_generate import generate_graph_samples
from multiprocessing import cpu_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed","--seed",type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, nargs='+', default=['/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train'], help="dataset root(s) containing case_* folders")
    parser.add_argument("-ds","--use_discrete_space",action="store_true", help="use discrete space")
    parser.add_argument("-ns","--num_samples",type=int, default=1500, help="number of sampled nodes")
    parser.add_argument("-nn","--num_neighbors",type=float, default=13.0, help="number of neighbors")
    parser.add_argument("-min_el","--min_edge_len",type=float, default=1e-10, help="minimum edge length")
    parser.add_argument("-max_el","--max_edge_len",type=float, default=5+1e-10, help="maximum edge length")
    parser.add_argument("-ngs","--num_graph_samples",type=int, default=1, help="number of graph samples per case")
    parser.add_argument("-rmt","--road_map_type",type=str, default='prm', help="road map type")
    parser.add_argument("-gng","--generate_new_graph",action="store_true", help="regenerate existing samples")
    parser.add_argument("-no-gbn","--no_generate_boundary_nodes",action="store_true", help="disable boundary nodes (on by default for this generator)")
    parser.add_argument("-sps","--num_sp_sources",type=int, default=16, help="Dijkstra sources for shortest-path supervision pairs")
    parser.add_argument("-spp","--num_sp_pairs",type=int, default=2048, help="max shortest-path supervision pairs per sample")
    parser.add_argument("-ar","--agent_radius",type=float, default=0.5, help="agent radius (inflation)")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="map resolution")
    parser.add_argument("-w","--num_workers",type=int, default=1, help="number of parallel workers")
    args = parser.parse_args()

    folder_path = [Path(p) for p in args.path]
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    config = {
        "seed": args.seed,
        "use_discrete_space": args.use_discrete_space,
        "num_samples": args.num_samples,
        "num_neighbors": args.num_neighbors,
        "min_edge_len": args.min_edge_len,
        "max_edge_len": args.max_edge_len,
        "num_graph_samples": args.num_graph_samples,
        "road_map_type": args.road_map_type,
        "target_space": "cluster",
        "generate_new_graph": args.generate_new_graph,
        "generate_boundary_nodes": not args.no_generate_boundary_nodes,
        "num_sp_sources": args.num_sp_sources,
        "num_sp_pairs": args.num_sp_pairs,
        "agent_radius": args.agent_radius,
        "resolution": args.resolution,
        "weighted_sampling": False,
    }
    for file_path in folder_path:
        generate_graph_samples(file_path, config, num_workers=num_workers)
