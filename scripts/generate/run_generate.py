'''
Generate a dataset of MAPF instances and their solutions with a single provided path
Only provide the save path
python scripts/generate/run_generate.py

Generate a dataset of MAPF instances and their solutions for 4 agents in a 32x32 grid with 0.1 obstacles and 64 permutations.
python scripts/generate/run_generate.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1

Generate a dataset of MAPF instances and their solutions with multiple provided paths
Note: the paths args are the same
python scripts/generate/run_generate.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1  benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 

Generate a dataset of MAPF instances and their solutions with multiple provided paths with 1 worker
python scripts/generate/run_generate.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1  benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 -w 1

Generate a dataset of MAPF instances and their solutions with multiple provided paths with max number of workers
Generate 1000 random samples + the start/goal samples
Generate edges based on the number of neighbors of each nodes with a minimum edge length of 1e-10 and a maximum edge length of 5+1e-10
Create 1 new (generate_new_graph True) graph sample with the prm roadmap type and the convolution_binary target space as a labeler
Let the generator know that the start and goal are discrete (is_start_goal_discrete True)
Useing Weighted sampling (weighted_sampling True) and sample from the target space with a probability of 50% of the times (samp_from_prob_map_ratio 0.5)
python scripts/generate/run_generate.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 --num_samples 1000 --num_neighbors 13.0 --min_edge_len 1e-10 --max_edge_len 5.0000001 --num_graph_samples 1 --road_map_type prm --target_space convolution_binary --generate_new_graph True --is_start_goal_discrete True --weighted_sampling True --samp_from_prob_map_ratio 0.5
'''

import argparse
from pathlib import Path
from path_planning.data_generation.dataset_generate import generate_graph_samples
from multiprocessing import cpu_count

if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-seed","--seed",type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, nargs='+', default=['benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1'], help="input file containing map and obstacles")
    parser.add_argument("-ds","--use_discrete_space",type=bool, default=False, help="use discrete space")
    parser.add_argument("-ns","--num_samples",type=int, default=1000, help="number of samples")
    parser.add_argument("-nn","--num_neighbors",type=float, default=13.0, help="number of neighbors")
    parser.add_argument("-min_el","--min_edge_len",type=float, default=1e-10, help="minimum edge length")
    parser.add_argument("-max_el","--max_edge_len",type=float, default=5+1e-10, help="maximum edge length")
    parser.add_argument("-ngs","--num_graph_samples",type=int, default=1, help="number of graph samples")
    parser.add_argument("-rmt","--road_map_type",type=str, default='prm', help="road map type")
    parser.add_argument("-ts","--target_space",type=str, default='convolution_binary', help="target space")
    parser.add_argument("-gn","--generate_new_graph",type=bool, default=True, help="generate new graph")
    parser.add_argument("-isg","--is_start_goal_discrete",type=bool, default=True, help="use discrete space for start and goal")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("-ws","--weighted_sampling",type=bool, default=True, help="weighted sampling")
    parser.add_argument("-spfr","--samp_from_prob_map_ratio",type=float, default=0.5, help="sample from prob map ratio")
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
            "target_space": args.target_space,
            "generate_new_graph": args.generate_new_graph,
            "is_start_goal_discrete": args.is_start_goal_discrete,
            "weighted_sampling": args.weighted_sampling,
            "samp_from_prob_map_ratio": args.samp_from_prob_map_ratio,
        }
    for file_path in folder_path:
        generate_graph_samples(file_path,config,num_workers=num_workers)
