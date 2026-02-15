'''
Run trajectory parser for a single path with no visualization and max number of workers
python scripts/generate/run_trajectory_parser.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1

Run trajectory parser for multiple paths with no visualization and max number of workers
Only provide the save path
python scripts/generate/run_trajectory_parser.py -s benchmark/train 

Note: the paths args are the same
python scripts/generate/run_trajectory_parser.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1  benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1

Visualize the density map for a single path with max number of workers
python scripts/generate/run_trajectory_parser.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 -v

Visualize the density map for single paths with 1 worker
python scripts/generate/run_trajectory_parser.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 -w 1
'''
import argparse
from pathlib import Path
from path_planning.data_generation.trajectory_parser import parse_dataset_trajectories
from multiprocessing import cpu_count

if __name__ == "__main__":
    """Main entry point for trajectory parsing."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--paths", default=['benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1'], nargs='+', type=Path, help="input folder containing cases")
    parser.add_argument("-v","--visualize", action='store_true', help="visualize the density map")
    parser.add_argument("-w","--num_workers", type=int, default=None, help="number of parallel workers (default: auto-detect CPU cores)")
    args = parser.parse_args()
    
    paths = args.paths
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    
    for idx, path in enumerate(paths):
        parse_dataset_trajectories(path, args.visualize, num_workers=num_workers)
        print(f"Path Trajectory -- [{idx+1}/{len(paths)}] Complete!")
