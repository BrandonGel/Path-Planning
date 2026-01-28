import argparse
from pathlib import Path
from path_planning.data_generation.trajectory_parser import parse_dataset_trajectories
from multiprocessing import cpu_count

if __name__ == "__main__":
    """Main entry point for trajectory parsing."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--paths", nargs='+', type=Path, required=True, help="input folder containing cases")
    parser.add_argument("-v","--visualize", action='store_true', help="visualize the density map")
    parser.add_argument("-w","--num_workers", type=int, default=None, help="number of parallel workers (default: auto-detect CPU cores)")
    args = parser.parse_args()
    
    paths = args.paths
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    
    for idx, path in enumerate(paths):
        parse_dataset_trajectories(path, args.visualize, num_workers=num_workers)
        print(f"Path Trajectory -- [{idx+1}/{len(paths)}] Complete!")
