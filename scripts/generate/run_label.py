'''
Run trajectory parser for a single path with no visualization and max number of workers
python scripts/generate/run_label.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1

Run trajectory parser for multiple paths with no visualization and max number of workers
Note: the paths args are the same
python scripts/generate/run_label.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1  benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1

Visualize the density map for a single path with max number of workers
python scripts/generate/run_label.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 -v

Visualize the density map for single paths with 1 worker
python scripts/generate/run_label.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 -w 1
'''
import argparse
from pathlib import Path
from path_planning.data_generation.dataset_label import label_dataset
from multiprocessing import cpu_count

if __name__ == "__main__":
    """Main entry point for trajectory parsing."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--paths", default=['benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1/radius0.0'], nargs='+', type=Path, help="input folder containing cases")
    parser.add_argument("-mapf","--mapf_solver_name", type=str, default='cbs', choices=['cbs', 'icbs', 'lacam', 'lacam_random'], help="MAPF solver to use")
    parser.add_argument("-rmt","--roadmap_type", type=str, default='grid', choices=['grid', 'prm', 'planar'], help="roadmap type")
    parser.add_argument("-av","--agent_velocity", type=float, default=0.0, help="agent velocity")
    parser.add_argument("-g","--graph_file_name", type=Path, default=None, help="graph file name")
    parser.add_argument("-v","--visualize", action='store_true', help="visualize the density map")
    parser.add_argument("-w","--num_workers", type=int, default=None, help="number of parallel workers (default: auto-detect CPU cores)")
    args = parser.parse_args()
    
    paths = args.paths
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    mapf_solver_name = args.mapf_solver_name
    roadmap_type = args.roadmap_type
    agent_velocity = args.agent_velocity
    visualize = args.visualize
    graph_file_name = args.graph_file_name
    roadmap_type = args.roadmap_type
    for idx, path in enumerate(paths):
        label_dataset(path, graph_file_name, mapf_solver_name, roadmap_type, agent_velocity, visualize, num_workers=num_workers)
        print(f"Path Trajectory -- [{idx+1}/{len(paths)}] Complete!")
