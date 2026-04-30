"""
Run dataset loader with the default settings.
python scripts/train/run_dataset_loader.py

Run dataset loader with the default settings for a specific map and graph data.
python scripts/train/run_dataset_loader.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1

Run dataset loader with the default settings for a specific map and graph data.
-rmt: road map type (prm, planar)
-ts: target space (convolution_binary, convolution_distribution, binary, distribution, fuzzy_binary, fuzzy_distribution)
python scripts/train/run_dataset_loader.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 -rmt rrg -ts convolution_binary

Run dataset loader with the default settings for a specific map and graph data with 1 worker.
python scripts/train/run_dataset_loader.py -s benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1 -w 1
"""

from pathlib import Path
from path_planning.gnn.dataloader import get_graph_dataset_file_paths, GraphDataset
from multiprocessing import cpu_count
import argparse

if __name__ == "__main__":
    """Main entry point for dataset loading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, nargs='+', default=[
        '/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map64.0x64.0_resolution1.0/agents1_obst0.025/radius0.0',
        '/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map64.0x64.0_resolution1.0/agents2_obst0.025/radius0.0',
        '/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map64.0x64.0_resolution1.0/agents4_obst0.025/radius0.0',
        ], help="input file containing map and obstacles")
    parser.add_argument("-l","--load_file",type=str, default=None, help="load file")
    parser.add_argument("-sf","--save_file",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/data.pt', help="save file")
    parser.add_argument("-rmt","--road_map_type",type=str, nargs='+', default=['prm','cdt'], help="road map type")
    parser.add_argument("-ts","--target_space",type=str, nargs='+', default=['convolution_binary'], help="target space")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    args = parser.parse_args()


    file_paths = [Path(p) for p in args.path]
    load_file = Path(args.load_file) if args.load_file is not None else None
    save_file = Path(args.save_file) if args.save_file is not None else None
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    config = {
            "road_map_types": args.road_map_type,
            "target_spaces": args.target_space,
            "num_workers": num_workers,
        }
    data_files = get_graph_dataset_file_paths(file_paths,config)
    graph_dataset =GraphDataset(data_files,load_file=load_file,save_file=save_file,num_hops=-1,num_workers=num_workers)
    if save_file is not None:
        graph_dataset.save()
