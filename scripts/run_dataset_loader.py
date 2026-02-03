from pathlib import Path
from path_planning.gnn.dataloader import create_graph_dataset_loader, GraphDataset
from multiprocessing import cpu_count
import argparse

if __name__ == "__main__":
    """Main entry point for dataset loading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, nargs='+', default=['benchmark/train'], help="input file containing map and obstacles")
    parser.add_argument("-rmt","--road_map_type",type=str, default='planar', help="road map type")
    parser.add_argument("-ts","--target_space",type=str, default='binary', help="target space")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    args = parser.parse_args()


    file_paths = [Path(p) for p in args.path]
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    config = {
            "road_map_type": args.road_map_type,
            "target_space": args.target_space,
            "num_workers": num_workers,
        }
    data_files = create_graph_dataset_loader(file_paths,config)
    GraphDataset(data_files,num_workers=num_workers)