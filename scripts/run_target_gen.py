'''
Generate a dataset of MAPF instances and their solutions.
python scripts/run_dataset_gen.py -s benchmark/train -ds False -ns 1000 -nn 4.0 -min_el 1e-10 -max_el 5+1e-10 -ngs 100 -rmt planar -ts binary -gn True -w None
'''

import argparse
from pathlib import Path
from path_planning.data_generation.target_gen import generate_graph_samples
from multiprocessing import cpu_count

if __name__ == "__main__":
    """Main entry point for dataset generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='benchmark/train', help="input file containing map and obstacles")
    parser.add_argument("-ds","--use_discrete_space",type=bool, default=False, help="use discrete space")
    parser.add_argument("-ns","--num_samples",type=int, default=1000, help="number of samples")
    parser.add_argument("-nn","--num_neighbors",type=float, default=4.0, help="number of neighbors")
    parser.add_argument("-min_el","--min_edge_len",type=float, default=1e-10, help="minimum edge length")
    parser.add_argument("-max_el","--max_edge_len",type=float, default=5+1e-10, help="maximum edge length")
    parser.add_argument("-ngs","--num_graph_samples",type=int, default=100, help="number of graph samples")
    parser.add_argument("-rmt","--road_map_type",type=str, default='planar', help="road map type")
    parser.add_argument("-ts","--target_space",type=str, default='binary', help="target space")
    parser.add_argument("-gn","--generate_new_graph",type=bool, default=True, help="generate new graph")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    args = parser.parse_args()
    

    folder_path = Path(args.path)
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    # config = {
    #         "use_discrete_space": args.use_discrete_space,
    #         "num_samples": args.num_samples,
    #         "num_neighbors": args.num_neighbors,
    #         "min_edge_len": args.min_edge_len,
    #         "max_edge_len": args.max_edge_len,
    #         "num_graph_samples": args.num_graph_samples,
    #         "road_map_type": args.road_map_type,
    #         "target_space": args.target_space,
    #         "generate_new_graph": args.generate_new_graph,
    #         "num_workers": num_workers,
    #     }
    # generate_graph_samples(folder_path, config)

    planar_config = {
    'use_discrete_space':False,
    'num_samples':1000,
    'num_neighbors':4.0,
    'min_edge_len':1.e-10,
    'max_edge_len':5+1e-10,
    'num_graph_samples':100,
    'road_map_type':'planar',
    'target_space':'binary',
    'generate_new_graph':True
    }
    prm_config = {
        'use_discrete_space':False,
        'num_samples':1000,
        'num_neighbors':12.0,
        'min_edge_len':1e-10,
        'max_edge_len':5+1e-10,
        'num_graph_samples':100,
        'road_map_type':'prm',
        'target_space':'binary',
        'generate_new_graph':True
    }

    file_paths = [
        Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map32.0x32.0_resolution1.0/agents4_obst0.1'),
        Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map32.0x32.0_resolution1.0/agents8_obst0.1'),
        Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map16.0x16.0_resolution1.0/agents4_obst0.1'),
    ]
    for file_path in file_paths:
        try:
            generate_graph_samples(file_path,planar_config)
        except Exception as e:
            print(f"Error generating graph samples for {file_path}: {e}")
        try:
            generate_graph_samples(file_path,prm_config)
        except Exception as e:
            print(f"Error generating graph samples for {file_path}: {e}")

