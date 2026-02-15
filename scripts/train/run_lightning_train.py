'''
python scripts/train/run_lightning_train.py
'''

import os
import warnings
import logging

# Suppress torch internal logging warnings BEFORE importing torch
os.environ['TORCH_LOGS'] = '-all'
os.environ['TORCHDYNAMO_VERBOSE'] = '0'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'

from pathlib import Path
from path_planning.gnn.dataloader import get_graph_dataset_file_paths, GraphDataset
from path_planning.gnn.train import split_dataset
from multiprocessing import cpu_count
from path_planning.utils.util import set_global_seed
from typing import List
import torch
import argparse
import yaml
from path_planning.utils.util import set_train_config

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.fx.experimental.symbolic_shapes')
warnings.filterwarnings('ignore', message='.*_maybe_guard_rel.*')

# Suppress torch loggers
logging.getLogger('torch._dynamo').setLevel(logging.CRITICAL)
logging.getLogger('torch._inductor').setLevel(logging.CRITICAL)
logging.getLogger('torch.fx.experimental.symbolic_shapes').setLevel(logging.CRITICAL)

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--folder_paths",type=str, nargs='+', help="folder paths of map and graph data")
    parser.add_argument("-l","--load_file",type=str, default=None, help="load file")
    parser.add_argument("-s","--save_file",type=str, default=None, help="save file")
    parser.add_argument("-cf","--config_file",type=str, default='config/train.yaml', help="config file")
    parser.add_argument("-ds","--use_discrete_space",type=bool, default=False, help="use discrete space")
    parser.add_argument("-rmt","--road_map_type",type=str, default='prm', help="road map type")
    parser.add_argument("-ts","--target_space",type=str, default='convolution_binary', help="target space")    
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("-cuda","--cuda",type=bool, default=True, help="use cuda")
    parser.add_argument("-train","--train_config",type=str, default='config/train.yaml', help="train config file")
    parser.add_argument("-compile","--compile",type=bool, default=False, help="compile model")
    parser.add_argument("-dynamic","--compile_dynamic",type=bool, default=True, help="compile dynamic mode")
    parser.add_argument("-seed","--seed",type=int, default=42, help="seed")
    args = parser.parse_args()

    folder_path = [
        # Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map32x32_resolution1.0/agents8_obst0.1'),
        Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32.0x32.0_resolution1.0/agents1_obst0.1'),
        # Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32.0x32.0_resolution1.0/agents2_obst0.1'),
        # Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32.0x32.0_resolution1.0/agents4_obst0.1'),
        # Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map32.0x32.0_resolution1.0/agents8_obst0.1'),
        # Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map32.0x32.0_resolution1.0/agents16_obst0.1'),
        # Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map16.0x16.0_resolution1.0/agents4_obst0.1'),
    ]
    import yaml
    with open(args.config_file, 'r') as f:
        train_config = yaml.load(f,Loader=yaml.FullLoader)
    train_config['dataset']['folder_path'] = folder_path
    train_config = set_train_config(train_config,args)

    from path_planning.gnn.lightning import train_graph
    test_results = train_graph(train_config,num_workers=args.num_workers,use_cuda=args.cuda)
    print(test_results)
