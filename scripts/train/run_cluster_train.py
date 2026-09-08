"""
Train the self-supervised cluster-GNN encoder (cluster.md milestones 2-3).

python scripts/train/run_cluster_train.py -train config/train_cluster.yaml -w 4 --no-cuda
python scripts/train/run_cluster_train.py -f <dataset folder> -e 50 -online
"""
import argparse
import sys
from pathlib import Path

# Ensure we import the local `path_planning` package (this repo) instead of an
# unrelated installed version from site-packages.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import yaml

from path_planning.gnn.train_cluster import run_cluster_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train_config", type=str, default="config/train_cluster.yaml", help="training config file")
    parser.add_argument("-f", "--folder_paths", type=str, nargs='+', default=None, help="override dataset folder path(s)")
    parser.add_argument("-s", "--save_folder", type=str, default=None, help="override model save folder")
    parser.add_argument("-e", "--num_epochs", type=int, default=None, help="override number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=None, help="override batch size")
    parser.add_argument("-seed", "--seed", type=int, default=None, help="override seed")
    parser.add_argument("-w", "--num_workers", type=int, default=1, help="number of dataloader workers")
    parser.add_argument("--cuda", dest="use_cuda", action="store_true")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false")
    parser.add_argument("-online", "--online", action="store_true", help="wandb online mode")
    parser.add_argument("--compile", dest="compile", action="store_true", help="torch.compile the model")
    parser.set_defaults(use_cuda=True, compile=None)
    args = parser.parse_args()

    with open(args.train_config, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    if args.folder_paths is not None:
        train_config['dataset']['folder_path'] = args.folder_paths
    if args.save_folder is not None:
        train_config['train']['save_folder'] = args.save_folder
    if args.num_epochs is not None:
        train_config['train']['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        train_config['train']['batch_size'] = args.batch_size
    if args.seed is not None:
        train_config['seed'] = args.seed
    if args.compile is not None:
        train_config['device']['compile'] = args.compile

    run_cluster_train(train_config, num_workers=args.num_workers,
                      use_cuda=args.use_cuda, online=args.online)
