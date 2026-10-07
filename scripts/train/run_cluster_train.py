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
    parser.add_argument("--no-boundary-edges", dest="use_boundary_edges", action="store_false",
                        help="edge ablation: drop the ('node','boundary','node') relation")
    parser.add_argument("--no-task-edges", dest="use_task_edges", action="store_false",
                        help="edge ablation: drop the ('node','approx','node') start/goal relation "
                             "(also removes it from the shortest-path loss)")
    parser.add_argument("--flat-node-features", dest="use_node_type_features", action="store_false",
                        help="information ablation: flatten the [start/goal, free, boundary] node "
                             "one-hot to [0, 1, 0] for every node")
    parser.add_argument("--no-boundary-features", dest="use_boundary_node_features", action="store_false",
                        help="information ablation: re-label boundary nodes as free in the node one-hot "
                             "(start/goal column kept)")
    parser.add_argument("--cluster-alpha", type=float, default=None,
                        help="override encoder.loss.cluster.args.alpha (0 = no obstacle weighting)")
    parser.set_defaults(use_cuda=True, compile=None, use_boundary_edges=True, use_task_edges=True,
                        use_node_type_features=True, use_boundary_node_features=True)
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
    # Edge-ablation flags live in dataset.config so wandb persists them into the
    # run's config.yaml, where gnn_cluster_map.load_cluster_encoder reads them back.
    if not args.use_boundary_edges:
        train_config['dataset']['config']['use_boundary_edges'] = False
    if not args.use_task_edges:
        train_config['dataset']['config']['use_task_edges'] = False
    if not args.use_node_type_features:
        train_config['dataset']['config']['use_node_type_features'] = False
    if not args.use_boundary_node_features:
        train_config['dataset']['config']['use_boundary_node_features'] = False
    if args.cluster_alpha is not None:
        train_config['encoder']['loss']['cluster']['args']['alpha'] = float(args.cluster_alpha)

    run_cluster_train(train_config, num_workers=args.num_workers,
                      use_cuda=args.use_cuda, online=args.online)
