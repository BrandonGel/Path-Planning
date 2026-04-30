import sys
from pathlib import Path

# Prefer local repo package over installed site-packages.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from path_planning.gnn.dataset_prune import read_prune_mechanism_from_yaml
from path_planning.gnn.dataset_prune import create_gnn_map_tasks
import argparse
from path_planning.data_generation.dataset_ground_truth_solve import  create_path_parameter_directory
from path_planning.utils.util import set_map_config
import yaml
import warnings
import logging
import torch
from path_planning.data_generation.dataset_visualize_graph import collect_graph_tasks
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
    parser.add_argument("-seed", "--seed", type=int, default=42, help="seed")
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test', help="input file containing map and obstacles")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,64.0,0,64.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.05, help="number of obstacles or obstacle density")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-rmt","--road_map_types",type=str, nargs="+", default=["grid", "prm", "cdt"], help="road map types")
    parser.add_argument("-ar","--agent_radii",nargs='+',type=float, default=[0.0, 1.0], help="agent radius")
    parser.add_argument("-rf","--run_folder",type=str, default="logs/gatv2_compile/wandb/run-20260402_181926-wsk652k2", help="run folder")
    parser.add_argument("-cp_path","--checkpoint_path",type=str, help="checkpoint path")
    parser.add_argument("-train","--train_config",type=str, default='config/train.yaml', help="train config file")
    parser.add_argument("-run_id","--run_id",type=str, help="run_id")
    parser.add_argument("-c","--num_cases",type=int, default=None, help="number of cases to generate")
    parser.add_argument("-g","--ground_truth_graph_file_name",type=str, default="graph_map.pkl", help="ground truth graph file name")
    parser.add_argument("-rmtgt","--ground_truth_road_map_type",type=str, default="grid", help="ground truth roadmap type")
    parser.add_argument("-prune_config","--prune_config",type=str, default='config/prune_config.yaml', help="prune config")
    parser.add_argument("-bs","--batch_size",type=int, default=32, help="graphs per GPU forward when num_workers<=1 (HeteroData batching)")
    parser.add_argument("-v","--verbose",dest="verbose",action="store_true", help="verbose")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="config file")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        map_config = yaml.load(f,Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config,args=args)
    num_workers=map_config['num_workers']
    base_path = map_config['path']
    seed = map_config['seed']
    bounds = map_config['bounds']
    resolution = map_config['resolution']
    nb_agents = map_config['nb_agents']

    # Array with all algorithms to run
    prune_mechanism = read_prune_mechanism_from_yaml(args.prune_config)
    agent_radii=args.agent_radii
    road_map_types=args.road_map_types

    for agent_radius in agent_radii:
        map_config = {
            "bounds": bounds,
            "resolution": args.resolution,
            "nb_agents": nb_agents,
            "nb_obstacles": args.nb_obstacles,
            "agent_radius": agent_radius,
        }
        path = create_path_parameter_directory(base_path, map_config,dump_config=False)

        tasks = collect_graph_tasks(
            base_path=path,
            road_map_types=road_map_types,
            target_space="ignore",
            case_mode="all",   
            ground_truth_graph_file_name=args.ground_truth_graph_file_name,
            ground_truth_road_map_type=args.ground_truth_road_map_type,
            agent_velocity=0.0,
            show=False,
            verbose=True,
            folder_key="map",
        )
        create_gnn_map_tasks(tasks, map_config, run_folder=args.run_folder, checkpoint_path=args.checkpoint_path, config_path=args.train_config, run_id=args.run_id, verbose=args.verbose, prune_mechanism=prune_mechanism,batch_size=args.batch_size)
