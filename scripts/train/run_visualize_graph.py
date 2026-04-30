import argparse
import os
import yaml
from multiprocessing import cpu_count

from path_planning.data_generation.dataset_visualize_graph import (
    collect_graph_tasks,
    visualize_graphs,
)
from path_planning.utils.util import set_map_config
from path_planning.data_generation.dataset_util import create_path_parameter_directory
from path_planning.gnn.dataset_prune import get_prune_function,get_gnn_paths
from path_planning.gnn.dataset_prune import get_prune_mechanism_folder,read_prune_mechanism_from_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train', help="input file containing map and obstacles")
    parser.add_argument("-rmt","--road_map_types",type=str,nargs="+",default=["prm"],help="Road map types to visualize (matches folders under sample/<road_type>/).",)
    parser.add_argument("-cm","--case_mode",type=str, default="all", choices=["all", "first_n", "range", "specific"], help="How to select case_*/ folders.")
    parser.add_argument("-cn","--num_cases",type=int, default=25, help="Number of cases when case_mode=first_n.")
    parser.add_argument("-cr","--case_range",type=int, nargs=2, default=[0, 16], help="Range of cases when case_mode=range (start end).")
    parser.add_argument("-cs","--specific_cases",type=int, nargs='+', default=[0, 5, 10], help="Specific cases when case_mode=specific.")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,32.0,0,32.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.1, help="number of obstacles or obstacle density")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-ar","--agent_radius",type=float, default=0.0, help="agent radius")
    parser.add_argument("-av","--agent_velocity",type=float, default=0.0, help="agent velocity")
    parser.add_argument("-rf","--run_folder",type=str, default="logs/gatv2_compile/wandb/run-20260409_022405-dgaev32o", help="run folder")
    parser.add_argument("-cp_path","--checkpoint_path",type=str, help="checkpoint path")
    parser.add_argument("-cfg_path","--config_path",type=str, help="config path")
    parser.add_argument("-run_id","--run_id",type=str, help="run_id")
    parser.add_argument("-gf","--graph_file_name",type=str, help="gnn folder name")
    parser.add_argument("-g","--ground_truth_graph_file_name",type=str, default="graph_map.pkl", help="ground truth graph file name")
    parser.add_argument("-rmtgt","--ground_truth_road_map_type",type=str, default="grid", help="ground truth roadmap type")
    parser.add_argument("-prune_config","--prune_config",type=str, default='config/prune_config.yaml', help="prune config file")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="Number of parallel workers (default: auto).")
    parser.add_argument("-v","--verbose",action="store_true", help="verbose")
    parser.add_argument("--show",action="store_true",help="Call visualizer.show() (slower; useful for debugging).")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="config file")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        map_config = yaml.load(f,Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config,args=args)
    base_path = map_config['path']

    # Create/resolve the map folder that contains case_*/sample/... for the config.
    path = create_path_parameter_directory(base_path, map_config, dump_config=False)
    assert os.path.exists(path), f"Path does not exist: {path}"

    _, _, _, gnn_folder_name = get_gnn_paths(
        run_folder=args.run_folder,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        run_id=args.run_id,
    )


    prune_mechanism = read_prune_mechanism_from_yaml(args.prune_config)
    prune_name = get_prune_mechanism_folder(prune_mechanism)
    k_hop = prune_mechanism.get('k_hop', 0)
    if len(prune_name) == 0:
        graph_file_name = "graph_map.pkl"
        target_space = 'predictions'
    else:
        graph_file_name = f"graph_sampler_{prune_name}.pkl"
        target_space = f'{prune_name}'

   
    

    tasks = collect_graph_tasks(
        base_path=path,
        road_map_types=args.road_map_types,
        graph_file_name =graph_file_name,
        target_space=target_space,
        case_mode=args.case_mode,
        num_cases=args.num_cases,   
        ground_truth_graph_file_name=args.ground_truth_graph_file_name,
        ground_truth_road_map_type=args.ground_truth_road_map_type,
        agent_velocity=args.agent_velocity,
        case_range=args.case_range,
        specific_cases=args.specific_cases,
        gnn_folder_name=gnn_folder_name,
        show=args.show,
        verbose=args.verbose,
    )

    if len(tasks) == 0:
        print(f"No graph samples found under: {path}")
        raise SystemExit(0)

    resolved_num_workers = args.num_workers
    if resolved_num_workers is None:
        resolved_num_workers = map_config.get("num_workers", None)
    if resolved_num_workers is None:
        resolved_num_workers = cpu_count()

    visualize_graphs(tasks, num_workers=resolved_num_workers)
