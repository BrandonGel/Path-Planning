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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train', help="input file containing map and obstacles")
    parser.add_argument("-rmt","--road_map_type",type=str,default="prm",help="Road map type to visualize (matches folders under sample/<road_type>/).",)
    parser.add_argument("-ts","--target_space",type=str, default="convolution_binary", help="Target space to visualize.")
    parser.add_argument("-cm","--case_mode",type=str, default="first_n", choices=["all", "first_n", "range", "specific"], help="How to select case_*/ folders.")
    parser.add_argument("-cn","--num_cases",type=int, default=2, help="Number of cases when case_mode=first_n.")
    parser.add_argument("-cr","--case_range",type=int, nargs=2, default=[0, 16], help="Range of cases when case_mode=range (start end).")
    parser.add_argument("-cs","--specific_cases",type=int, nargs='+', default=[0, 5, 10], help="Specific cases when case_mode=specific.")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,64.0,0,64.0], help="bounds of the map as x_min x_max y_min y_max (e.g., 0 32.0 0 32.0)")
    parser.add_argument("-n","--nb_agents",type=int, default=4, help="number of agents")
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.025, help="number of obstacles or obstacle density")
    parser.add_argument("-r","--resolution",type=float, default=1.0, help="resolution of the map")
    parser.add_argument("-ar","--agent_radius",type=float, default=1.0, help="agent radius")
    parser.add_argument("-av","--agent_velocity",type=float, default=0.0, help="agent velocity")
    parser.add_argument("-g","--ground_truth_graph_file_name",type=str, default="graph_map.pkl", help="ground truth graph file name")
    parser.add_argument("-rmtgt","--ground_truth_road_map_type",type=str, default="grid", help="ground truth roadmap type")
    parser.add_argument("-w","--num_workers",type=int, default=None, help="Number of parallel workers (default: auto).")
    parser.add_argument("-v","--verbose",action="store_true", help="verbose")
    parser.add_argument("-cfg","--config",type=str, default='config/map.yaml', help="config file")
    parser.add_argument("--show",action="store_true",help="Call visualizer.show() (slower; useful for debugging).")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        map_config = yaml.load(f,Loader=yaml.FullLoader)
    map_config = set_map_config(map_config=map_config,args=args)
    base_path = map_config['path']

    # Create/resolve the map folder that contains case_*/sample/... for the config.
    path = create_path_parameter_directory(base_path, map_config, dump_config=False)
    assert os.path.exists(path), f"Path does not exist: {path}"

    tasks = collect_graph_tasks(
        base_path=path,
        road_map_types=args.road_map_types,
        target_space=args.target_space,
        case_mode=args.case_mode,
        num_cases=args.num_cases,   
        ground_truth_graph_file_name=args.ground_truth_graph_file_name,
        ground_truth_road_map_type=args.ground_truth_road_map_type,
        agent_velocity=args.agent_velocity,
        case_range=args.case_range,
        specific_cases=args.specific_cases,
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
