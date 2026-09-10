"""
Visualize cluster-distilled roadmaps (ctopprm/kmeans/em/gnn) produced by
run_all_cluster_solvers.py / run_all_cluster_gnn_maps.py.

For each selected case x road map type x method, renders the occupancy map
(Visualizer2D.plot_grid_map: obstacles + inflation + start/goal cells) with
the distilled roadmap on top, saved next to the pkl as
``maps/<rmt>/cluster/graph_map_<method>.png``. With --source it also renders
the full source roadmap (``maps/<rmt>/graph_map.png``) for side-by-side
comparison.

python scripts/test/run_visualize_cluster_graph.py -s <root> -n 4 -rmt prm -cmm gnn kmeans -cn 2
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.data_generation.dataset_ground_truth_map import create_map
from path_planning.data_generation.dataset_util import (
    create_path_parameter_directory,
    generate_base_case_path,
    generate_cluster_path,
    get_graph_file_path,
    get_input_file_path,
)

METHOD_GRAPHS = {
    "ctopprm": "graph_map_ctopprm.pkl",
    "kmeans": "graph_map_kmeans.pkl",
    "em": "graph_map_em.pkl",
    "gnn": "graph_map_gnn.pkl",
    # Full source roadmap, no clustering (the comparison baseline); lives in
    # maps/<rmt>/ rather than the cluster/ subdir.
    "none": "graph_map.pkl",
}


def render(map_, title, out_file):
    plt.close('all')
    vis = Visualizer2D(figname=str(out_file), figsize=(9, 9))
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_, map_.nodes, map_.road_map, map_frame=map_.use_discrete_space)
    vis.ax.set_title(title)
    vis.savefig(out_file)
    vis.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--path",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test_samples1500', help="dataset root")
    parser.add_argument("-b","--bounds",type=float, nargs='+', default=[0,64.0,0,64.0])
    parser.add_argument("-n","--nb_agents",type=int, default=4)
    parser.add_argument("-o","--nb_obstacles",type=float, default=0.025)
    parser.add_argument("-r","--resolution",type=float, default=1.0)
    parser.add_argument("-ar","--agent_radius",type=float, default=0.5)
    parser.add_argument("-rmt","--road_map_types",type=str, nargs='+', default=['grid', 'prm', 'cdt'])
    parser.add_argument("-cmm","--cluster_methods",type=str, nargs='+', default=['none'], choices=['none'] + list(METHOD_GRAPHS))
    parser.add_argument("-cm","--case_mode",type=str, default="first_n", choices=["all", "first_n", "specific"])
    parser.add_argument("-cn","--num_cases",type=int, default=2)
    parser.add_argument("-cs","--specific_cases",type=int, nargs='+', default=[0])
    parser.add_argument("-c","--total_cases",type=int, default=25, help="cases in the dataset")
    parser.add_argument("--source",action="store_true", help="also render the full source roadmap")
    args = parser.parse_args()

    map_config = {
        "bounds": [[args.bounds[0], args.bounds[1]], [args.bounds[2], args.bounds[3]]],
        "resolution": args.resolution,
        "nb_agents": args.nb_agents,
        "nb_obstacles": args.nb_obstacles,
        "agent_radius": args.agent_radius,
    }
    path = create_path_parameter_directory(Path(args.path), map_config, dump_config=False)

    if args.case_mode == "all":
        case_ids = list(range(args.total_cases))
    elif args.case_mode == "specific":
        case_ids = args.specific_cases
    else:
        case_ids = list(range(min(args.num_cases, args.total_cases)))

    rendered, missing = 0, 0
    for road_map_type in args.road_map_types:
        for case_id in case_ids:
            case_path, map_path = generate_base_case_path(path, case_id, road_map_type)
            input_file = get_input_file_path(case_path)
            if not input_file.exists():
                continue
            with open(input_file) as f:
                inpt = yaml.safe_load(f)
            cluster_dir = generate_cluster_path(map_path)
            if args.source:
                src_file = get_graph_file_path(map_path)
                if src_file.exists():
                    m = create_map(inpt, graph_file=src_file, verbose=False,
                                   args={"use_constraint_sweep": False})
                    render(m, f"case_{case_id} {road_map_type} source ({len(m.nodes)} nodes)",
                           map_path / "graph_map.png")
                    rendered += 1
            for method in args.cluster_methods:
                pkl_dir = map_path if method == "none" else cluster_dir
                pkl = get_graph_file_path(pkl_dir, METHOD_GRAPHS[method])
                if not pkl.exists():
                    missing += 1
                    continue
                m = create_map(inpt, graph_file=pkl, verbose=False,
                               args={"use_constraint_sweep": False})
                render(m, f"case_{case_id} {road_map_type}/{method} "
                          f"({len(m.nodes)} nodes, {len(m.edges)} edges)",
                       pkl_dir / f"{pkl.stem}.png")
                rendered += 1
                print(f"case_{case_id} {road_map_type}/{method} -> {pkl.stem}.png")
    print(f"rendered {rendered} figures ({missing} method pkls missing)")
