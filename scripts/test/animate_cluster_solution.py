"""
Animate a MAPF solution solved on a cluster-distilled roadmap from the
test_cluster campaign (case_*/perm/perm_j/sipp/<rmt>/solution_<graph>_velocity<v>.yaml).

Mirrors the styling of results/cluster/cluster7_video.ipynb: per-agent tab10
colours, goal triangles as world-unit patches, start markers hidden (the agent
circle sits there), no legend, roadmap drawn underneath.

python scripts/test/animate_cluster_solution.py --cf 1 -n 64 --case 1 --perm 1 \
    --rmt halton --graph graph_map_gnn6 -o results/cluster/videos/out.gif
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import yaml
from python_motion_planning.common import TYPES

from path_planning.utils.util import (read_agents_from_yaml, agents_yaml_to_roadmap_frame,
                                      read_graph_sampler_from_yaml)
from path_planning.common.visualizer.visualizer_2d import Visualizer2D

DATA = Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test_cluster')
ZORDER = {
    'density_map': -1, 'grid_map': 0, 'voxels': 10, 'esdf': 20, 'road_map': 0,
    'expand_tree_edge': 30, 'expand_tree_node': 40, 'path_2d': 50, 'path_3d': 700,
    'traj': 0, 'lookahead_pose_node': 70, 'lookahead_pose_orient': 80, 'pred_traj': 90,
    'robot_circle': 100, 'robot_orient': 110, 'robot_text': 120, 'env_info_text': 10000,
}


def case_root(cf, n_agents: int, case: int, dataset: str = None) -> Path:
    if dataset is None:
        if cf is None:
            sys.exit("pass --cf or --dataset")
        dataset = f"test_samples1500_cf{cf:02d}"
    return (DATA / dataset / "map64.0x64.0_resolution1.0"
            / f"agents{n_agents}_obst0.025" / "radius0.5" / f"case_{case}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", type=int, default=None, help="cluster fraction index, e.g. 1 -> _cf01")
    ap.add_argument("--dataset", default=None,
                    help="dataset folder under test_cluster (e.g. test_samples450); overrides --cf")
    ap.add_argument("-n", "--agents", type=int, required=True)
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--perm", type=int, default=1)
    ap.add_argument("--rmt", default="halton")
    ap.add_argument("--graph", default="graph_map_gnn6", help="pickle stem under maps/<rmt>/cluster")
    ap.add_argument("--velocity", type=float, default=1.0)
    ap.add_argument("--agent_radius", type=float, default=0.5)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--solution", default=None,
                    help="solution yaml to animate instead of the campaign file (e.g. a re-solve)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--intermediate_frames", type=int, default=3)
    ap.add_argument("--speed", type=float, default=5)
    ap.add_argument("--show_paths", action="store_true")
    args = ap.parse_args()

    root = case_root(args.cf, args.agents, args.case, args.dataset)
    graph_file = root / "maps" / args.rmt / "cluster" / f"{args.graph}.pkl"
    if not graph_file.exists():  # un-clustered source roadmap (--graph graph_map)
        graph_file = root / "maps" / args.rmt / f"{args.graph}.pkl"
    perm_path = root / "perm" / f"perm_{args.perm}"
    input_file = perm_path / "input.yaml"
    sol_file = (Path(args.solution) if args.solution else
                perm_path / "sipp" / args.rmt / f"solution_{args.graph}_velocity{args.velocity}.yaml")
    for f in (graph_file, input_file, sol_file):
        if not f.exists():
            sys.exit(f"missing: {f}")

    solution = yaml.safe_load(open(sol_file))
    if not solution.get("success") or not solution.get("schedule"):
        sys.exit(f"solution not successful (empty schedule): {sol_file}")

    map_ = read_graph_sampler_from_yaml(str(input_file), graph_file=str(graph_file))
    agents = read_agents_from_yaml(str(input_file))
    agents_rt = agents_yaml_to_roadmap_frame(map_, agents)
    map_.set_start([a["start"] for a in agents_rt])
    map_.set_goal([a["goal"] for a in agents_rt])
    for t in (TYPES.START, TYPES.GOAL, TYPES.INFLATION):
        map_.type_map.data[map_.type_map.data == t] = TYPES.FREE

    path_colors = plt.get_cmap('tab10').colors[:5] + plt.get_cmap('tab10').colors[6:]
    goal_marker_size = 500
    start_specs, goal_specs = [], []
    for ii in range(len(map_.start)):
        color = path_colors[ii % len(path_colors)]
        start_specs.append(([map_.start[ii]], color, 'o', 0, 'Start'))           # hidden
        goal_specs.append(([map_.goal[ii]], color, '^', goal_marker_size, 'Goal'))
    extra_special_specs = start_specs + goal_specs

    # Schedule agent order must match map_.start order for colours to line up.
    schedule = {"schedule": {a["name"]: solution["schedule"][a["name"]] for a in agents}}

    rel = sol_file.relative_to(DATA) if sol_file.is_relative_to(DATA) else sol_file
    print(f"{rel}: {len(agents)} agents, makespan {solution['makespan']:.1f}, "
          f"{len(map_.nodes)} nodes")
    vis = Visualizer2D(figsize=(8, 8), zorder=ZORDER)
    vis.animate(
        args.out, map_, schedule,
        road_map=map_.road_map,
        show_legend=False,
        show_paths=args.show_paths,
        skip_frames=1,
        intermediate_frames=args.intermediate_frames,
        speed=args.speed,
        radius=args.agent_radius,
        map_frame=map_.use_discrete_space,
        agent_colors=path_colors,
        extra_special_specs=extra_special_specs,
        special_patch_radius=args.agent_radius,
        dpi=args.dpi,
    )
    vis.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
