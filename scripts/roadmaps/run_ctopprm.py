"""
Run the Python CTopPRM (multi-start/goal) on a 2D map with agents.
python scripts/roadmaps/run_ctopprm.py
"""

from typing import Any


import os
import sys
import time
from itertools import permutations

# The installed path_planning package is non-editable; make the local
# checkout (which contains path_planning.cluster) win the import.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt

from path_planning.cluster.ctopprm import CTopPRM
from path_planning.cluster.shortening import path_length
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.utils.util import (
    agents_yaml_to_roadmap_frame,
    read_agents_for_map,
    read_graph_sampler_from_yaml,
    set_global_seed,
)

MAP_YAML = 'path_planning/maps/2d/2d_8agents.yaml'
OUT_DIR = 'figs/ctopprm'


if __name__ == "__main__":
    set_global_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Continuous mode keeps every node in world floats, the cleanest frame
    # for CTopPRM's continuous-space shortening/homotopy checks.
    map_ = read_graph_sampler_from_yaml(MAP_YAML, use_discrete_space=False)
    map_.inflate_obstacles(radius=1)
    agents = read_agents_for_map(MAP_YAML)
    agents = agents_yaml_to_roadmap_frame(map_, agents)

    starts = [tuple(a['start']) for a in agents]
    goals = [tuple(a['goal']) for a in agents]
    map_.set_start(starts)
    map_.set_goal(goals)
    map_.set_parameters(sample_num=800, num_neighbors=13, min_edge_len=1e-10, max_edge_len=8.0)

    st = time.time()
    nodes = map_.generateRandomNodes(generate_grid_nodes=False)
    road_map = map_.generate_roadmap(nodes)
    num_edges = sum(len(e) for e in road_map)
    print(f"Roadmap: {len(nodes)} nodes, {num_edges} edges in {time.time() - st:.2f}s")

    # Looser-than-default budgets so the demo surfaces alternate homotopy
    # classes that the C++ defaults (1.8 / 1.5) would filter on this map.
    planner = CTopPRM(map_, max_path_length_ratio=2.2,
                      cutoff_distance_ratio_to_shortest=2.0,
                      shortening_mode='none'
                      )
    # All ordered start/goal pairs from agent starts and goals.
    starts_goals = starts + goals
    # pairs = list(permutations(starts_goals, 2))
    pairs = list(zip(starts, goals)) + [(starts[0], goals[1])]


    st = time.time()
    results = planner.find_distinct_paths(pairs)
    num_initial_seeds = len(set(starts + goals))
    print(f"CTopPRM: {time.time() - st:.2f}s, seeds {num_initial_seeds} -> "
          f"{len(planner.seed_indices)} clusters")

    # --- clusters figure ---
    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_, map_.nodes, map_.road_map,
                      node_value=planner.cluster_labels,
                      cmap=plt.get_cmap('tab20'), edge_alpha=0.15)
    seed_pts = planner._points[planner.seed_indices]
    vis.ax.scatter(seed_pts[:, 0], seed_pts[:, 1], c='black', marker='*',
                   s=180, zorder=60, label='Cluster seeds')
    vis.savefig(f'{OUT_DIR}/clusters.png')
    vis.close()

    # --- per-pair path figures ---
    for i, (s, g) in enumerate(pairs):
        key = (planner.resolve_endpoint(s), planner.resolve_endpoint(g))
        paths = results[key]
        label = f"pair{i}"
        print(f"{label}: {s} -> {g}: {len(paths)} distinct path(s), "
              f"lengths {[round(path_length(p), 2) for p in paths]}")
        vis = Visualizer2D()
        vis.plot_grid_map(map_)
        colors = plt.get_cmap('tab10')
        for j, path in enumerate(paths):
            vis.ax.plot(path[:, 0], path[:, 1], color=colors(j % 10),
                        linewidth=2.5, zorder=50, label=f'path {j}')
        vis.ax.scatter([s[0], g[0]], [s[1], g[1]], c=['red', 'blue'], s=120, zorder=60)
        vis.ax.legend(loc='upper right', fontsize=8)
        vis.savefig(f'{OUT_DIR}/paths_{label}.png')
        vis.close()

    print(f"Figures written to {OUT_DIR}/")
