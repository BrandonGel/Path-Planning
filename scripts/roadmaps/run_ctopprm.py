"""
Run the Python CTopPRM (multi-start/goal) on a 2D map with agents.
python scripts/roadmaps/run_ctopprm.py
"""

import os
import sys
import time

# The installed path_planning package is non-editable; make the local
# checkout (which contains path_planning.cluster) win the import.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from path_planning.cluster.CTopPRMpy.ctopprm import CTopPRM
from path_planning.cluster.cluster_map import install_roadmap
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
    map_.set_roadmap_config(sample_num=800, num_neighbors=13, min_edge_length=1e-10, max_edge_length=8.0)

    st = time.time()
    map_.generateRandomNodes(generate_grid_nodes=False)
    map_.generate_roadmap("cdt")
    print(f"Roadmap: {len(map_.nodes)} nodes, {len(map_.edges)} edges in {time.time() - st:.2f}s")

    # Looser-than-default budgets so the demo surfaces alternate homotopy
    # classes that the C++ defaults (1.8 / 1.5) would filter on this map.
    planner = CTopPRM(map_, max_path_length_ratio=2.2,
                      cutoff_distance_ratio_to_shortest=2.0,
                      min_clusters=100,
                      force_min_clusters=True
                      )
    # Each agent's own pair, plus one cross pair.
    pairs = list(zip(starts, goals)) + [(starts[0], goals[1])]


    st = time.time()
    planner.set_up_distinct_paths(pairs)
    num_initial_seeds = len(set(starts + goals))
    print(f"CTopPRM: {time.time() - st:.2f}s, seeds {num_initial_seeds} -> "
          f"{len(planner.seed_indices)} clusters")

    # --- clusters figure ---
    map_ = read_graph_sampler_from_yaml(MAP_YAML, use_discrete_space=False)
    map_.inflate_obstacles(radius=1)
    map_.set_start(starts)
    map_.set_goal(goals)
    points, edges = planner.get_roadmap()
    install_roadmap(map_, points, edges)
    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_, map_.nodes, map_.road_map, map_frame=False)
    vis.savefig(f"{OUT_DIR}/ctopprm_2d.png")
    vis.close()