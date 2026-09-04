"""
Run CTopPRM with graph-distance EM clustering on a 2D map with agents.
python scripts/roadmaps/run_em.py
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from path_planning.cluster.CTopPRMpy.ctopprm import CTopPRM
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.utils.util import (
    agents_yaml_to_roadmap_frame,
    read_agents_for_map,
    read_graph_sampler_from_yaml,
    set_global_seed,
)

MAP_YAML = 'path_planning/maps/2d/2d_8agents.yaml'
OUT_DIR = 'figs/em'


if __name__ == "__main__":
    set_global_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

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
    road_map = map_.generate_planar_map(nodes)
    num_edges = sum(len(e) for e in road_map)
    print(f"Roadmap: {len(nodes)} nodes, {num_edges} edges in {time.time() - st:.2f}s")

    # EM seeding: agent endpoints stay fixed components, plus free
    # components whose anchors follow the responsibility-weighted centroids.
    num_endpoint_seeds = len(dict.fromkeys(starts + goals))
    k = num_endpoint_seeds + 84
    planner = CTopPRM(map_, clustering="em", min_clusters=k)
    pairs = list(zip(starts, goals))

    st = time.time()
    planner.set_up_distinct_paths(pairs)
    print(f"CTopPRM (em): {time.time() - st:.2f}s, seeds "
          f"{num_endpoint_seeds} -> {len(planner.seed_indices)} clusters")

    # --- clusters figure: nodes colored by cluster, seeds marked, and the
    # fitted mixture drawn as 1/2-sigma covariance ellipses per component ---
    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_, map_.nodes, map_.road_map,
                      node_value=planner.cluster_labels,
                      cmap=plt.get_cmap('tab20'), edge_alpha=0.15)
    em = planner.cluster_model
    if em is not None:
        # Keep the big 2-sigma ellipses from expanding the autoscaled
        # axes limits; they get clipped at the map bounds instead.
        xlim, ylim = vis.ax.get_xlim(), vis.ax.get_ylim()
        for k, (idx, cov) in enumerate(
            zip(em.center_node_indices, em.covariances_)
        ):
            vals, vecs = np.linalg.eigh(cov)
            angle = float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])))
            for n_std, style in ((1.0, '-'), (2.0, '--')):
                vis.ax.add_patch(Ellipse(
                    xy=em._points[idx],
                    width=2.0 * n_std * float(np.sqrt(vals[-1])),
                    height=2.0 * n_std * float(np.sqrt(vals[0])),
                    angle=angle, fill=False, edgecolor='black',
                    linestyle=style, linewidth=0.9, alpha=0.45, zorder=55,
                    label=f'components $\\pm{int(n_std)}\\sigma$' if k == 0 else None,
                ))
        vis.ax.set_xlim(xlim)
        vis.ax.set_ylim(ylim)
    seed_pts = planner._points[planner.seed_indices]
    fixed_pts = seed_pts[:num_endpoint_seeds]
    free_pts = seed_pts[num_endpoint_seeds:]
    vis.ax.scatter(fixed_pts[:, 0], fixed_pts[:, 1], c='black', marker='*',
                   s=200, zorder=60, label='Fixed components (endpoints)')
    if len(free_pts):
        vis.ax.scatter(free_pts[:, 0], free_pts[:, 1], c='black', marker='^',
                       s=120, zorder=60, label='EM / refinement seeds')
    handles, labels = vis.ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels)
            if l.startswith(('Fixed', 'EM', 'components'))]
    legend = vis.ax.legend(*zip(*keep), loc='upper right', fontsize=8)
    legend.set_zorder(100)
    vis.savefig(f'{OUT_DIR}/em_2d.png')
    vis.close()

    # --- distilled roadmap figure: min cluster tours as a custom roadmap ---
    map_ = read_graph_sampler_from_yaml(MAP_YAML, use_discrete_space=False)
    map_.inflate_obstacles(radius=1)
    map_.set_start(starts)
    map_.set_goal(goals)
    points, edges = planner.get_roadmap()
    map_.generate_custom_nodes(points)
    map_.generate_custom_roadmap(edges)
    print(f"Distilled roadmap: {len(points)} nodes, {len(edges)} edges")
    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_, map_.nodes, map_.road_map, map_frame=False)
    vis.savefig(f'{OUT_DIR}/em_roadmap_2d.png')
    vis.close()

    print(f"Figures written to {OUT_DIR}/")
