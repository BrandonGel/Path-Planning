"""
Run Halton-based roadmap generation for 2D and 3D maps.

Halton quasi-random samples are drawn in the map bounds and filtered by
distance-to-obstacle (see path_planning/common/environment/map/halton.py).
2D: CDT over obstacles; 3D: Delaunay triangulation on Halton samples (CDT is 2D-only).

Usage:
    python scripts/roadmaps/run_halton.py
"""

import os
import time

from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.common.visualizer.visualizer_3d import Visualizer3D
from path_planning.utils.util import read_graph_sampler_from_yaml, set_global_seed
from python_motion_planning.common import TYPES


def run_halton(
    graph_sampler,
    start,
    goal,
    sample_num=1000,
    roadmap_type="halton",
    min_edge_len=0.5,
    max_edge_len=5.0,
    halton_cfg=None,
):
    """Build a Halton-sampled roadmap and optionally persist it."""
    set_global_seed(42)

    if not hasattr(graph_sampler, "sampling_dist_dict"):
        graph_sampler.sampling_dist_dict = {}
    if halton_cfg:
        graph_sampler.sampling_dist_dict["halton"] = halton_cfg

    for s in start:
        graph_sampler.type_map[graph_sampler.world_to_map(s, discrete=True)] = TYPES.START
    for g in goal:
        graph_sampler.type_map[graph_sampler.world_to_map(g, discrete=True)] = TYPES.GOAL
    graph_sampler.set_start(start)
    graph_sampler.set_goal(goal)
    graph_sampler.set_parameters(
        sample_num=sample_num,
        num_neighbors=0,
        min_edge_len=min_edge_len,
        max_edge_len=max_edge_len,
    )

    st = time.time()
    nodes = graph_sampler.generateRandomNodes(
        generate_grid_nodes=False,
        roadmap_type=roadmap_type,
    )
    print(f"Generated {len(nodes)} Halton nodes in {time.time() - st:.2f} s")

    st = time.time()
    if graph_sampler.dim == 2:
        graph_sampler.generate_map(roadmap_type, nodes)
        connect_label = "CDT"
    else:
        # CDT boundary extraction is 2D-only; connect Halton nodes with 3D Delaunay.
        graph_sampler.generate_planar_map(nodes, use_option="dt")
        connect_label = "Delaunay"
    road_map = graph_sampler.road_map
    num_edges = sum(len(edges) for edges in road_map)
    print(f"Generated {num_edges} {connect_label} edges in {time.time() - st:.2f} s")
    print(f"Roadmap has {len(graph_sampler.nodes)} nodes after {connect_label}")
    return graph_sampler, graph_sampler.nodes, road_map


if __name__ == "__main__":
    os.makedirs("figs/halton", exist_ok=True)

    halton_cfg = {
        "d_min": 0.3,
        "d_opt": 0.4,
        "sigma": 0.5,
        "floor": 0.2,
    }

    # 2D — continuous space (Halton + CDT)
    print("Running Halton roadmap on 2D map (continuous space)...")
    map_2d = read_graph_sampler_from_yaml(
        "path_planning/maps/2d/2d.yaml", use_discrete_space=False
    )
    map_2d.inflate_obstacles(radius=3)
    start_2d = [(5, 15), (44, 15)]
    goal_2d = [(44, 19), (20, 25)]

    map_2d, nodes_2d, road_map_2d = run_halton(
        map_2d,
        start_2d,
        goal_2d,
        sample_num=1000,
        halton_cfg=halton_cfg,
    )

    vis = Visualizer2D()
    vis.plot_grid_map(map_2d)
    vis.plot_road_map(map_2d, nodes_2d, road_map_2d, map_frame=False)
    vis.savefig("figs/halton/halton_2d.png")
    vis.show()
    vis.close()

    # 3D — continuous space
    print("\nRunning Halton roadmap on 3D map (continuous space)...")
    map_3d = read_graph_sampler_from_yaml(
        "path_planning/maps/3d/3d.yaml", use_discrete_space=False
    )
    map_3d.inflate_obstacles(radius=3)
    start_3d = [(25, 5, 5)]
    goal_3d = [(5, 25, 25)]

    map_3d, nodes_3d, road_map_3d = run_halton(
        map_3d,
        start_3d,
        goal_3d,
        sample_num=1000,
        halton_cfg=halton_cfg,
    )

    vis = Visualizer3D()
    vis.plot_grid_map(map_3d)
    vis.plot_road_map(map_3d, nodes_3d, road_map_3d)
    vis.show()
    vis.savefig("figs/halton/halton_3d.png")
    vis.close()

    print("\nHalton roadmap examples completed.")
