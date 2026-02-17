"""
Example script to run RRG (Rapidly-exploring Random Graph) path planner.

RRG builds a graph structure that can be used for multi-query path planning.
After building the graph, you can use graph search algorithms (Dijkstra, A*)
to find paths between any start/goal pairs in the graph.

Usage:
    python scripts/run_rrg.py
"""

from path_planning.utils.util import set_global_seed
from path_planning.global_planner.sample_search.rrg import RRG
from path_planning.utils.util import read_grid_from_yaml
from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *

if __name__ == "__main__":
    set_global_seed(42)
    os.makedirs("figs/rrg", exist_ok=True)

    # Simple Example: 2D map (uncomment to run)
    print("=" * 60)
    print("RRG (Rapidly-exploring Random Graph) Example")
    print("=" * 60)

    # Example 1: 2D map
    print("Running RRG on 2D map...")
    map_2d = read_grid_from_yaml("path_planning/maps/2d/2d.yaml")
    map_2d.inflate_obstacles(radius=3)
    start_2d = (5, 5)
    goal_2d = (45, 25)
    map_2d.type_map[start_2d] = TYPES.START
    map_2d.type_map[goal_2d] = TYPES.GOAL

    # RRG expects start and goal to be lists
    planner_2d = RRG(
        map_=map_2d,
        start=[start_2d],  # List of start positions
        goal=[goal_2d],  # List of goal positions
        max_dist=1.5,  # Maximum connection distance
        sample_num=5000,  # Number of samples to generate
        goal_sample_rate=0.1,  # Probability of sampling goal directly
        discrete=False,  # Use continuous space
        use_faiss=False,  # Use FAISS for faster nearest neighbor search
    )

    path2d, path2d_info = planner_2d.plan()

    # RRG builds a graph structure (road_map) that can be used for path finding
    # The graph is stored in:
    #   - planner_2d.road_map: adjacency list representation
    #   - result_2d[1]["node_list"]: list of node positions
    #   - result_2d[1]["expand"]: dictionary of all nodes
    #
    # To find a path, you can use graph search algorithms (Dijkstra, A*) on the road_map
    print(f"RRG built graph with {len(planner_2d.road_map)} nodes")
    print(f"Graph structure stored in planner_2d.road_map")
    print(f"Use graph search (Dijkstra/A*) to find paths between nodes")

    vis_2d = Visualizer2D()
    vis_2d.plot_grid_map(map_2d)
    if "expand" in path2d_info:
        vis_2d.plot_expand_tree(path2d_info["expand"])
    vis_2d.plot_path(path2d[0])
    vis_2d.savefig("figs/rrg/rrg_2d.png")
    vis_2d.show()

    # Example 2: 3D map
    print("\nRunning RRG on 3D map...")
    map_3d = read_grid_from_yaml('path_planning/maps/3d/3d.yaml')
    map_3d.inflate_obstacles(radius=3)
    start_3d = (25, 5, 5)
    goal_3d = (5, 25, 25)
    map_3d.type_map[start_3d] = TYPES.START
    map_3d.type_map[goal_3d] = TYPES.GOAL

    planner_3d = RRG(
        map_=map_3d,
        start=[start_3d],
        goal=[goal_3d],
        max_dist=5.0,
        sample_num=10000,
        goal_sample_rate=0.1,
        discrete=False,
        use_faiss=True
    )

    path3d, path3d_info = planner_3d.plan()

    print(f"RRG built graph with {len(planner_3d.road_map)} nodes")

    vis_3d = Visualizer3D()
    vis_3d.plot_grid_map(map_3d)
    vis_3d.plot_path(path3d[0])
    vis_3d.show()
    vis_3d.savefig('figs/rrg/rrg_3d.png')
    vis_3d.close()

    # Example 3: Multiple start/goal positions (multi-query)
    print("\nRunning RRG with multiple start/goal positions...")
    map_multi = read_grid_from_yaml("path_planning/maps/2d/2d.yaml")
    map_multi.inflate_obstacles(radius=3)
    starts_multi = [(5, 5), (10, 10)]  # Multiple start positions
    goals_multi = [(45, 25), (45, 5)]  # Multiple goal positions

    for i, start in enumerate(starts_multi):
        map_multi.type_map[start] = TYPES.START
    for i, goal in enumerate(goals_multi):
        map_multi.type_map[goal] = TYPES.GOAL

    planner_multi = RRG(
        map_=map_multi,
        start=starts_multi,
        goal=goals_multi,
        max_dist=1.5,
        sample_num=5000,
        goal_sample_rate=0.1,
        discrete=False,
        use_faiss=False,
    )

    path_multi, path_multi_info = planner_multi.plan()

    print(f"RRG built graph with {len(planner_multi.road_map)} nodes")
    print("Graph can be used for multi-query path planning")

    vis_multi = Visualizer2D()
    vis_multi.plot_grid_map(map_multi)
    if "expand" in path_multi_info:
        vis_multi.plot_expand_tree(path_multi_info["expand"])
    for path in path_multi:
        vis_multi.plot_path(path)
    vis_multi.savefig("figs/rrg/rrg_multi.png")
    vis_multi.show()
    vis_multi.close()
    print("\nRRG examples completed!")
