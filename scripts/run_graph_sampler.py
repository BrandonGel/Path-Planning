from path_planning.utils.util import read_grid_from_yaml
from path_planning.global_planner.sample_search.graph_sampler import GraphSampler
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.common.visualizer.visualizer_3d import Visualizer3D
from python_motion_planning.common import TYPES
import numpy as np
import random
import time

random.seed(0)
np.random.seed(0)

def run_graph_sampler(map_,start,goal,generate_grid_nodes = True):
    for s in start:
        map_.type_map[map_.world_to_map(s,discrete=True)] = TYPES.START
    for g in goal:
        map_.type_map[map_.world_to_map(g,discrete=True)] = TYPES.GOAL
    graph_sampler = GraphSampler(map_,start,goal,num_sample=1000,num_neighbors = 13.0, min_edge_len = 0.0, max_edge_len = 5.0)
    
    
    st = time.time()
    nodes = graph_sampler.generateRandomNodes(generate_grid_nodes = generate_grid_nodes)
    print(f"Generated {len(nodes)} nodes in {time.time() - st} seconds")

    st = time.time()
    road_map = graph_sampler.generate_roadmap(nodes)
    num_edges = sum([len(edges) for edges in road_map])
    print(f"Generated {num_edges} road map edges in {time.time() - st} seconds")

    st = time.time()
    planar_map = graph_sampler.generate_planar_map(nodes)
    num_edges = sum([len(edges) for edges in planar_map])   
    print(f"Generated {num_edges} planar map edges in {time.time() - st} seconds")

    return graph_sampler,nodes, road_map,planar_map


if __name__ == "__main__":
    map_ =read_grid_from_yaml('path_planning/maps/2d/2d.yaml')
    map_.inflate_obstacles(radius=3)
    start = [(5.67,14.88),(44.67,14.88)]
    goal = [(43.67,19.12)]
    print("Running graph sampler on 2D map...")
    graph_sampler,nodes, road_map,planar_map = run_graph_sampler(map_,start,goal,generate_grid_nodes = False)
    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,road_map)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_2d_roadmap.png')
    vis.close()

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,planar_map)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_2d_planarmap.png')
    vis.close()
    

    map_ =read_grid_from_yaml('path_planning/maps/3d/3d.yaml')
    map_.inflate_obstacles(radius=3)
    start = [(25, 5, 5)]
    goal = [(5, 25, 25)]
    print("Running graph sampler on 3D map...")
    graph_sampler,nodes, road_map,planar_map = run_graph_sampler(map_,start,goal,generate_grid_nodes = False)
    vis = Visualizer3D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,road_map)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_3d_roadmap.png')
    vis.close()

    vis = Visualizer3D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,planar_map)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_3d_planarmap.png')
    vis.close()



