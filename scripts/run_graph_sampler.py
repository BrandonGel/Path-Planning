import random
random.seed(0)

import numpy as np
np.random.seed(0)

from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.common.visualizer.visualizer_3d import Visualizer3D
from path_planning.utils.util import read_graph_sampler_from_yaml
from python_motion_planning.common import TYPES
import os
import time

def run_graph_sampler(graph_sampler,start,goal,generate_grid_nodes = True,sample_num = 1000,num_neighbors = 4.0,min_edge_len = 0.5,max_edge_len = 5.0):
    for s in start:
        graph_sampler.type_map[graph_sampler.world_to_map(s,discrete=True)] = TYPES.START
    for g in goal:
        graph_sampler.type_map[graph_sampler.world_to_map(g,discrete=True)] = TYPES.GOAL
    graph_sampler.set_start(start)
    graph_sampler.set_goal(goal)
    graph_sampler.set_parameters(sample_num=sample_num, num_neighbors=num_neighbors, min_edge_len=min_edge_len, max_edge_len=max_edge_len)
    
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
    os.makedirs('figs/graph_sampler',exist_ok=True)
    map_ =read_graph_sampler_from_yaml('path_planning/maps/2d/2d.yaml')
    map_.inflate_obstacles(radius=3)
    start = [(5,15),(44,15)]
    goal = [(44,19), (20,25)]
    print("Running graph sampler on 2D map with discrete space and grid nodes...")
    map_,nodes, road_map,planar_map = run_graph_sampler(map_,start,goal,sample_num = 0)
    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,road_map)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_2d_roadmap_discrete_space.png')
    vis.close()

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,planar_map)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_2d_planarmap_discrete_space.png')
    vis.close()

    map_ =read_graph_sampler_from_yaml('path_planning/maps/2d/2d.yaml',use_discrete_space=False)
    map_.inflate_obstacles(radius=3)
    start = [(5,15),(44,15)]
    goal = [(44,19), (20,25)]
    print("Running graph sampler on 2D map with continuous space and random sampling nodes...")
    map_,nodes, road_map,planar_map = run_graph_sampler(map_,start,goal,sample_num = 1000,generate_grid_nodes=False)
    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,road_map,map_frame=False)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_2d_roadmap.png')
    vis.close()

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,planar_map,map_frame=False)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_2d_planarmap.png')
    vis.close()
    

    map_ =read_graph_sampler_from_yaml('path_planning/maps/3d/3d.yaml')
    start = [(25, 5, 5)]
    goal = [(5, 25, 25)]
    print("Running graph sampler on 3D map with discrete space and random sampling nodes...")
    map_,nodes, road_map,planar_map = run_graph_sampler(map_,start,goal,generate_grid_nodes = False,sample_num =1000)
    vis = Visualizer3D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,road_map)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_3d_roadmap_discrete_space.png')
    vis.close()

    vis = Visualizer3D()
    vis.plot_grid_map(map_)
    vis.plot_road_map(map_,nodes,planar_map)
    vis.show()
    vis.savefig('figs/graph_sampler/graph_sampler_3d_planarmap_discrete_space.png')
    vis.close()


    map_ =read_graph_sampler_from_yaml('path_planning/maps/3d/3d.yaml',use_discrete_space=False)
    start = [(25, 5, 5)]
    goal = [(5, 25, 25)]
    print("Running graph sampler on 3D map with continuous space and random sampling nodes...")
    map_,nodes, road_map,planar_map = run_graph_sampler(map_,start,goal,generate_grid_nodes = False,sample_num =1000)
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



