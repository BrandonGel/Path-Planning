import random
random.seed(0)

import numpy as np
np.random.seed(0)

from path_planning.utils.util import read_grid_from_yaml
from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *

if __name__ == "__main__":
    os.makedirs('figs/voronoiplanner',exist_ok=True)

    map_ =read_grid_from_yaml('path_planning/maps/2d/2d.yaml')
    map_.inflate_obstacles(radius=3)
    start = (5, 5)
    goal = (45, 25)
    map_.type_map[start] = TYPES.START
    map_.type_map[goal] = TYPES.GOAL

    planner = VoronoiPlanner(map_=map_, start=start, goal=goal)
    path, path_info = planner.plan()

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_path(path)       
    vis.plot_expand_tree(path_info["expand"])
    vis.savefig('figs/voronoiplanner/voronoiplanner_2d.png')
    vis.show()
    vis.close()


    map_ =read_grid_from_yaml('path_planning/maps/3d/3d.yaml')
    map_.inflate_obstacles(radius=3)
    start = (25, 5, 5)
    goal = (5, 25, 25)
    map_.type_map[start] = TYPES.START
    map_.type_map[goal] = TYPES.GOAL

    planner = VoronoiPlanner(map_=map_, start=start, goal=goal)
    path, path_info = planner.plan()

    vis = Visualizer3D()
    vis.plot_grid_map(map_)
    vis.plot_path(path)       
    
    # For 3D visualization, PyVista requires show() before savefig()
    # Call show() first to initialize the renderer, then save
    vis.show()
    vis.savefig('figs/voronoiplanner/voronoiplanner_3d.png')
    vis.close()
