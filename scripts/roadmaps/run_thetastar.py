"""
Run Theta* for 2D and 3D maps.
python scripts/roadmaps/run_thetastar.py
"""

from path_planning.utils.util import read_grid_from_yaml
from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *
from path_planning.utils.util import set_global_seed

if __name__ == "__main__":
    set_global_seed(42)
    os.makedirs('figs/thetastar',exist_ok=True)

    map_ =read_grid_from_yaml('path_planning/maps/2d/2d.yaml')
    map_.inflate_obstacles(radius=3)
    start = (5, 5)
    goal = (45, 25)
    map_.type_map[start] = TYPES.START
    map_.type_map[goal] = TYPES.GOAL

    planner = ThetaStar(map_=map_, start=start, goal=goal,diagonal=False)
    path, path_info = planner.plan()

    vis = Visualizer2D()
    vis.plot_grid_map(map_)
    vis.plot_path(path)       
    vis.plot_expand_tree(path_info["expand"])
    vis.savefig('figs/thetastar/thetastar_2d.png')
    vis.close()


    map_ =read_grid_from_yaml('path_planning/maps/3d/3d.yaml')
    map_.inflate_obstacles(radius=3)
    start = (25, 5, 5)
    goal = (5, 25, 25)
    map_.type_map[start] = TYPES.START
    map_.type_map[goal] = TYPES.GOAL

    planner = ThetaStar(map_=map_, start=start, goal=goal,diagonal=False)
    path, path_info = planner.plan()

    vis = Visualizer3D()
    vis.plot_grid_map(map_)
    vis.plot_path(path)       
    
    # For 3D visualization, PyVista requires show() before savefig()
    # Call show() first to initialize the renderer, then save
    vis.show()
    vis.savefig('figs/thetastar/thetastar_3d.png')
