from python_motion_planning.common import Grid, TYPES
from path_planning.utils.util import convert_grid_to_yaml
import random
random.seed(0)
import numpy as np
np.random.seed(0)

def make_grid2d_map(filename: str = "path_planning/maps/2d/2d.yaml"):
    # Create environment with custom obstacles
    map_ = Grid(bounds=[[0, 51], [0, 31]])
    map_.fill_boundary_with_obstacles()
    map_.type_map[10:21, 15] = TYPES.OBSTACLE
    map_.type_map[20, :15] = TYPES.OBSTACLE
    map_.type_map[30, 15:] = TYPES.OBSTACLE
    map_.type_map[40, :16] = TYPES.OBSTACLE
    start = [(5,15),(44,15)]
    goal = [(44,19), (20,25)]
    agents = [
        { 
            "goal": goal[ii],
            "name": "agent"+str(ii),
            "start": start[ii],
        }
        for ii in range(len(start))
    ]
    for s in start:
        map_.type_map[s] = TYPES.START
    for g in goal:
        map_.type_map[g] = TYPES.GOAL
    convert_grid_to_yaml(map_,agents,filename)

def make_grid3d_map(filename: str = "path_planning/maps/3d/3d.yaml"):
    # Create environment with custom obstacles
    map_ = Grid(bounds=[[0, 31], [0, 31], [0, 31]], resolution=1.0)
    for i in range(75):     # 75 random obstacles
        rd_p = tuple(np.random.randint(0, 30, size=3))
        map_.type_map[rd_p[0], rd_p[1], :rd_p[2]] = TYPES.OBSTACLE
    start = [(5,5,5),(9,15,15)]
    goal = [(12,3,5), (25,10,5)]
    agents = [
        { 
            "goal": goal[ii],
            "name": "agent"+str(ii),
            "start": start[ii],
        }
        for ii in range(len(start))
    ]
    for s in start:
        map_.type_map[s] = TYPES.START
    for g in goal:
        map_.type_map[g] = TYPES.GOAL
    convert_grid_to_yaml(map_,agents,filename)
    

if __name__ == "__main__":
    make_grid2d_map()
    make_grid3d_map()