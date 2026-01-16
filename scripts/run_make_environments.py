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
    convert_grid_to_yaml(map_,filename)

def make_grid3d_map(filename: str = "path_planning/maps/3d/3d.yaml"):
    # Create environment with custom obstacles
    map_ = Grid(bounds=[[0, 31], [0, 31], [0, 31]], resolution=1.0)
    for i in range(75):     # 75 random obstacles
        rd_p = tuple(np.random.randint(0, 30, size=3))
        map_.type_map[rd_p[0], rd_p[1], :rd_p[2]] = TYPES.OBSTACLE
    map_.inflate_obstacles(radius=3)
    convert_grid_to_yaml(map_,filename)
    

if __name__ == "__main__":
    make_grid2d_map()
    make_grid3d_map()