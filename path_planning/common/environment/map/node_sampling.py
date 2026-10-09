import numpy as np
from itertools import product
from python_motion_planning.common.env.map.node import Node

# def convert_np_to_tuple(array: np.ndarray) -> tuple:
#     grid_points = [tuple[int, ...](int(p) for p in pt) for pt in grid_points.reshape(self.dim,-1).T]
#     return tuple(int(p) for p in array)


class NodeSampling:
    def __init__(self, map_, bounds: np.ndarray, resolution: float, dim: int):
        self.map_ = map_
        self.resolution = map_.resolution
        self.bounds = map_.bounds
        self.dim = map_.dim

    
    def generate_grid_nodes(grid_shape, ):
        # Generate all grid coordinate combinations 
        grid_shape = self.shape
        dim = len(grid_shape)
        grid_ranges = [np.arange(grid_shape[d]) for d in range(self.dim)]
        grid_points = np.array(np.meshgrid(*grid_ranges, indexing='ij'))
        grid_points = [tuple[int, ...](int(p) for p in pt) for pt in grid_points.reshape(self.dim,-1).T]
        world_grid_points = [self.map_to_world(pt) for pt in grid_points]
        
        for grid_pt,world_pt in zip(grid_points,world_grid_points):
            if self.is_expandable(grid_pt):
                node = Node(world_pt, None, 0, 0)
                nodes.append(node)
                self.node_index_dict[node] = len(nodes) - 1
                self.grid_nodes_index[node] = len(nodes)-1
                self.grid_points.append(world_pt)