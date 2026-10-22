import numpy as np
from itertools import product
from path_planning.common.environment.node import Node
from typing import List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase 

# def convert_np_to_tuple(array: np.ndarray) -> tuple:
#     grid_points = [tuple[int, ...](int(p) for p in pt) for pt in grid_points.reshape(self.dim,-1).T]
#     return tuple(int(p) for p in array)


    
def get_random_nodes(self) -> List[Node]:
    if len(self.nodes) >= self.sample_num:
        return self.nodes[:self.sample_num]
    print("Warning: number of nodes is less than sample num")
    return self.nodes

def sample_grid_nodes(graph_map: "GraphSamplerBase"):

    grid_shape = graph_map.shape
    grid_ranges = [np.arange(grid_shape[d]) for d in range(graph_map.dim)]
    grid_points = np.array(np.meshgrid(*grid_ranges, indexing='ij')).reshape(graph_map.dim,-1).T
    world_grid_points = graph_map.path_map_to_world(grid_points)
    num_grid_points = len(world_grid_points)
    
    expandable_mask = graph_map.is_expandable(grid_points)
    accepted_nodes = []
    accepted_world_grid_points = []
    for ii in range(num_grid_points):
        if expandable_mask[ii]:
            node = Node(world_grid_points[ii], None, 0, 0)
            accepted_nodes.append(node)
            accepted_world_grid_points.append(world_grid_points[ii])

    return accepted_nodes, accepted_world_grid_points
