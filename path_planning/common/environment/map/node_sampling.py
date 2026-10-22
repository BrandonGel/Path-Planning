import numpy as np
from itertools import product
from path_planning.common.environment.node import Node
from python_motion_planning.common import TYPES
from typing import List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase 

    
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


def boundary_vertices(graph_map: "GraphSamplerBase", include_map_edge: bool = True):
    """Lattice vertices on the outline of free space, with a corner flag.

    The outline of free space is the set of cell faces between a free cell
    (neither OBSTACLE nor INFLATION) and a blocked cell, or the map edge when
    ``include_map_edge`` is True. Its vertices are the cell corners of those
    faces, so along a straight wall there is one vertex every ``resolution``.

    A vertex touches 2^dim cells. It is on the outline when at least one of
    them is free and at least one is blocked. It is an *edge* vertex when the
    blocked cells are exactly one half of that neighbourhood split by a single
    axis (a flat wall passes through it); every other outline vertex is
    flagged as a *corner* (convex or concave, or where a wall meets the map
    edge). In 2-D that is exactly the corner set; in 3-D the flag also covers
    vertices lying on the straight edges of a wall, i.e. anything not interior
    to a flat face.

    Returns:
        vertices: (K, dim) float64 points in the MAP frame (cell coordinates,
            where integer (r, c) is the centre of cell (r, c)), in row-major
            order. Corners sit half a cell off the centres, so the lower
            corner of cell (i, j) is returned as (i - 0.5, j - 0.5); pass them
            to ``graph_map.path_map_to_world`` to get world points.
        is_corner: (K,) bool.
        free_dir: (K, dim) int in {-1, 0, +1}, the direction from the vertex
            into free space (wall normal on an edge, diagonal at a corner).
    """
    type_map = graph_map.type_map.data
    blocked = (type_map == TYPES.OBSTACLE) | (type_map == TYPES.INFLATION)
    dim = blocked.ndim
    # Pad by one cell: outside the map counts as blocked (map edge is part of
    # the outline) or as free (it is not).
    padded = np.pad(blocked, 1, mode="constant", constant_values=include_map_edge)
    vertex_shape = tuple(n + 1 for n in blocked.shape)

    # patterns[k] = blocked state of the cell at corner offset k of each vertex
    offsets = list(product((0, 1), repeat=dim))
    patterns = np.stack([
        padded[tuple(slice(o, o + n) for o, n in zip(off, vertex_shape))] for off in offsets
    ])                                                   # (2^dim, *vertex_shape)

    any_blocked = patterns.any(axis=0)
    any_free = ~patterns.all(axis=0)
    on_outline = any_blocked & any_free

    offsets_arr = np.asarray(offsets, dtype=bool)        # (2^dim, dim)
    is_edge = np.zeros(vertex_shape, dtype=bool)
    for d in range(dim):
        lower = patterns[~offsets_arr[:, d]]             # cells on the low side of axis d
        upper = patterns[offsets_arr[:, d]]
        is_edge |= lower.all(axis=0) & ~upper.any(axis=0)
        is_edge |= upper.all(axis=0) & ~lower.any(axis=0)

    corners = np.argwhere(on_outline)                  # (K, dim) lattice indices
    idx = tuple(corners.T)
    # Map frame: cell (r, c) is centred on (r, c), so corner (i, j) is (i - 0.5, j - 0.5).
    vertices = corners.astype(np.float64) - 0.5
    is_corner = ~is_edge[idx]

    # Direction into free space, per axis in {-1, 0, +1}: the sign of the sum
    # of the (centred) offsets of the free cells around each vertex. Along a
    # straight wall that is the wall normal; at a corner it is the diagonal
    # into the free quadrant. If it cancels out (free cells only on opposite
    # diagonals) fall back to the first free cell.
    free_cells = ~patterns[(slice(None),) + idx]                 # (2^dim, K)
    centred = offsets_arr.astype(np.int64) * 2 - 1               # (2^dim, dim) in {-1, +1}
    free_dir = np.sign(free_cells.T.astype(np.int64) @ centred)  # (K, dim)

    # A zero component leaves a tie on that axis. Along a straight wall the
    # tie is between two free cells and harmless; at a vertex where free cells
    # sit on opposite diagonals (checkerboard, more common in 3-D) the summed
    # direction can point at a blocked cell. Keep ``free_dir`` only if every
    # cell consistent with it is free, otherwise aim at the first free cell.
    towards = offsets_arr[:, None, :] == (free_dir[None, :, :] > 0)          # (2^dim, K, dim)
    consistent = np.all(towards | (free_dir[None, :, :] == 0), axis=2)      # (2^dim, K)
    unsafe = np.any(consistent & ~free_cells, axis=0)
    if unsafe.any():
        first_free = np.argmax(free_cells[:, unsafe], axis=0)
        free_dir[unsafe] = centred[first_free]
    return vertices, is_corner, free_dir


def sample_boundary_nodes(graph_map: "GraphSamplerBase", include_map_edge: bool = True,
                          corners_only: bool = False, inset: float = 1e-3):
    """Nodes on the outline of free space: its corners and the points every
    ``resolution`` along its edges (see ``boundary_vertices``).

    Each node is moved ``inset * resolution`` from its vertex into free space
    (along the wall normal on an edge, along the diagonal at a corner), so it
    is unambiguously inside a free cell: ``world_to_map`` of a point exactly
    on a cell corner rounds half-to-even and can land in a blocked cell,
    which would make ``in_collision`` reject every edge from that node.
    ``inset=0`` keeps the exact vertex positions. The nodes are off the
    cell-centre lattice either way. With ``corners_only`` just the corner
    vertices are returned.

    Returns:
        (nodes, world_points): parallel lists, one entry per kept vertex.
    """
    vertices, is_corner, free_dir = boundary_vertices(graph_map, include_map_edge)
    if corners_only:
        vertices, free_dir = vertices[is_corner], free_dir[is_corner]
    if vertices.shape[0] == 0:
        return [], []

    # Vertices are in the map frame; the inset is a fraction of a cell there too.
    world_points = graph_map.path_map_to_world(vertices + inset * free_dir)
    accepted_nodes = [Node(point, None, 0, 0) for point in world_points]
    return accepted_nodes, world_points
