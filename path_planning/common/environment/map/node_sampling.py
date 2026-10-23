import numpy as np
from itertools import product
from path_planning.common.environment.map.numbaUtil import _world_min_wall_distance, convert_np_to_tuple
from python_motion_planning.common.env.map.grid import _grid_path_map_to_world, _grid_path_world_to_map_int, _grid_path_world_to_map_float
from path_planning.common.environment.node import Node
from python_motion_planning.common import TYPES
from typing import List
from typing import TYPE_CHECKING
from scipy.stats import qmc

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase 


def boundary_vertices(type_map: np.ndarray, include_map_edge: bool = True):
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


def density_probability(d, d_min, d_opt, sigma, floor):
    """Vectorized Gaussian-shaped probability of accepting a sample with wall distance `d`.
    Returns 0 wherever d < d_min."""
    d_arr = np.asarray(d, dtype=float)
    prob = floor + (1.0 - floor) * np.exp(-((d_arr - d_opt) ** 2) / (2.0 * sigma ** 2))
    prob = np.where(d_arr < d_min, 0.0, prob)
    if np.isscalar(d) or np.ndim(d) == 0:
        return float(prob)
    return prob


def get_random_nodes(self) -> List[Node]:
    if len(self.nodes) >= self.sample_num:
        return self.nodes[:self.sample_num]
    print("Warning: number of nodes is less than sample num")
    return self.nodes

def random_sampling(graph_map: "GraphSamplerBase",
    n_samples,
):
    bounds = np.asarray(graph_map.bounds, dtype=float)
    dim = graph_map.dim
    normalized_points = np.random.random((n_samples,dim))
    world_points = normalized_points * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
    grid_points = graph_map.path_world_to_map(world_points)
    expandable_mask = graph_map.is_expandable(grid_points)

    accepted_points = graph_map.path_map_to_world(grid_points[expandable_mask]) if graph_map.use_discrete_space else world_points[expandable_mask] 
    accepted_points = convert_np_to_tuple(accepted_points)
    accepted_nodes = [Node(point, None, 0, 0) for point in accepted_points]
    return accepted_nodes, accepted_points

def distributed_sampling(graph_map: "GraphSamplerBase",
    n_samples,
    prob_map: np.ndarray,
    samp_from_prob_map_ratio: float = 0.5,
):
    """``n_samples`` world points: a ``samp_from_prob_map_ratio`` share drawn
    from ``prob_map`` (one weight per cell), the rest from its complement
    ``1 - p`` so under-weighted cells are still covered.

    Each sample is the chosen cell's centre plus a uniform jitter of
    +-0.5 cell on every axis, so it stays inside that cell at any resolution.

    Returns:
        (nodes, points, num_rejected_weighted): the accepted (collision-free)
        nodes and their world points as tuples, prob-map samples first, and
        how many of the prob-map samples were rejected.
    """
    dim = graph_map.dim
    n_samples = int(n_samples)
    num_prob_map_samples = int(n_samples * samp_from_prob_map_ratio)
    num_unchosen_samples = n_samples - num_prob_map_samples

    prob_map = np.asarray(prob_map, dtype=np.float64)
    if prob_map.shape != tuple(graph_map.shape):
        raise ValueError(f"prob_map shape {prob_map.shape} does not match map shape {tuple(graph_map.shape)}")
    prob_map_flat = prob_map.ravel()
    prob_map_flat = prob_map_flat / prob_map_flat.sum()

    cells = np.zeros((n_samples, dim), dtype=np.int64)
    if num_prob_map_samples > 0:
        idx = np.random.choice(prob_map_flat.size, size=num_prob_map_samples, p=prob_map_flat)
        cells[:num_prob_map_samples] = np.column_stack(np.unravel_index(idx, prob_map.shape))

    if num_unchosen_samples > 0:
        unchosen_flat = 1.0 - prob_map_flat
        total = unchosen_flat.sum()
        unchosen_flat = unchosen_flat / total if total > 0 else np.full_like(prob_map_flat, 1.0 / prob_map_flat.size)
        idx = np.random.choice(prob_map_flat.size, size=num_unchosen_samples, p=unchosen_flat)
        cells[num_prob_map_samples:] = np.column_stack(np.unravel_index(idx, prob_map.shape))

    expandable_mask = graph_map.is_expandable(cells)
    accepted_cells = cells[expandable_mask]
    # Number of prob-map samples that were rejected; the caller uses it to
    # re-balance samp_from_prob_map_ratio for the next round.
    num_rejected_weighted = int(num_prob_map_samples - expandable_mask[:num_prob_map_samples].sum())

    if graph_map.use_discrete_space:
        # Discrete nodes sit on the cell centres.
        accepted_points = graph_map.path_map_to_world(accepted_cells)
    else:
        # Continuous nodes get a uniform +-0.5 cell jitter so they cover the
        # whole cell rather than only its centre.
        jitter = np.random.random((len(accepted_cells), dim)) - 0.5
        accepted_points = graph_map.path_map_to_world(accepted_cells + jitter)
    accepted_points = convert_np_to_tuple(accepted_points)
    accepted_nodes = [Node(point, None, 0, 0) for point in accepted_points]

    return accepted_nodes, accepted_points, num_rejected_weighted

def sample_halton_nodes(graph_map: "GraphSamplerBase",
        n_samples,
        d_min=0.3,
        d_opt=0.4,
        sigma=0.5,
        floor=0.2,
        halton_sampler=None,
        oversample=1,
    ):
    """Draw `n_samples` Halton points scaled to `bounds`, then apply distance-based
    rejection sampling so points near obstacles/inflation are unlikely to be kept.

    The Halton sampler is persisted across calls — pass the returned sampler back
    in to continue the deterministic sequence (otherwise repeated calls would
    return the same points).
    """
    bounds = np.asarray(graph_map.bounds, dtype=float)
    dim = graph_map.dim
    if halton_sampler is None:
        halton_sampler = qmc.Halton(d=dim, scramble=False)

    n_draw = int(max(1, n_samples * max(1, oversample)))
    raw_samples = halton_sampler.random(n_draw)
    world_point_samples = qmc.scale(raw_samples, bounds[:, 0], bounds[:, 1])

    dists = _world_min_wall_distance(
        np.ascontiguousarray(world_point_samples, dtype=np.float64),
        graph_map._esdf_flat(),
        float(graph_map.inflation_radius),
        bounds,
        float(graph_map.resolution),
        graph_map._shape_array,
    )
    probs = density_probability(dists, d_min, d_opt, sigma, floor)
    halton_accept = np.random.random(len(world_point_samples)) < probs
    world_points = world_point_samples[halton_accept]
    grid_points = graph_map.path_world_to_map(world_points)
    expandable_mask = graph_map.is_expandable(grid_points)

    accepted_points = graph_map.path_map_to_world(grid_points[expandable_mask]) if graph_map.use_discrete_space else world_points[expandable_mask] 
    accepted_points = convert_np_to_tuple(accepted_points)
    accepted_nodes = [Node(point, None, 0, 0) for point in accepted_points]

    return accepted_nodes, accepted_points, halton_sampler

def sample_grid_nodes(graph_map: "GraphSamplerBase"):

    grid_shape = graph_map.shape
    grid_ranges = [np.arange(grid_shape[d]) for d in range(graph_map.dim)]
    grid_points = np.array(np.meshgrid(*grid_ranges, indexing='ij')).reshape(graph_map.dim,-1).T
    world_grid_points = graph_map.path_map_to_world(grid_points)

    expandable_mask = graph_map.is_expandable(grid_points)
    accepted_points = convert_np_to_tuple(world_grid_points[expandable_mask])
    accepted_nodes = [Node(point, None, 0, 0) for point in accepted_points]
    return accepted_nodes, accepted_points

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
    vertices, is_corner, free_dir = boundary_vertices(graph_map.type_map.data, include_map_edge)
    if corners_only:
        vertices, free_dir = vertices[is_corner], free_dir[is_corner]
    if vertices.shape[0] == 0:
        return [], []

    # Vertices are in the map frame; the inset is a fraction of a cell there too.
    world_points = graph_map.path_map_to_world(vertices + inset * free_dir)
    world_points = convert_np_to_tuple(world_points)
    accepted_nodes = [Node(point, None, 0, 0) for point in world_points]
    return accepted_nodes, world_points
