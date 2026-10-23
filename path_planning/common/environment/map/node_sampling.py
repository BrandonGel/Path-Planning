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
    """Lattice vertices on the outline of free space.

    A vertex is on the outline when its 2^dim surrounding cells contain both
    a free cell (neither OBSTACLE nor INFLATION) and a blocked one; the map
    edge counts as blocked when ``include_map_edge`` is True. It is an *edge*
    vertex when a single axis splits the blocked cells from the free ones (a
    flat wall), otherwise a *corner*. In 3-D the corner flag also covers the
    straight edges of a wall.

    Returns:
        vertices: (K, dim) float64 map-frame points, row-major. Integer
            (r, c) is the centre of cell (r, c), so the lower corner of cell
            (i, j) is (i - 0.5, j - 0.5). Use ``path_map_to_world`` for world
            points.
        is_corner: (K,) bool.
        free_dir: (K, dim) int in {-1, 0, +1}, direction from the vertex
            into free space (wall normal on an edge, diagonal at a corner).
    """
    blocked = (type_map == TYPES.OBSTACLE) | (type_map == TYPES.INFLATION)
    dim = blocked.ndim
    # Outside the map is blocked (edge is part of the outline) or free.
    padded = np.pad(blocked, 1, mode="constant", constant_values=include_map_edge)
    vertex_shape = tuple(n + 1 for n in blocked.shape)

    # patterns[k]: blocked state of the cell at corner offset k of each vertex
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
    # Corner (i, j) of the lattice is map-frame (i - 0.5, j - 0.5).
    vertices = corners.astype(np.float64) - 0.5
    is_corner = ~is_edge[idx]

    # free_dir: sign of the summed centred offsets of the free cells around
    # each vertex (wall normal on an edge, diagonal at a corner).
    free_cells = ~patterns[(slice(None),) + idx]                 # (2^dim, K)
    centred = offsets_arr.astype(np.int64) * 2 - 1               # (2^dim, dim) in {-1, +1}
    free_dir = np.sign(free_cells.T.astype(np.int64) @ centred)  # (K, dim)

    # When free cells sit on opposite diagonals the sum can point at a
    # blocked cell; then aim at the first free cell instead.
    towards = offsets_arr[:, None, :] == (free_dir[None, :, :] > 0)          # (2^dim, K, dim)
    consistent = np.all(towards | (free_dir[None, :, :] == 0), axis=2)      # (2^dim, K)
    unsafe = np.any(consistent & ~free_cells, axis=0)
    if unsafe.any():
        first_free = np.argmax(free_cells[:, unsafe], axis=0)
        free_dir[unsafe] = centred[first_free]
    return vertices, is_corner, free_dir


def density_probability(d, d_min, d_opt, sigma, floor):
    """Gaussian-shaped acceptance probability for wall distance ``d``,
    peaking at ``d_opt`` and never below ``floor``; 0 where ``d < d_min``."""
    d_arr = np.asarray(d, dtype=float)
    prob = floor + (1.0 - floor) * np.exp(-((d_arr - d_opt) ** 2) / (2.0 * sigma ** 2))
    prob = np.where(d_arr < d_min, 0.0, prob)
    if np.isscalar(d) or np.ndim(d) == 0:
        return float(prob)
    return prob

def random_sampling(graph_map: "GraphSamplerBase",
    n_samples,
):
    """``n_samples`` uniform points over the map bounds, keeping those in
    expandable cells. Discrete space snaps them to cell centres.

    Returns:
        (nodes, points): parallel lists of accepted nodes and world tuples.
    """
    bounds = np.asarray(graph_map.bounds, dtype=float)
    dim = graph_map.dim
    normalized_points = np.random.random((n_samples,dim))
    world_points = normalized_points * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
    grid_points = graph_map.path_world_to_map(world_points)
    expandable_mask = graph_map.is_expandable(grid_points)

    accepted_cells = grid_points[expandable_mask]
    inflated_esdf = graph_map.inflated_esdf[tuple(accepted_cells.T)]
    accepted_points = graph_map.path_map_to_world(accepted_cells) if graph_map.use_discrete_space else world_points[expandable_mask]
    accepted_points = convert_np_to_tuple(accepted_points)
    accepted_nodes = [Node(point, None, 0, 0, radius) for point, radius in zip(accepted_points, inflated_esdf)]
    return accepted_nodes, accepted_points

def distributed_sampling(graph_map: "GraphSamplerBase",
    n_samples,
    prob_map: np.ndarray,
    samp_from_prob_map_ratio: float = 0.5,
):
    """``n_samples`` cells: a ``samp_from_prob_map_ratio`` share drawn from
    ``prob_map`` (one weight per cell), the rest from its complement
    ``1 - p``. Expandable cells are kept; discrete space uses the cell
    centre, continuous space adds a uniform +-0.5 cell jitter.

    Returns:
        (nodes, points, num_rejected_weighted): accepted nodes and world
        tuples, prob-map samples first, plus how many prob-map samples were
        rejected (used by the caller to re-balance the ratio).
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
        # Complement is all zeros when one cell holds all the mass: use uniform.
        unchosen_flat = unchosen_flat / total if total > 0 else np.full_like(prob_map_flat, 1.0 / prob_map_flat.size)
        idx = np.random.choice(prob_map_flat.size, size=num_unchosen_samples, p=unchosen_flat)
        cells[num_prob_map_samples:] = np.column_stack(np.unravel_index(idx, prob_map.shape))

    expandable_mask = graph_map.is_expandable(cells)
    accepted_cells = cells[expandable_mask]
    num_rejected_weighted = int(num_prob_map_samples - expandable_mask[:num_prob_map_samples].sum())

    if graph_map.use_discrete_space:
        accepted_points = graph_map.path_map_to_world(accepted_cells)
    else:
        jitter = np.random.random((len(accepted_cells), dim)) - 0.5
        accepted_points = graph_map.path_map_to_world(accepted_cells + jitter)
    inflated_esdf = graph_map.inflated_esdf[tuple(accepted_cells.T)]
    accepted_points = convert_np_to_tuple(accepted_points)
    accepted_nodes = [Node(point, None, 0, 0, radius) for point, radius in zip(accepted_points, inflated_esdf)]

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
    """``n_samples * oversample`` Halton points over the map bounds, thinned
    by ``density_probability`` of their wall distance, then filtered to
    expandable cells. Discrete space snaps them to cell centres.

    Pass the returned ``halton_sampler`` back in to continue the sequence;
    a fresh sampler repeats the same points.

    Returns:
        (nodes, points, halton_sampler): accepted nodes, world tuples, and
        the sampler to reuse.
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

    accepted_cells = grid_points[expandable_mask]
    inflated_esdf = graph_map.inflated_esdf[tuple(accepted_cells.T)]
    accepted_points = graph_map.path_map_to_world(accepted_cells) if graph_map.use_discrete_space else world_points[expandable_mask]
    accepted_points = convert_np_to_tuple(accepted_points)
    accepted_nodes = [Node(point, None, 0, 0, radius) for point, radius in zip(accepted_points, inflated_esdf)]

    return accepted_nodes, accepted_points, halton_sampler

def sample_grid_nodes(graph_map: "GraphSamplerBase"):
    """One node at the centre of every expandable cell.

    Returns:
        (nodes, points): parallel lists of nodes and world tuples, row-major.
    """
    grid_shape = graph_map.shape
    grid_ranges = [np.arange(grid_shape[d]) for d in range(graph_map.dim)]
    grid_points = np.array(np.meshgrid(*grid_ranges, indexing='ij')).reshape(graph_map.dim,-1).T
    world_grid_points = graph_map.path_map_to_world(grid_points)

    expandable_mask = graph_map.is_expandable(grid_points)
    accepted_cells = grid_points[expandable_mask]
    inflated_esdf = graph_map.inflated_esdf[tuple(accepted_cells.T)]
    accepted_points = convert_np_to_tuple(world_grid_points[expandable_mask])
    accepted_nodes = [Node(point, None, 0, 0, radius) for point, radius in zip(accepted_points, inflated_esdf)]
    return accepted_nodes, accepted_points

def sample_boundary_nodes(graph_map: "GraphSamplerBase", include_map_edge: bool = True,
                          corners_only: bool = False, inset: float = 1e-3):
    """Nodes on the outline of free space: its corners and one point every
    ``resolution`` along its edges (see ``boundary_vertices``).

    Each node is moved ``inset`` cells from its vertex into free space so it
    lies inside a free cell; a point exactly on a corner can round into a
    blocked cell and lose all its edges. ``inset=0`` keeps the exact
    vertices. ``corners_only`` drops the edge vertices.

    Returns:
        (nodes, points): parallel lists of nodes and world tuples.
    """
    vertices, is_corner, free_dir = boundary_vertices(graph_map.type_map.data, include_map_edge)
    if corners_only:
        vertices, free_dir = vertices[is_corner], free_dir[is_corner]
    if vertices.shape[0] == 0:
        return [], []
    world_points = graph_map.path_map_to_world(vertices + inset * free_dir)
    cells = graph_map.path_world_to_map(world_points)
    inflated_esdf = graph_map.inflated_esdf[tuple(cells.T)]
    world_points = convert_np_to_tuple(world_points)
    accepted_nodes = [Node(point, None, 0, 0, radius) for point, radius in zip(world_points, inflated_esdf)]
    return accepted_nodes, world_points
