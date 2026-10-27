import numpy as np
from itertools import product
from python_motion_planning.common import TYPES
SAMPLING_TYPES = ('random', 'halton', 'distributed')

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

def validate_sampling_type(sampling_type: str) -> str:
    """``sampling_type`` normalised to one of ``SAMPLING_TYPES``; a roadmap
    type such as ``halton_prm`` counts as ``halton``."""
    sampling_type = str(sampling_type).lower()
    if 'halton' in sampling_type:
        return 'halton'
    if sampling_type not in SAMPLING_TYPES:
        raise ValueError(f"sampling_type must be one of {SAMPLING_TYPES}, got {sampling_type!r}")
    return sampling_type
