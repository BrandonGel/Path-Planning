from itertools import product
from typing import Iterable, Union, Tuple, List, Dict
import time

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates

try:
    from numba import njit as _numba_njit
except Exception:  # pragma: no cover - keeps import compatibility without numba.
    _numba_njit = None

from python_motion_planning.common.env.map.grid import _njit, _grid_world_to_map_int, _grid_is_expandable

def convert_np_to_tuple(array: np.ndarray) -> list:
    """(N, dim) array -> list of N tuples of Python scalars.

    ``tolist()`` converts every element in C in one call; transposing first
    makes ``zip`` regroup the columns back into one tuple per row. ~5x faster
    than a per-element generator.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected an (N, dim) array, got shape {array.shape}")
    return list(zip(*array.T.tolist()))

@_njit(cache=True)
def _world_line_of_sight(p1: np.ndarray, p2: np.ndarray, bounds: np.ndarray, resolution: float, shape: np.ndarray) -> np.ndarray:
    """
    Check if two world points are in line of sight.

    Args:
        p1: Start point of the line (world coordinates).
        p2: End point of the line (world coordinates).

    Returns:
        bool: True if the two points are in line of sight, False otherwise.
    """
    dim = p1.size
    inv_res = 1.0 / resolution

    # Continuous grid-index space: cell k occupies [k, k+1).
    p1_f = np.empty(dim, dtype=np.float64)
    p2_f = np.empty(dim, dtype=np.float64)
    for d in range(dim):
        p1_f[d] = (p1[d] - bounds[d, 0]) * inv_res
        p2_f[d] = (p2[d] - bounds[d, 0]) * inv_res

    # Start/end cells with the map's own convention (round to nearest centre, clip).
    p1_grid = _grid_world_to_map_int(p1, bounds, resolution, shape)
    p2_grid = _grid_world_to_map_int(p2, bounds, resolution, shape)
    current = p1_grid.copy()

    step = np.zeros(dim, dtype=np.int64)
    t_max = np.full(dim, np.inf)
    t_delta = np.full(dim, np.inf)
    for d in range(dim):
        delta = p2_f[d] - p1_f[d]
        if delta > 1e-10:
            step[d] = 1
            t_max[d] = ((p1_grid[d] + 1) - p1_f[d]) / delta
            t_delta[d] = 1.0 / delta
        elif delta < -1e-10:
            step[d] = -1
            t_max[d] = (p1_f[d] - p1_grid[d]) / (-delta)
            t_delta[d] = 1.0 / (-delta)
    
    max_steps = 0
    for d in range(dim):
        max_steps += abs(p2_grid[d] - p1_grid[d])
    out = np.empty((max_steps + 1, dim), dtype=np.int64)
    out[0, :] = current
    count = 1

    # Traverse with early termination
    tie_eps = 1e-9
    for _ in range(max_steps):
        done = True
        for d in range(dim):
            if current[d] != p2_grid[d]:
                done = False
        if done:
            break
        t_min = np.inf
        for d in range(dim):
            if t_max[d] < t_min:
                t_min = t_max[d]
        if not np.isfinite(t_min):
            break
        
        # Step every tied axis that has not reached its end cell (diagonal tie-break; no overshoot at t=1).
        moved = False
        for d in range(dim):
            if t_max[d] <= t_min + tie_eps and current[d] != p2_grid[d]:
                current[d] += step[d]
                t_max[d] += t_delta[d]
                moved = True
        if not moved:
            break
        out[count, :] = current
        count += 1
    return out[:count]

@_njit(cache=True)
def _world_in_collision(p1: np.ndarray, p2: np.ndarray, bounds: np.ndarray, resolution: float, shape: np.ndarray, type_map: np.ndarray, esdf: np.ndarray, obstacle_type: int, inflation_type: int, strict_collision: bool) -> np.ndarray:
    """
    Check if two world points are in collision.

    Args:
        p1: Start point of the line (world coordinates).
        p2: End point of the line (world coordinates).
        bounds: Bounds of the map.
        resolution: Resolution of the map.
        shape: Shape of the map.
        type_map: Type map of the map.
        esdf: ESDF of the map.
        obstacle_type: Obstacle type.
        inflation_type: Inflation type.
        strict_collision: Strict collision flag.

    Returns:
        bool: True if the two points are in collision, False otherwise.
    """

    # Endpoint checks on CELL indices (mirrors upstream _grid_in_collision).
    # Start/end cells with the map's own convention (round to nearest centre, clip).
    p1_grid = _grid_world_to_map_int(p1, bounds, resolution, shape)
    if not _grid_is_expandable(p1_grid, p1_grid, False, shape, type_map, esdf, obstacle_type, inflation_type, False):
        return True
    p2_grid = _grid_world_to_map_int(p2, bounds, resolution, shape)
    if not _grid_is_expandable(p2_grid, p1_grid, True, shape, type_map, esdf, obstacle_type, inflation_type, False):
        return True

    # Check if the points are the same
    dim = p1.size
    same_point = True
    for d in range(dim):
        if p1[d] != p2[d]:
            same_point = False
            break
    if same_point:
        return False

    inv_res = 1.0 / resolution

    # Continuous grid-index space: cell k occupies [k, k+1).
    p1_f = np.empty(dim, dtype=np.float64)
    p2_f = np.empty(dim, dtype=np.float64)
    for d in range(dim):
        p1_f[d] = (p1[d] - bounds[d, 0]) * inv_res
        p2_f[d] = (p2[d] - bounds[d, 0]) * inv_res

    current = p1_grid.copy()
    prev = p1_grid.copy()
    step = np.zeros(dim, dtype=np.int64)
    t_max = np.full(dim, np.inf)
    t_delta = np.full(dim, np.inf)
    for d in range(dim):
        delta = p2_f[d] - p1_f[d]
        if delta > 1e-10:
            step[d] = 1
            t_max[d] = ((p1_grid[d] + 1) - p1_f[d]) / delta
            t_delta[d] = 1.0 / delta
        elif delta < -1e-10:
            step[d] = -1
            t_max[d] = (p1_f[d] - p1_grid[d]) / (-delta)
            t_delta[d] = 1.0 / (-delta)
    
    max_steps = 0
    for d in range(dim):
        max_steps += abs(p2_grid[d] - p1_grid[d])

    # Traverse with early termination
    tie_eps = 1e-9
    for _ in range(max_steps):
        done = True
        for d in range(dim):
            if current[d] != p2_grid[d]:
                done = False
        if done:
            break
        t_min = np.inf
        for d in range(dim):
            if t_max[d] < t_min:
                t_min = t_max[d]
        if not np.isfinite(t_min):
            break
        
        # Step every tied axis that has not reached its end cell (diagonal tie-break; no overshoot at t=1).
        moved = False
        for d in range(dim):
            prev[d] = current[d]
        for d in range(dim):
            if t_max[d] <= t_min + tie_eps and current[d] != p2_grid[d]:
                current[d] += step[d]
                t_max[d] += t_delta[d]
                moved = True
        if not moved:
            return True
        # The check that was missing: every cell entered must be expandable.
        if not _grid_is_expandable(current, prev, True, shape, type_map, esdf, obstacle_type, inflation_type, strict_collision):
            return True
    return False

@_njit(cache=True)
def _grid_vector_is_expandable(
    points: np.ndarray,
    src_points: np.ndarray,
    has_src_point: bool,
    shape: np.ndarray,
    type_map: np.ndarray,
    esdf: np.ndarray,
    obstacle_type: int,
    inflation_type: int,
    strict_collision: bool,
) -> np.ndarray:
    node_num = points.shape[0]
    mask = np.zeros(node_num, dtype=np.bool_)

    assert points.shape == src_points.shape, "Points and src_points must have the same shape"

    for i in range(node_num):
        mask[i] = _grid_is_expandable(
            points[i],
            src_points[i],
            has_src_point,
            shape,
            type_map,
            esdf,
            obstacle_type,
            inflation_type,
            strict_collision,
        )

    return mask

@_njit(cache=True)
def _world_min_wall_distance(pts: np.ndarray, esdf: np.ndarray, inflation_radius: float,
                             bounds: np.ndarray, resolution: float, shape: np.ndarray) -> np.ndarray:
    """World-unit distance from (N, dim) world points to the nearest
    OBSTACLE-or-INFLATION cell, numba kernel behind
    ``GraphSamplerBase.min_wall_distance``.

    The ESDF (world units, cell centres at integer grid coordinates) is
    sampled with multilinear interpolation (bilinear in 2-D, trilinear in
    3-D) at each point's continuous grid coordinate, clamping the corner
    indices to the map like ``scipy.ndimage.map_coordinates(order=1,
    mode="nearest")``. Subtracting ``inflation_radius`` and clipping at 0
    collapses obstacle and inflation cells to distance 0. Points outside the
    map (more than half a cell past the outermost centres) get distance 0.

    ``esdf`` must be flattened C-order (``grid._esdf_flat()``); ``shape`` is
    the map shape as an int64 array.
    """
    n, dim = pts.shape
    out = np.zeros(n, dtype=np.float64)
    inv_res = 1.0 / resolution
    n_corners = 1 << dim
    lo = np.empty(dim, dtype=np.int64)
    frac = np.empty(dim, dtype=np.float64)

    # C-order strides of the flattened esdf
    strides = np.empty(dim, dtype=np.int64)
    acc = 1
    for d in range(dim - 1, -1, -1):
        strides[d] = acc
        acc *= shape[d]

    for i in range(n):
        inside = True
        for d in range(dim):
            g = (pts[i, d] - bounds[d, 0]) * inv_res - 0.5      # continuous grid coordinate
            if g < -0.5 or g > shape[d] - 0.5:
                inside = False
                break
            f = np.floor(g)
            lo[d] = np.int64(f)
            frac[d] = g - f
        if not inside:
            continue                                            # out of bounds -> 0

        value = 0.0
        for c in range(n_corners):
            w = 1.0
            flat = 0
            for d in range(dim):
                if (c >> d) & 1:
                    idx = lo[d] + 1
                    w *= frac[d]
                else:
                    idx = lo[d]
                    w *= 1.0 - frac[d]
                if idx < 0:                                     # mode="nearest"
                    idx = 0
                elif idx >= shape[d]:
                    idx = shape[d] - 1
                flat += idx * strides[d]
            value += w * esdf[flat]

        dist = value - inflation_radius
        out[i] = dist if dist > 0.0 else 0.0
    return out
