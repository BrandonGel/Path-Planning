"""Continuous-space path geometry utilities for topological planning.

Ports the geometric layer of CTopPRM (Novosad, Penicka, Vonasek, RA-L 2023;
C++ reference in ``path_planning/cluster/CTopPRM/``) onto ``GraphSampler``:
equal-arclength resampling, segment clearance checks, ESDF-gradient path
shortening (``shorten_path``), the straight-line-homotopy deformability test
(``is_deformable``), and the two path filters used by the pipeline.

Frame convention: all paths and points are in the same frame as
``GraphSampler`` node coordinates (``node.current``), which the rest of the
repo passes directly into the world-coordinate predicates ``in_collision`` /
``min_wall_distance`` (see e.g. ``sipp/graph_generation.py``). Paths are
``np.ndarray`` of shape ``(M, dim)``; inserted push-out waypoints are free
continuous points even when the roadmap itself is discrete.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
from python_motion_planning.common import TYPES

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass
class GeometryConfig:
    """Parameters shared by every geometric predicate in this module.

    Attributes:
        collision_distance_check: Discretization step (world units) for all
            segment sampling, homotopy checks, and push-out probing. The C++
            reference uses 0.1 at fine voxel resolutions; ``CTopPRM`` defaults
            it to ``0.25 * map.resolution`` (~4 samples per cell).
        min_clearance: Required ESDF clearance. ``<= 0`` means "rely on the
            grid's obstacle inflation": segment checks use the exact DDA
            ``in_collision`` instead of sampled ``min_wall_distance``. The
            C++ default of 0.3 would double-count inflation on this repo's
            inflated grids, hence 0.0 here.
        push_out_max_factor: Push-out probes run over
            ``t in [c_eff, push_out_max_factor * c_eff)`` like the C++
            ``[min_clearance, 4 * min_clearance)`` range.
    """

    collision_distance_check: float
    min_clearance: float = 0.0
    push_out_max_factor: float = 4.0


# ---------------------------------------------------------------------------
# Arclength primitives
# ---------------------------------------------------------------------------

def path_length(path: np.ndarray) -> float:
    """Total Euclidean length of a polyline of shape (M, dim)."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def sample_path(path: np.ndarray, num_samples: int) -> np.ndarray:
    """Resample a polyline at ``num_samples`` equal-arclength points.

    Mirrors C++ ``BaseMap::samplePath``: both endpoints are included exactly.

    Args:
        path: Polyline of shape (M, dim).
        num_samples: Number of output samples (>= 2).

    Returns:
        Array of shape (num_samples, dim).
    """
    path = np.asarray(path, dtype=float)
    if len(path) == 1 or num_samples < 2:
        return np.repeat(path[:1], max(num_samples, 1), axis=0)
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < _EPS:
        return np.repeat(path[:1], num_samples, axis=0)
    s = np.linspace(0.0, total, num_samples)
    out = np.empty((num_samples, path.shape[1]))
    for d in range(path.shape[1]):
        out[:, d] = np.interp(s, cum, path[:, d])
    # Guard against interp round-off on the exact endpoints.
    out[0] = path[0]
    out[-1] = path[-1]
    return out


def _segment_samples(p1: np.ndarray, p2: np.ndarray, step: float) -> np.ndarray:
    """Points along [p1, p2] every ``step``, endpoints included."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    dist = float(np.linalg.norm(p2 - p1))
    n = max(int(math.ceil(dist / step)) + 1, 2)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return p1[None, :] + t * (p2 - p1)[None, :]


# ---------------------------------------------------------------------------
# Clearance predicates
# ---------------------------------------------------------------------------

def point_free(map_, point: np.ndarray, cfg: GeometryConfig) -> bool:
    """Whether a single point satisfies the configured clearance."""
    return bool(points_free(map_, np.asarray(point, dtype=float)[None, :], cfg)[0])


def points_free(map_, pts: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    """Vectorized free-point check over an (N, dim) batch.

    The inflation-mode branch does the ``in_collision_point``-style
    OBSTACLE/INFLATION type-map lookup in one numpy pass, but is
    **boundary-inclusive**: a point lying exactly on a cell boundary touches
    up to ``2^dim`` cells and counts as free if ANY of them is free.
    ``generate_planar_map`` (CDT) places roadmap nodes exactly on the
    free/blocked cell boundary (e.g. corner vertices like (49.0, 2.0)), and
    plain ``in_collision_point`` rounding drops them into the blocked cell —
    while the map's DDA edge checks accept them. Out-of-bounds points count
    as not free.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    if cfg.min_clearance > 0.0:
        return np.asarray(map_.min_wall_distance(pts)) >= cfg.min_clearance
    bounds_lo = np.asarray(map_.bounds, dtype=float)[:, 0]
    shape = np.asarray(map_.shape, dtype=int)
    grid = _get_free_grid(map_)

    def cells_free(idx: np.ndarray) -> np.ndarray:
        ok = np.all((idx >= 0) & (idx < shape), axis=1)
        out = np.zeros(len(idx), dtype=bool)
        if np.any(ok):
            out[ok] = grid[tuple(idx[ok].T)]
        return out

    idx_f = (pts - bounds_lo) / float(map_.resolution) - 0.5
    eps = 1e-9
    idx_lo = np.round(idx_f - eps).astype(int)
    idx_hi = np.round(idx_f + eps).astype(int)
    free = cells_free(idx_lo)
    # Boundary points (idx_lo != idx_hi in some dim): OR over the touching
    # cells' verdicts. Non-boundary points are already fully decided above.
    bnd = np.nonzero(np.any(idx_hi != idx_lo, axis=1) & ~free)[0]
    if len(bnd):
        dim = pts.shape[1]
        for combo in product((0, 1), repeat=dim):
            if not any(combo):
                continue  # all-lo case already checked
            idx = idx_lo[bnd].copy()
            for d, use_hi in enumerate(combo):
                if use_hi:
                    idx[:, d] = idx_hi[bnd, d]
            free[bnd] |= cells_free(idx)
    return free


def _get_free_grid(map_) -> np.ndarray:
    """Boolean not-OBSTACLE/not-INFLATION grid, cached on the map object.

    Like the cached ESDF gradients, this goes stale if obstacles change
    after the first query; delete ``map_._ctopprm_free_grid`` to refresh.
    """
    grid = getattr(map_, "_ctopprm_free_grid", None)
    if grid is None:
        data = np.asarray(map_.type_map.data)
        grid = (data != TYPES.OBSTACLE) & (data != TYPES.INFLATION)
        map_._ctopprm_free_grid = grid
    return grid


def segment_free(map_, p1: np.ndarray, p2: np.ndarray, cfg: GeometryConfig) -> bool:
    """Whether the straight segment [p1, p2] satisfies the clearance.

    With ``min_clearance <= 0`` this is the exact DDA occupancy check
    (authoritative for downstream consumers of the paths); otherwise the
    C++ ``isSimplePathFreeBetweenNodes`` sampled-clearance semantics with a
    single batched ESDF query.
    """
    if cfg.min_clearance <= 0.0:
        return not map_.in_collision(tuple(p1), tuple(p2))
    pts = _segment_samples(p1, p2, cfg.collision_distance_check)
    return bool(np.all(map_.min_wall_distance(pts) >= cfg.min_clearance))


def _segment_free_sampled(map_, p1: np.ndarray, p2: np.ndarray, cfg: GeometryConfig) -> bool:
    """Sampled (batched) segment check — fast inner-loop variant.

    Can miss sub-cell corner clips between samples in inflation mode, so
    construction code that relies on it must validate the final polyline
    with ``polyline_free_exact`` before accepting it.
    """
    pts = _segment_samples(p1, p2, cfg.collision_distance_check)
    return bool(np.all(points_free(map_, pts, cfg)))


def polyline_free_exact(map_, path: np.ndarray, cfg: GeometryConfig) -> bool:
    """Exact validity of a waypoint polyline (DDA per consecutive pair).

    Stricter than the planner's own contract: DDA flags even infinitesimal
    cell-corner clips that lie entirely inside the inflation buffer.
    """
    return all(
        segment_free(map_, path[k], path[k + 1], cfg) for k in range(len(path) - 1)
    )


def polyline_free_sampled(
    map_, path: np.ndarray, cfg: GeometryConfig, step_factor: float = 0.5
) -> bool:
    """Sampled validity of a polyline at ``step_factor * collision_distance_check``.

    This is the CTopPRM contract (the C++ ``isSimplePathFreeBetweenNodes`` is
    itself a sampled check): a passing path may still clip cell corners at
    sub-sample depth, which on inflated grids stays inside the inflation
    buffer. One batched query over all segments.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return True
    step = max(cfg.collision_distance_check * step_factor, 1e-6)
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    counts = np.maximum(np.ceil(seg_len / step).astype(int) + 1, 2)
    pts = np.concatenate(
        [
            path[k][None, :]
            + np.linspace(0.0, 1.0, c)[:, None] * (path[k + 1] - path[k])[None, :]
            for k, c in enumerate(counts)
        ]
    )
    return bool(np.all(points_free(map_, pts, cfg)))


def segment_first_collision(
    map_, p1: np.ndarray, p2: np.ndarray, cfg: GeometryConfig
) -> Optional[np.ndarray]:
    """First sampled point along [p1, p2] violating the clearance, or None.

    Used by the shortener to locate where to push out. With the DDA
    predicate, sampled points can all be free while the DDA still reports a
    corner clip; the caller falls back to the segment midpoint in that case
    (the gradient lookup only needs an approximate location).
    """
    pts = _segment_samples(p1, p2, cfg.collision_distance_check)
    bad = np.nonzero(~points_free(map_, pts, cfg))[0]
    return pts[bad[0]] if len(bad) else None


# ---------------------------------------------------------------------------
# ESDF gradient
# ---------------------------------------------------------------------------

def get_esdf_gradients(map_) -> List[np.ndarray]:
    """Per-axis gradients of the map's signed ESDF (world units).

    Computed once per map and cached on the map object; the signed field
    (negative inside obstacles, see ``Grid.update_esdf``) makes the gradient
    point out of obstacles even when evaluated inside one.
    """
    if not getattr(map_, "_esdf_initialized", False):
        map_.update_esdf()
        map_._esdf_initialized = True
    cached = getattr(map_, "_ctopprm_esdf_gradients", None)
    if cached is not None and cached[0] is map_._esdf:
        return cached[1]
    grads = np.gradient(np.asarray(map_._esdf, dtype=float), float(map_.resolution))
    if map_.dim == 1:  # np.gradient returns a bare array for 1-D input
        grads = [grads]
    map_._ctopprm_esdf_gradients = (map_._esdf, list(grads))
    return map_._ctopprm_esdf_gradients[1]


def esdf_gradient(
    map_, point: np.ndarray, grads: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of C++ ``gradientInVoxelCenter``.

    Returns:
        (unit_gradient, cell_center): gradient of the signed ESDF at the
        cell containing ``point`` (zero vector if degenerate) and that
        cell's center in world coordinates.
    """
    if grads is None:
        grads = get_esdf_gradients(map_)
    idx = np.asarray(map_.world_to_map(tuple(point), discrete=True), dtype=int)
    idx = np.clip(idx, 0, np.asarray(map_.shape, dtype=int) - 1)
    g = np.array([grads[d][tuple(idx)] for d in range(map_.dim)], dtype=float)
    norm = np.linalg.norm(g)
    if norm > _EPS:
        g = g / norm
    center = np.asarray(map_.map_to_world(tuple(idx)), dtype=float)
    return g, center


# ---------------------------------------------------------------------------
# Shortening
# ---------------------------------------------------------------------------

def _perpendicular_fallback(map_, s_hat: np.ndarray, center: np.ndarray,
                            probe: float) -> np.ndarray:
    """Unit direction orthogonal to ``s_hat`` when the ESDF gradient is
    parallel to the segment; the sign with larger clearance wins."""
    dim = len(s_hat)
    if dim == 2:
        v = np.array([-s_hat[1], s_hat[0]])
    else:
        axis = np.zeros(dim)
        axis[int(np.argmin(np.abs(s_hat)))] = 1.0
        v = np.cross(s_hat, axis)
        v = v / max(np.linalg.norm(v), _EPS)
    d_plus = float(map_.min_wall_distance(center + v * probe))
    d_minus = float(map_.min_wall_distance(center - v * probe))
    return v if d_plus >= d_minus else -v


def shorten_path(
    map_,
    path: np.ndarray,
    cfg: GeometryConfig,
    forward: bool = True,
    grads: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Gradient push-out shortcutter, port of C++ ``shorten_path`` (:2019).

    Resamples the path at ``collision_distance_check`` arclength, then
    greedily extends straight runs from an anchor; where the straight
    connection is blocked, a waypoint is inserted at the blocking cell's
    center pushed along the ESDF-gradient component perpendicular to the
    segment (``g - (g.s)s``, the C++ ``s x (g x s)`` in any dimension).

    The forward and backward passes give different results; callers apply
    both, as the C++ does. On push-out failure the input is returned
    unchanged (the C++ logs an error and does the same).

    Args:
        map_: GraphSampler (or Grid) providing collision/ESDF queries.
        path: Polyline (M, dim) in node-coordinate frame.
        cfg: Geometry parameters.
        forward: Pass direction.
        grads: Optional precomputed ``get_esdf_gradients(map_)``.

    Returns:
        Shortened polyline (K, dim); endpoints preserved exactly.
    """
    path = np.asarray(path, dtype=float)
    total = path_length(path)
    if len(path) < 2 or total < _EPS:
        return path.copy()
    if grads is None:
        grads = get_esdf_gradients(map_)

    n = int(math.ceil(total / cfg.collision_distance_check)) + 1
    sampled = sample_path(path, max(n, 2))
    if not forward:
        sampled = sampled[::-1]

    # Probe range: the C++ uses [min_clearance, 4*min_clearance); with
    # inflation-based maps (min_clearance == 0) that range would be empty,
    # so probe from one cell outward instead.
    c_eff = max(cfg.min_clearance, float(map_.resolution))
    out = [sampled[0]]
    for i in range(1, len(sampled) - 1):
        anchor = out[-1]
        # Sampled (batched) check in the hot loop; the exact DDA validation
        # of the final polyline below catches any missed corner clip.
        if _segment_free_sampled(map_, anchor, sampled[i], cfg):
            continue
        coll = segment_first_collision(map_, anchor, sampled[i], cfg)
        if coll is None:
            coll = 0.5 * (anchor + sampled[i])
        g, center = esdf_gradient(map_, coll, grads)
        seg = sampled[i] - anchor
        seg_norm = np.linalg.norm(seg)
        if seg_norm < _EPS:
            continue
        s_hat = seg / seg_norm
        v = g - np.dot(g, s_hat) * s_hat
        v_norm = np.linalg.norm(v)
        if v_norm < _EPS:
            v = _perpendicular_fallback(map_, s_hat, center, c_eff)
        else:
            v = v / v_norm
        placed = False
        t = c_eff
        while t < cfg.push_out_max_factor * c_eff:
            cand = center + v * t
            if point_free(map_, cand, cfg):
                out.append(cand)
                placed = True
                break
            t += cfg.collision_distance_check
        if not placed:
            logger.warning(
                "shorten_path: push-out failed near %s; returning path unchanged",
                np.round(coll, 3),
            )
            return path.copy()
    out.append(sampled[-1])
    if not forward:
        out = out[::-1]
    result = np.asarray(out)
    if not polyline_free_sampled(map_, result, cfg):
        logger.debug("shorten_path: validation failed; returning path unchanged")
        return path.copy()
    return result


def greedy_shorten_path(map_, path: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    """Classic farthest-visible shortcutting (fallback shortener).

    Never inserts off-path waypoints, so unlike the gradient push-out it
    cannot fail on coarse grids; selected via ``shortening_mode='greedy'``.
    """
    path = np.asarray(path, dtype=float)
    total = path_length(path)
    if len(path) < 2 or total < _EPS:
        return path.copy()
    n = int(math.ceil(total / cfg.collision_distance_check)) + 1
    sampled = sample_path(path, max(n, 2))
    out = [sampled[0]]
    i = 0
    last = len(sampled) - 1
    while i < last:
        j = last
        while j > i + 1 and not _segment_free_sampled(map_, sampled[i], sampled[j], cfg):
            j -= 1
        out.append(sampled[j])
        i = j
    result = np.asarray(out)
    if not polyline_free_sampled(map_, result, cfg):
        return path.copy()
    return result


# ---------------------------------------------------------------------------
# Homotopy / path filters
# ---------------------------------------------------------------------------

def is_deformable(map_, path1: np.ndarray, path2: np.ndarray, cfg: GeometryConfig) -> bool:
    """Straight-line homotopy test, port of C++ ``isDeformablePath``.

    Both paths are resampled to N equal-arclength correspondences
    (N from the longer path) and each corresponding straight connection
    must be free. Not transitive — callers treating it as an equivalence
    relation (``remove_equivalent_paths``) accept order dependence.
    """
    l1 = path_length(path1)
    l2 = path_length(path2)
    n = int(math.ceil(max(l1, l2) / cfg.collision_distance_check)) + 1
    n = max(n, 2)
    s1 = sample_path(np.asarray(path1, dtype=float), n)
    s2 = sample_path(np.asarray(path2, dtype=float), n)
    # One batched check over all correspondence segments: sample each
    # straight connection at `collision_distance_check` (like the sampled
    # C++ predicate) instead of n per-segment DDA calls — this test only
    # steers clustering/dedup, the exact DDA stays in path construction.
    # Each connection is sampled by its OWN gap (ragged batch), not the max
    # gap: near the shared endpoints gaps shrink to zero and a uniform grid
    # would waste most of its points there.
    gaps = np.linalg.norm(s2 - s1, axis=1)
    counts = np.clip(
        np.ceil(gaps / cfg.collision_distance_check).astype(int) + 1, 2, 512
    )
    total = int(counts.sum())
    seg = np.repeat(np.arange(n), counts)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    within = np.arange(total) - np.repeat(offsets, counts)
    t = (within / (counts[seg] - 1))[:, None]
    pts = s1[seg] * (1.0 - t) + s2[seg] * t
    return bool(np.all(points_free(map_, pts, cfg)))


def remove_too_long_paths(
    paths: List[np.ndarray], endpoints_dist: float, cutoff_ratio: float
) -> List[np.ndarray]:
    """Port of C++ ``removeTooLongPaths`` (:1926).

    Keeps paths no longer than ``cutoff_ratio`` times the shortest
    valid path and no shorter than the straight-line endpoint distance.
    """
    if not paths:
        return []
    lengths = [path_length(p) for p in paths]
    min_allowed = endpoints_dist - 1e-6  # tolerate an exactly-straight path
    valid = [l for l in lengths if l > min_allowed]
    shortest = min(valid) if valid else float("inf")
    return [
        p
        for p, l in zip(paths, lengths)
        if min_allowed <= l <= cutoff_ratio * shortest
    ]


def deformability_key(path1: np.ndarray, path2: np.ndarray) -> tuple:
    """Canonical cache key for ``is_deformable`` on a pair of polylines.

    The test is symmetric under swapping the two paths and under reversing
    both together (same correspondence segments either way), so both
    variants collapse to one key.
    """
    fwd = tuple(sorted((path1.tobytes(), path2.tobytes())))
    rev = tuple(sorted((path1[::-1].tobytes(), path2[::-1].tobytes())))
    return min(fwd, rev)


def is_deformable_cached(
    map_, path1: np.ndarray, path2: np.ndarray, cfg: GeometryConfig,
    cache: Optional[dict],
) -> bool:
    """``is_deformable`` memoized in ``cache`` (keyed on path geometry)."""
    if cache is None:
        return is_deformable(map_, path1, path2, cfg)
    key = deformability_key(path1, path2)
    hit = cache.get(key)
    if hit is None:
        hit = is_deformable(map_, path1, path2, cfg)
        cache[key] = hit
    return hit


def remove_equivalent_paths(
    map_, paths: List[np.ndarray], cfg: GeometryConfig,
    cache: Optional[dict] = None,
) -> List[np.ndarray]:
    """Port of C++ ``removeEquivalentPaths`` (:1957).

    Greedy grouping by pairwise deformability, keeping the shortest
    representative of each group. Order-dependent by design (deformability
    is not transitive); callers pass length-sorted input for determinism.
    ``cache`` (optional dict) memoizes deformability verdicts across calls —
    with unshortened paths the same geometry recurs across pairs and passes.
    """
    paths = list(paths)
    i = 0
    while i < len(paths):
        keep = paths[i]
        keep_len = path_length(keep)
        remaining = []
        for j in range(i + 1, len(paths)):
            if is_deformable_cached(map_, paths[i], paths[j], cfg, cache):
                l_j = path_length(paths[j])
                if l_j < keep_len:
                    keep, keep_len = paths[j], l_j
            else:
                remaining.append(paths[j])
        paths = paths[:i] + [keep] + remaining
        i += 1
    return paths
