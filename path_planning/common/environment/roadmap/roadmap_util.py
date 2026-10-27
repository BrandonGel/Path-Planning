from numba import njit, prange
import numpy as np
from scipy.sparse import csr_array
from path_planning.common.environment.map.numbaUtil import _world_in_collision
from typing import Tuple
from scipy.sparse.csgraph import dijkstra as _csgraph_dijkstra

@njit(parallel=True, cache=True)
def _filter_edges_kernel(pts, u, v, min_len, max_len, check_collision,
                         bounds, res, shape, type_map, esdf, obs_t, inf_t, strict):
    """Keep edge ``e`` when ``min_len <= |pts[u[e]] - pts[v[e]]| <= max_len``
    and (if ``check_collision``) the segment is free in both directions, so
    an endpoint outside the map or inside an obstacle kills the edge
    whichever end it is. Returns the keep mask and every edge's length.
    Edges are independent.
    """
    E = len(u)
    keep = np.zeros(E, np.bool_)
    w = np.zeros(E)
    for e in prange(E):
        d2 = 0.0
        for k in range(pts.shape[1]):
            diff = pts[u[e], k] - pts[v[e], k]
            d2 += diff * diff
        d = np.sqrt(d2)
        w[e] = d
        if d < min_len or d > max_len:
            continue
        if check_collision and (
            _world_in_collision(pts[u[e]], pts[v[e]], bounds, res, shape, type_map, esdf, obs_t, inf_t, strict)
            or _world_in_collision(pts[v[e]], pts[u[e]], bounds, res, shape, type_map, esdf, obs_t, inf_t, strict)
        ):
            continue
        keep[e] = True
    return keep, w


@njit(cache=True)
def _khop_ball(indptr, indices, data, src, k, n):
    """BFS from ``src`` over at most ``k`` hops on a CSR adjacency. Returns the
    ball (node ids in BFS order, ``src`` first), each node's hop count (``-1``
    outside the ball) and the length of its BFS-tree path from ``src``, which
    bounds its true shortest-path length from above. Touches only the ball."""
    hop = np.full(n, -1, np.int64)
    bound = np.zeros(n)
    order = np.empty(n, np.int64)
    hop[src] = 0
    order[0] = src
    head, tail = 0, 1
    while head < tail:
        u = order[head]
        head += 1
        if hop[u] == k:
            continue
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            if hop[v] < 0:
                hop[v] = hop[u] + 1
                bound[v] = bound[u] + data[j]
                order[tail] = v
                tail += 1
    return order[:tail], hop, bound


def _unique_undirected(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected edges ``(eu < ev)`` of the pair list and, for every
    input slot, the id of its edge. Callers drop self-loops beforehand."""
    u = np.asarray(u, dtype=np.int64).ravel()
    v = np.asarray(v, dtype=np.int64).ravel()
    pairs = np.stack([np.minimum(u, v), np.maximum(u, v)], axis=1)
    if len(pairs) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return uniq[:, 0], uniq[:, 1], inverse.ravel()


def _symmetric_csr(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> csr_array:
    """CSR adjacency holding every undirected edge in both directions."""
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.concatenate([w, w])
    adj = csr_array((data, (rows, cols)), shape=(n, n))
    adj.sort_indices()
    return adj

def shortest_paths(sources, adj, limit: float = np.inf, return_predecessors: bool = False):
    """Dijkstra over the roadmap from one or several source nodes.

    Returns ``dist`` of shape ``(len(sources), num_nodes)`` (``inf`` where
    unreachable, ``limit`` caps the search radius) and, when asked, the
    ``predecessors`` array of the same shape (``-9999`` for the source and
    unreachable nodes, scipy's convention). A scalar ``sources`` gives 1-D
    results.
    """
    num_nodes = adj.shape[0]
    scalar = np.isscalar(sources)
    idx = np.atleast_1d(np.asarray(sources, dtype=np.int64))
    if num_nodes == 0:
        raise ValueError("shortest_paths: the roadmap has no nodes")
    if idx.size and (idx.min() < 0 or idx.max() >= num_nodes):
        raise IndexError(f"shortest_paths: source out of range [0, {num_nodes})")
    out = _csgraph_dijkstra(adj, directed=True, indices=idx, limit=limit,
                            return_predecessors=return_predecessors)
    if return_predecessors:
        dist, pred = out
        return (dist[0], pred[0]) if scalar else (dist, pred)
    return out[0] if scalar else out

