from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from scipy.spatial import KDTree

from path_planning.common.environment.node import Node
from path_planning.common.environment.map.numbaUtil import _world_in_collision
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase, RoadmapConfig

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


@njit(parallel=True, cache=True)
def _prm_edges_kernel(q, pts, dist, idx, min_len, max_len, k_cap,
                      bounds, res, shape, type_map, esdf, obs_t, inf_t, strict):
    """Per query row: walk the distance-sorted candidates, keep collision-free
    ones inside [min_len, max_len], stop after k_cap accepted. Rows are
    independent. ``q[ii]`` is the row's own point, ``pts[idx[ii, jj]]`` a
    candidate; ``dist`` may hold ``inf`` past a ``distance_upper_bound``.

    ``complete[ii]`` is True when no candidate outside the row could still be
    chosen: the walk ran past ``max_len``, or the quota was filled at a
    distance strictly below the row's last distance (so no tie is cut off).
    """
    n, m = dist.shape
    K = k_cap if k_cap > 0 else m
    out_idx = np.full((n, K), -1, np.int64)
    out_w = np.zeros((n, K))
    count = np.zeros(n, np.int64)
    complete = np.zeros(n, np.bool_)
    for ii in prange(n):
        c = 0
        d_stop = -1.0
        hit_max = False
        for jj in range(1, m):              # column 0 is the node itself
            d = dist[ii, jj]
            if d > max_len:
                hit_max = True              # sorted; inf (beyond bound) lands here too
                break
            if d < min_len:
                continue
            j = idx[ii, jj]
            if _world_in_collision(q[ii], pts[j], bounds, res, shape,
                                   type_map, esdf, obs_t, inf_t, strict):
                continue
            out_idx[ii, c] = j
            out_w[ii, c] = d
            c += 1
            if c >= K:
                d_stop = d
                break
        count[ii] = c
        complete[ii] = hit_max or (c >= K and d_stop < dist[ii, m - 1])
    return out_idx, out_w, count, complete


def _to_csr(out_idx: np.ndarray, out_w: np.ndarray, count: np.ndarray) -> csr_array:
    """Pack fixed-width kernel output (first ``count[i]`` slots of row i) into CSR."""
    n = len(count)
    keep = np.arange(out_idx.shape[1])[None, :] < count[:, None]
    indptr = np.concatenate(([0], np.cumsum(count)))
    return csr_array((out_w[keep], out_idx[keep], indptr), shape=(n, n))


def _canonical_rows(dist: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort each row by (distance, node index) so ties are deterministic
    and independent of the KD-tree and of the query size."""
    order = np.lexsort((idx, dist), axis=-1)
    return (np.ascontiguousarray(np.take_along_axis(dist, order, 1)),
            np.ascontiguousarray(np.take_along_axis(idx, order, 1)))


class PRM(RoadmapBase):
    """k-nearest-neighbour probabilistic roadmap over the given samples."""

    name = "prm"

    def __init__(self, graph_map: "GraphSamplerBase", config: dict | RoadmapConfig | None = None):
        super().__init__(graph_map, config)

    def generate_prm(self, k_query: int | None = None, grow: int | None = None) -> csr_array:
        """k-nearest PRM edges with an adaptive candidate query.

        Each round queries ``k`` candidates per unfinished node within
        ``max_edge_length``, runs the edge kernel, and keeps the rows the
        kernel proves complete. The rest are re-queried with ``k * grow``
        until ``k`` reaches the node count. Memory is O(N * k) instead of
        O(N^2) for a full query. ``k_query`` sets the first ``k``; default
        ``4 * num_neighbors + 1``. Ties are broken by node index. ``k_query``
        and ``grow`` default to the config's values.
        """
        k_query = self.config.k_query if k_query is None else k_query
        grow = self.config.grow if grow is None else int(grow)
        n = self.num_nodes
        if n == 0:
            self._adj = csr_array((0, 0), dtype=float)
            return self.adj

        K = self.num_neighbors if self.num_neighbors > 0 else n
        k = n if k_query is None and self.num_neighbors <= 0 else (k_query or 4 * self.num_neighbors + 1)
        k = int(min(max(k, 2), n))
        args = (float(self.min_edge_length), float(self.max_edge_length), int(K)) + self._collision_args()

        tree = KDTree(self.nodes)
        # KDTree's bound is exclusive; nudge it so edges exactly at
        # max_edge_length are returned (the kernel applies the inclusive test).
        upper = np.nextafter(float(self.max_edge_length), np.inf)
        rows = np.arange(n)
        out_idx = np.full((n, K), -1, dtype=np.int64)
        out_w = np.zeros((n, K))
        count = np.zeros(n, dtype=np.int64)
        while True:
            dist, idx = tree.query(self.nodes[rows], k=k, workers=-1,
                                   distance_upper_bound=upper)
            dist = np.atleast_2d(np.asarray(dist, dtype=np.float64))
            idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
            idx[idx >= n] = 0                       # placeholder for inf entries, never read
            dist, idx = _canonical_rows(dist, idx)

            oi, ow, c, done = _prm_edges_kernel(self.nodes[rows], self.nodes, dist, idx, *args)
            if k >= n:
                done[:] = True                      # nothing left to grow into
            out_idx[rows[done]] = oi[done]
            out_w[rows[done]] = ow[done]
            count[rows[done]] = c[done]
            rows = rows[~done]
            if len(rows) == 0:
                break
            k = int(min(n, k * grow))

        self._adj = _to_csr(out_idx, out_w, count)
        return self.adj

    generate = generate_prm
