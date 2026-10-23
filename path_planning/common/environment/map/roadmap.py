from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from scipy.spatial import KDTree
from python_motion_planning.common.env.map.grid import TYPES

from path_planning.common.environment.node import Node
from path_planning.common.environment.map.numbaUtil import _world_in_collision

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


# Numpy based roadmap class
class Roadmap:
    """Roadmap as an (N, dim) point array plus a CSR adjacency.

    ``adj.indices[adj.indptr[u]:adj.indptr[u+1]]`` are the neighbours of ``u``
    and ``adj.data`` the matching edge lengths. ``road_map`` / ``edge_weights``
    give the list-of-lists view the rest of the codebase reads.
    """

    def __init__(self, graph_map: "GraphSamplerBase"):
        self.graph_map = graph_map

        self._dim = self.graph_map.dim
        self._num_nodes = 0

        self.min_edge_length = graph_map.min_edge_length
        self.max_edge_length = graph_map.max_edge_length
        self.num_neighbors = int(graph_map.num_neighbors)

        self.nodes = np.zeros((0, self.dim))
        self.clearance = np.zeros(0)
        self.adj = csr_array((0, 0), dtype=float)

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_edges(self):
        return int(self.adj.nnz)

    @property
    def dim(self):
        return self._dim

    def _rows(self, values: np.ndarray) -> list:
        if self.num_nodes == 0:
            return []
        return [r.tolist() for r in np.split(values, self.adj.indptr[1:-1])]

    @property
    def road_map(self) -> List[List[int]]:
        return self._rows(self.adj.indices)

    @property
    def edge_weights(self) -> List[List[float]]:
        return self._rows(self.adj.data)

    def neighbors(self, u: int) -> np.ndarray:
        return self.adj.indices[self.adj.indptr[u]:self.adj.indptr[u + 1]]

    def in_collision(self, p1, p2=None) -> bool:
        return self.graph_map.in_collision(p1, p2)

    def set_nodes(self, samples: List[Node]):
        n = len(samples)
        self._num_nodes = n
        self.nodes = np.asarray([s.current for s in samples], dtype=float).reshape(n, self.dim)
        self.clearance = np.asarray([s.max_clearance for s in samples], dtype=float)
        self.adj = csr_array((n, n), dtype=float)

    def generate_roadmap(self, samples: List[Node]) -> csr_array:
        self.set_nodes(samples)
        return self.generate_prm()

    def generate_grid(self) -> csr_array:
        """Cardinal-neighbour roadmap for nodes on the Cartesian cell grid.

        Every node must sit at a cell centre and no two nodes may share a cell.
        Each node is linked to the nodes in the adjacent cells one step along
        each axis in both directions, so at most ``2 * dim`` neighbours (4 in
        2D). All edges have the uniform length ``resolution``; the min/max
        edge-length band does not apply. The segment between two adjacent
        free-cell centres stays inside those two cells, so no line-of-sight
        check is needed.
        """
        n = self.num_nodes
        if n == 0:
            self.adj = csr_array((0, 0), dtype=float)
            return self.adj

        m = self.graph_map
        shape = tuple(int(s) for s in m.shape)
        cells = np.rint(np.asarray(m.path_world_to_map(self.nodes), dtype=float)).astype(np.int64)
        cells = cells.reshape(n, self.dim)
        centres = np.asarray(m.path_map_to_world(cells), dtype=float).reshape(n, self.dim)
        off_grid = ~np.all(np.isclose(centres, self.nodes, rtol=0.0, atol=1e-6 * m.resolution), axis=1)
        if off_grid.any():
            raise ValueError(f"generate_grid: {int(off_grid.sum())} node(s) are not at cell centres")

        flat = np.ravel_multi_index(tuple(cells.T), shape)
        if len(np.unique(flat)) != n:
            raise ValueError("generate_grid: several nodes share a cell")
        cell_to_node = np.full(shape, -1, dtype=np.int64)
        cell_to_node.flat[flat] = np.arange(n)

        rows, cols = [], []
        for axis in range(self.dim):
            for step in (-1, 1):
                nb = cells.copy()
                nb[:, axis] += step
                inside = (nb[:, axis] >= 0) & (nb[:, axis] < shape[axis])
                j = np.full(n, -1, dtype=np.int64)
                j[inside] = cell_to_node[tuple(nb[inside].T)]
                has = j >= 0
                rows.append(np.nonzero(has)[0])
                cols.append(j[has])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        # Adjacent centres are exactly one cell apart: uniform weight, no
        # min/max edge-length band.
        w = np.full(len(rows), float(m.resolution))
        self.adj = csr_array((w, (rows, cols)), shape=(n, n))
        self.adj.sort_indices()
        return self.adj

    def _collision_args(self) -> tuple:
        m = self.graph_map
        return (np.asarray(m.bounds, dtype=np.float64), float(m.resolution),
                np.asarray(m.shape, dtype=np.int64), m._type_map_flat(), m._esdf_flat(),
                int(TYPES.OBSTACLE), int(TYPES.INFLATION), bool(m.strict_collision))

    def generate_prm(self, k_query: int | None = None, grow: int = 4) -> csr_array:
        """k-nearest PRM edges with an adaptive candidate query.

        Each round queries ``k`` candidates per unfinished node within
        ``max_edge_length``, runs the edge kernel, and keeps the rows the
        kernel proves complete. The rest are re-queried with ``k * grow``
        until ``k`` reaches the node count. Memory is O(N * k) instead of
        O(N^2) for a full query. ``k_query`` sets the first ``k``; default
        ``4 * num_neighbors + 1``. Ties are broken by node index.
        """
        n = self.num_nodes
        if n == 0:
            self.adj = csr_array((0, 0), dtype=float)
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

        self.adj = _to_csr(out_idx, out_w, count)
        return self.adj

