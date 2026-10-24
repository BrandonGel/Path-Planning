"""Array-backed roadmaps: an ``(N, dim)`` point array plus a CSR adjacency.

``RoadmapBase`` holds the shared state and helpers; one subclass per roadmap type
implements ``generate_roadmap`` (see ``prm.py``, ``lattice.py``,
``delaunay.py`` and the planar family under ``roadmap_planar.py``).
"""
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from python_motion_planning.common.env.map.grid import TYPES

from path_planning.common.environment.node import Node
from path_planning.common.environment.map.numbaUtil import _world_in_collision

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


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


class RoadmapBase:
    """Base roadmap: nodes, clearance and a CSR adjacency over a ``GraphSamplerBase``.

    ``adj.indices[adj.indptr[u]:adj.indptr[u+1]]`` are the neighbours of ``u``
    and ``adj.data`` the matching edge lengths. ``road_map`` / ``edge_weights``
    give the list-of-lists view the rest of the codebase reads.

    Subclasses implement ``generate_roadmap(samples, ...)``. ``PRM``,
    ``Lattice`` and ``DelaunayRoadmap`` connect the given samples; the planar
    family (``CDT``, ``Midpoints``, ``Centroids``, ``Voronoi``) replaces the
    nodes with points derived from a constrained triangulation.
    """

    name: str = "base"

    def __init__(self, graph_map: "GraphSamplerBase"):
        self.graph_map = graph_map

        self._dim = self.graph_map.dim
        self._num_nodes = 0

        self.min_edge_length = 0.0
        self.max_edge_length = float('inf')
        self.num_neighbors = float('inf')

        self.nodes = np.zeros((0, self.dim))
        self.clearance = np.zeros(0)
        self.adj = csr_array((0, 0), dtype=float)

    # ------------------------------------------------------------- identity
    def __hash__(self) -> int:
        return  hash((self.num_nodes, self.adj.nnz,
                     np.ascontiguousarray(self.nodes, dtype=np.float64).tobytes(),
                     np.ascontiguousarray(self.adj.indptr).tobytes(),
                     np.ascontiguousarray(self.adj.indices).tobytes(),
                     np.ascontiguousarray(self.adj.data, dtype=np.float64).tobytes()))

    def __eq__(self, other) -> bool:
        if not isinstance(other, RoadmapBase):
            return NotImplemented
        return (self.num_nodes == other.num_nodes and self.adj.nnz == other.adj.nnz
                and np.array_equal(self.nodes, other.nodes)
                and np.array_equal(self.adj.indptr, other.adj.indptr)
                and np.array_equal(self.adj.indices, other.adj.indices)
                and np.array_equal(self.adj.data, other.adj.data))

    # ----------------------------------------------------------------- views
    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return int(self.adj.nnz)

    @property
    def dim(self) -> int:
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

    # ----------------------------------------------------------------- nodes
    def set_nodes(self, samples: List[Node]) -> None:
        n = len(samples)
        self._num_nodes = n
        self.nodes = np.asarray([s.current for s in samples], dtype=float).reshape(n, self.dim)
        self.clearance = np.asarray([s.max_clearance for s in samples], dtype=float)
        self.adj = csr_array((n, n), dtype=float)

    def _set_points(self, points: np.ndarray) -> None:
        """Replace the nodes with raw points; clearance is read from the map."""
        pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, self.dim)
        self.nodes = pts
        self._num_nodes = len(pts)
        self.clearance = self._clearance_at(pts)

    def _clearance_at(self, points: np.ndarray) -> np.ndarray:
        """Max object radius at each point's cell; 0 outside the map."""
        m = self.graph_map
        n = len(points)
        if n == 0:
            return np.zeros(0)
        cells = np.rint(np.asarray(m.path_world_to_map(points), dtype=float)).astype(np.int64).reshape(n, self.dim)
        shape = np.asarray(m.shape, dtype=np.int64)
        inside = np.all((cells >= 0) & (cells < shape), axis=1)
        out = np.zeros(n)
        if inside.any():
            out[inside] = m.inflated_esdf[tuple(cells[inside].T)]
        return out

    # ------------------------------------------------------------- collision
    def in_collision(self, p1, p2=None) -> bool:
        return self.graph_map.in_collision(p1, p2)

    def _collision_args(self) -> tuple:
        """Map arrays the numba kernels need, bundled once per build."""
        m = self.graph_map
        return (np.asarray(m.bounds, dtype=np.float64), float(m.resolution),
                np.asarray(m.shape, dtype=np.int64), m._type_map_flat(), m._esdf_flat(),
                int(TYPES.OBSTACLE), int(TYPES.INFLATION), bool(m.strict_collision))

    def filter_edges(self, u: np.ndarray, v: np.ndarray, filter_collision: bool = True) -> csr_array:
        """Install the undirected edges ``(u, v)`` over the current nodes that
        pass the length band and (optionally) the line-of-sight test."""
        u = np.asarray(u, dtype=np.int64).ravel()
        v = np.asarray(v, dtype=np.int64).ravel()
        loops = u != v
        eu, ev, _ = _unique_undirected(u[loops], v[loops])
        pts = np.ascontiguousarray(self.nodes, dtype=np.float64)
        keep, w = _filter_edges_kernel(pts, eu, ev, float(self.min_edge_length), float(self.max_edge_length),
                                       bool(filter_collision), *self._collision_args())
        self.adj = _symmetric_csr(self.num_nodes, eu[keep], ev[keep], w[keep])
        return self.adj

    # --------------------------------------------------------------- builder
    def generate(self, **kwargs) -> csr_array:
        """Build the adjacency from the current nodes; each subclass binds this
        to its own builder (``generate_prm``, ``generate_lattice``, ...)."""
        raise NotImplementedError(f"{type(self).__name__} must implement generate")

    def generate_roadmap(self, samples: List[Node], **kwargs) -> csr_array:
        """Install ``samples`` as the nodes, then ``generate(**kwargs)``."""
        self.set_nodes(samples)
        return self.generate(**kwargs)
