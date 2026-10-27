"""Array-backed roadmaps: an ``(N, dim)`` point array plus a CSR adjacency.

``RoadmapBase`` holds the shared state and helpers; one subclass per roadmap type
implements ``generate_roadmap`` (see ``prm.py``, ``lattice.py``,
``delaunay.py`` and the planar family under ``roadmap_planar.py``).
"""
import math
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path as _csgraph_shortest_path
from path_planning.common.environment.roadmap.roadmap_config import RoadmapConfig
from python_motion_planning.common.env.map.grid import TYPES, _grid_path_world_to_map_int, _grid_path_map_to_world
from path_planning.common.environment.roadmap.roadmap_util import _filter_edges_kernel, _khop_ball, _unique_undirected, _symmetric_csr, shortest_paths
from path_planning.common.environment.node import Node


if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


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
    uses_anchors: bool = False          # builders that take anchor_points (start/goal) in generate()

    def __init__(self, graph_map: "GraphSamplerBase", config: "dict | RoadmapConfig | str | None" = None):
        """``config`` is a dict of ``RoadmapConfig`` fields (missing keys take
        the defaults, ``None`` means all defaults), a YAML path, or an
        already-built ``RoadmapConfig`` used as is."""
        self.graph_map = graph_map
        self.config = RoadmapConfig.load(config)

        self._dim = self.graph_map.dim
        self.reset()

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @property
    def clearance(self) -> np.ndarray:
        return self._clearance

    @property
    def adj(self) -> csr_array:
        return self._adj

    @property
    def snapped(self) -> dict:
        return self._snapped

    # ---------------------------------------------------------------- config
    @property
    def min_edge_length(self) -> float:
        return self.config.min_edge_length

    @property
    def max_edge_length(self) -> float:
        return self.config.max_edge_length

    @property
    def num_neighbors(self) -> int:
        return self.config.num_neighbors

    def _filter_collision(self, value) -> bool:
        """Resolve a per-call ``filter_collision`` override against the config."""
        return self.config.filter_collision if value is None else bool(value)

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
    def roadmap_edges(self) -> List[List[int]]:
        return self._rows(self.adj.indices)

    @property
    def roadmap_weights(self) -> List[List[float]]:
        return self._rows(self.adj.data)

    def neighbors(self, u: int) -> np.ndarray:
        return self.adj.indices[self.adj.indptr[u]:self.adj.indptr[u + 1]]

    @property
    def edges(self) -> np.ndarray:
        if self.num_nodes == 0 or self.adj.nnz == 0:
            return np.zeros((0, 2), dtype=np.int64)
        u = np.repeat(np.arange(self.num_nodes), np.diff(self.adj.indptr))
        edges = np.stack([u, self.adj.indices], axis=1)
        return edges

    @property
    def edge_weights(self) -> np.ndarray:
        return self.adj.data
        
    # ------------------------------------------------------------- shortest paths
    def dijkstra(self, source: int, target: int | None = None, *, limit: float = np.inf
                 ) -> float | List[Tuple[int, float]]:
        """Roadmap distances from node ``source``.

        With ``target``: the shortest-path length to that node as a float,
        ``inf`` when unreachable. Without: ``[(node, distance), ...]`` for every
        node reachable from ``source`` (itself included at 0), in node order.
        ``limit`` caps the search radius.
        """
        dist = shortest_paths(int(source), self.adj, limit=limit)
        if target is not None:
            return float(dist[int(target)])
        reach = np.flatnonzero(np.isfinite(dist))
        return list(zip(reach.tolist(), dist[reach].tolist()))

    def hops(self, sources) -> np.ndarray:
        """Hop count (unweighted shortest path) from each of ``sources`` to every
        node: shape ``(len(sources), num_nodes)``, 1-D for a scalar source,
        ``inf`` where unreachable, 0 at the source itself."""
        scalar = np.isscalar(sources)
        idx = np.atleast_1d(np.asarray(sources, dtype=np.int64))
        if self.num_nodes == 0:
            raise ValueError("hops: the roadmap has no nodes")
        if idx.size and (idx.min() < 0 or idx.max() >= self.num_nodes):
            raise IndexError(f"hops: source out of range [0, {self.num_nodes})")
        out = _csgraph_shortest_path(self.adj, directed=True, unweighted=True, indices=idx)
        return out[0] if scalar else out

    def khop_distances(self, sources, khop: int = -1) -> dict:
        """``{u: [(v, distance), ...]}`` from each node index in ``sources``:
        ``distance`` is the shortest-path length over the roadmap edges
        (Dijkstra), ``v`` are node indices in node order.

        ``khop=-1`` (or ``None``): every node, ``inf`` where ``v`` is not
        reachable from ``u``. ``khop=k >= 0``: only the nodes within ``k`` hops
        of ``u`` (``u`` itself at hop 0 with distance 0), each with its full
        shortest-path length, which may use more than ``k`` edges.
        """
        idx = [int(u) for u in np.atleast_1d(np.asarray(sources, dtype=np.int64))]
        if not idx:
            return {}
        n = self.num_nodes
        if n == 0:
            raise ValueError("khop_distances: the roadmap has no nodes")
        if min(idx) < 0 or max(idx) >= n:
            raise IndexError(f"khop_distances: source out of range [0, {n})")
        if khop is None or khop < 0:
            dist = shortest_paths(idx, self.adj)                   # (len(idx), n), inf = unreachable
            return {u: list(zip(range(n), dist[i].tolist())) for i, u in enumerate(idx)}
        # k-hop ball: BFS limited to k levels (touches only the ball), then a
        # Dijkstra capped at the longest BFS-tree path in the ball. Every ball
        # node's shortest path is no longer than its BFS-tree path, so the
        # capped search is exact there and never expands far beyond the ball.
        adj = self.adj
        out = {}
        for u in idx:
            ball, _, bound = _khop_ball(adj.indptr, adj.indices, adj.data, u, int(khop), n)
            limit = float(bound[ball].max()) if len(ball) > 1 else 0.0
            dist = shortest_paths(u, adj, limit=limit) if limit > 0 else np.where(np.arange(n) == u, 0.0, np.inf)
            within = np.sort(ball)
            out[u] = list(zip(within.tolist(), dist[within].tolist()))
        return out

    def shortest_path(self, source: int, target: int) -> Tuple[List[int], float]:
        """Node sequence from ``source`` to ``target`` and its length;
        ``([], inf)`` when unreachable."""
        dist, pred = shortest_paths(int(source), self.adj, return_predecessors=True)
        if not np.isfinite(dist[target]):
            return [], math.inf
        path, v = [int(target)], int(target)
        while v != source:
            v = int(pred[v])
            path.append(v)
        return path[::-1], float(dist[target])

    # ----------------------------------------------------------------- nodes
    def set_nodes(self, samples: List[Node]) -> None:
        n = len(samples)
        self._num_nodes = n
        if len(samples) > 0:
            self._nodes = np.asarray([s.current for s in samples], dtype=float).reshape(n, self.dim)
            self._clearance = np.asarray([s.max_clearance for s in samples], dtype=float)
            self._adj = csr_array((n, n), dtype=float)
        else:
            self._nodes = np.zeros((0, self.dim))
            self._clearance = np.zeros(0)
            self._adj = csr_array((0, 0), dtype=float)

    def set_adjacency(self, roadmap_edges: List[List[int]], roadmap_weights: List[List[float]] | None = None) -> None:
        """Install the adjacency from per-node neighbour lists (the
        ``roadmap_edges`` / ``roadmap_weights`` layout, as pickles store it).
        Missing weights are Euclidean lengths."""
        n = self.num_nodes
        if len(roadmap_edges) != n:
            raise ValueError(f"roadmap_edges has {len(roadmap_edges)} rows for {n} nodes")
        counts = np.fromiter((len(r) for r in roadmap_edges), dtype=np.int64, count=n)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        indices = np.fromiter((v for r in roadmap_edges for v in r), dtype=np.int64, count=int(indptr[-1]))
        if roadmap_weights is not None and len(roadmap_weights) == n and all(len(w) == c for w, c in zip(roadmap_weights, counts)):
            data = np.fromiter((d for r in roadmap_weights for d in r), dtype=np.float64, count=int(indptr[-1]))
        else:
            u = np.repeat(np.arange(n), counts)
            data = np.linalg.norm(self.nodes[u] - self.nodes[indices], axis=1) if len(u) else np.zeros(0)
        self._adj = csr_array((data, indices, indptr), shape=(n, n))

    def append_isolated_nodes(self, points) -> None:
        """Append nodes with no edges (e.g. a start or goal the builder did
        not produce); the adjacency grows by empty rows."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        if len(pts) == 0:
            return
        n, k = self.num_nodes, len(pts)
        self._nodes = np.vstack([self.nodes.reshape(n, self.dim), pts])
        self._clearance = np.concatenate([self.clearance, self._clearance_at(pts)])
        self._num_nodes = n + k
        indptr = np.concatenate([self.adj.indptr, np.full(k, self.adj.indptr[-1])])
        self._adj = csr_array((self.adj.data, self.adj.indices, indptr), shape=(n + k, n + k))

    def _set_points(self, points: np.ndarray) -> None:
        """Replace the nodes with raw points; clearance is read from the map."""
        pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, self.dim)
        self._nodes = pts
        self._num_nodes = len(pts)
        self._clearance = self._clearance_at(pts)

    # ---------------------------------------------------------------- frames
    def cells_of(self, points: np.ndarray) -> np.ndarray:
        """Cell index (N, dim) int64 of each world point: nearest cell centre,
        clipped to the grid (``_grid_path_world_to_map_int``)."""
        m = self.graph_map
        pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, self.dim)
        return _grid_path_world_to_map_int(pts, np.asarray(m.bounds, dtype=np.float64), float(m.resolution),
                                           np.asarray(m.shape, dtype=np.int64))

    def centres_of(self, cells: np.ndarray) -> np.ndarray:
        """World centre (N, dim) float64 of each cell (``_grid_path_map_to_world``)."""
        m = self.graph_map
        c = np.ascontiguousarray(cells, dtype=np.float64).reshape(-1, self.dim)
        return _grid_path_map_to_world(c, np.asarray(m.bounds, dtype=np.float64), float(m.resolution))

    def _clearance_at(self, points: np.ndarray) -> np.ndarray:
        """Max object radius at the cell nearest each point (clipped to the
        grid); 0 where nothing fits."""
        if len(points) == 0:
            return np.zeros(0)
        cells = self.cells_of(points)
        return np.asarray(self.graph_map.inflated_esdf[tuple(cells.T)], dtype=float)

    # ------------------------------------------------------------- collision
    def in_collision(self, p1, p2=None) -> bool:
        return self.graph_map.in_collision(p1, p2)

    def _collision_args(self) -> tuple:
        """Map arrays the numba kernels need, bundled once per build."""
        m = self.graph_map
        return (np.asarray(m.bounds, dtype=np.float64), float(m.resolution),
                np.asarray(m.shape, dtype=np.int64), m._type_map_flat(), m._esdf_flat(),
                int(TYPES.OBSTACLE), int(TYPES.INFLATION), bool(m.strict_collision))

    def filter_edges(self, u: np.ndarray, v: np.ndarray, filter_collision: bool | None = None) -> csr_array:
        """Install the undirected edges ``(u, v)`` over the current nodes that
        pass the config's length band and (unless ``filter_collision`` is
        False, default from the config) the line-of-sight test."""
        filter_collision = self._filter_collision(filter_collision)
        u = np.asarray(u, dtype=np.int64).ravel()
        v = np.asarray(v, dtype=np.int64).ravel()
        loops = u != v
        eu, ev, _ = _unique_undirected(u[loops], v[loops])
        pts = np.ascontiguousarray(self.nodes, dtype=np.float64)
        keep, w = _filter_edges_kernel(pts, eu, ev, float(self.min_edge_length), float(self.max_edge_length),
                                       bool(filter_collision), *self._collision_args())
        self._adj = _symmetric_csr(self.num_nodes, eu[keep], ev[keep], w[keep])
        return self.adj

    # --------------------------------------------------------------- builder
    def generate(self, **kwargs) -> csr_array:
        """Build the adjacency from the current nodes; each subclass binds this
        to its own builder (``generate_prm``, ``generate_lattice``, ...)."""
        raise NotImplementedError(f"{type(self).__name__} must implement generate")

    def generate_roadmap(self, samples: List[Node], **kwargs) -> csr_array:
        """Install ``samples`` as the nodes, then ``generate(**kwargs)``."""
        self.set_nodes(samples)
        if self.num_nodes == 0:
            raise ValueError("No nodes to generate roadmap")
        return self.generate(**kwargs)

    def reset(self) -> None:
        self._num_nodes = 0
        self._nodes: np.ndarray = np.zeros((0, self.dim))
        self._clearance: np.ndarray = np.zeros(0)
        self._adj: csr_array = csr_array((0, 0), dtype=float)
        self._snapped: dict = {}             # original position -> node position, for builders that move nodes


    