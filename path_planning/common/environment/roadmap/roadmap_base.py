"""Array-backed roadmaps: an ``(N, dim)`` point array plus a CSR adjacency.

``RoadmapBase`` holds the shared state and helpers; one subclass per roadmap type
implements ``generate_roadmap`` (see ``prm.py``, ``lattice.py``,
``delaunay.py`` and the planar family under ``roadmap_planar.py``).
"""
import math
from dataclasses import dataclass, fields, asdict
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from python_motion_planning.common.env.map.grid import TYPES, _grid_path_world_to_map_int, _grid_path_map_to_world

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


@dataclass
class RoadmapConfig:
    """Tunable parameters shared by every roadmap builder, built from a dict.

    ``RoadmapConfig({"num_neighbors": 6, "max_edge_length": 12})``; keys left
    out take the defaults below, unknown keys raise. Edge band and k apply to
    the builders that filter edges (PRM, Delaunay, planar family, RRG,
    custom); ``Lattice`` edges are fixed by the grid.
    """
    min_edge_length: float = 0.0          # drop edges shorter than this
    max_edge_length: float = math.inf     # drop edges longer than this; RRG steer / connect radius
    num_neighbors: int = 0                # PRM: accepted edges per node, 0 = unlimited
    sample_num: int = 0                   # RRG: nodes to add
    goal_sample_rate: float = 0.1         # RRG: probability of drawing a goal
    max_iters_ratio: float = 50.0         # RRG: iteration cap = ratio * sample_num
    filter_collision: bool = True         # line-of-sight test on candidate edges
    snap_off_grid: bool = True            # Lattice: snap off-centre nodes instead of raising
    k_query: int | None = None            # PRM: first KD query size (default 4*k+1)
    grow: int = 4                         # PRM: query growth factor per round

    def __init__(self, config: dict | None = None) -> None:
        config = dict(config or {})
        for f in fields(self):                      # every field, default when absent
            setattr(self, f.name, config.pop(f.name, f.default))
        if config:
            raise TypeError(f"RoadmapConfig: unknown keys {sorted(config)}")
        if self.min_edge_length < 0 or self.max_edge_length < self.min_edge_length:
            raise ValueError(f"need 0 <= min_edge_length <= max_edge_length, got {self.min_edge_length}, {self.max_edge_length}")
        if self.num_neighbors < 0 or self.sample_num < 0:
            raise ValueError("num_neighbors and sample_num must be >= 0")
        if not 0.0 <= self.goal_sample_rate <= 1.0:
            raise ValueError(f"goal_sample_rate must be in [0, 1], got {self.goal_sample_rate}")
        if self.max_iters_ratio <= 0 or self.grow < 2:
            raise ValueError("max_iters_ratio must be > 0 and grow >= 2")
        self.min_edge_length = float(self.min_edge_length)
        self.max_edge_length = float(self.max_edge_length)
        self.num_neighbors = int(self.num_neighbors)
        self.sample_num = int(self.sample_num)

    def as_dict(self) -> dict:
        return asdict(self)


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

    def __init__(self, graph_map: "GraphSamplerBase", config: dict | RoadmapConfig | None = None):
        """``config`` is a dict of ``RoadmapConfig`` fields (missing keys take
        the defaults, ``None`` means all defaults); an already-built
        ``RoadmapConfig`` is used as is."""
        self.graph_map = graph_map
        self.config = config if isinstance(config, RoadmapConfig) else RoadmapConfig(config)

        self._dim = self.graph_map.dim
        self._num_nodes = 0

        self.nodes = np.zeros((0, self.dim))
        self.clearance = np.zeros(0)
        self.adj = csr_array((0, 0), dtype=float)
        self.snapped: dict = {}             # original position -> node position, for builders that move nodes

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
