"""
Disc / sphere sweep collision checker over a ``GraphSampler`` roadmap, in 2D or 3D.

``CGALCollision(config)`` checks sweeps against a roadmap handed to ``set_graph(vertices, edges)``: the
node positions (``graph_map.position``, roadmap index order, ``(N, 2)`` or ``(N, 3)``) and the directed
``(src, tgt)`` pairs of ``graph_map.edges``, so the vertex indices it returns are the map's node indices;
the map's ``set_constraint_sweep`` does that call.

Agents are discs (2D) or spheres (3D), each with its own radius and speed:

- ``overlapping_interval_paths``: sweep whole agent paths, many agents at once, and report when each
  agent covers each roadmap vertex and directed edge.
- ``agent_conflicts``: when the balls of two agents overlap (first contact, or every window), for every
  pair of agents; no roadmap needed.

The same API as ``ShapelyCollision`` (2D, GEOS). The spatial layer finds candidate vertices and edges near
each segment, then the exact strict distance test (numba, parallel) shared in ``disc_sweep`` keeps the true
ones; the timing, assembly and conflicts are the shared N-D code in ``disc_sweep``. The candidate index is
``CGALCollisionConfig.spatial_index``:

- ``"grid"`` (default): a uniform grid over the vertices and one over the edge boxes
  (``spatial_grid.BoxGrid``), built and queried in parallel in numba. Fastest for roadmaps of roughly even
  density.
- ``"rtree"``: a SciPy KD-tree over the vertices and an R-tree (``rtree`` / libspatialindex) over the edge
  boxes, one bulk query each. Adapts to uneven density (clusters, large empty regions, long edges).

Both give the same results. The name is historical: no CGAL bindings are needed. Polygon / polyhedron agents:
``CGALCollisionHetero``.
"""

from __future__ import annotations

import itertools

import numpy as np

from path_planning.common.world.collision_checking.disc_sweep import (   # noqa: F401  (EdgeDir re-exported)
    EdgeDir, fold_directed_edges, check_velocity, sweep_paths, disc_agent_conflicts, segment_candidates,
    point_candidates)
from path_planning.common.world.collision_checking.spatial_grid import BoxGrid


SPATIAL_INDEXES = ("grid", "rtree")


class CGALCollisionConfig:
    """Stored with the map's other configs. ``spatial_index``: ``"grid"`` (default) or ``"rtree"``, the
    candidate index built by ``set_graph`` (see the module docstring; same results, different speed).
    ``record_sweep`` and ``use_exact_collision_check`` belonged to the former per-move queries (caches and
    an edge refinement); they are kept for serialisation only and no longer change any result."""
    def __init__(self, record_sweep: bool = True, use_exact_collision_check: bool = True,
                 spatial_index: str = "grid"):
        if spatial_index not in SPATIAL_INDEXES:
            raise ValueError(f"spatial_index must be one of {SPATIAL_INDEXES}, got {spatial_index!r}")
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check
        self.spatial_index = spatial_index

    def to_dict(self):
        return {
            "record_sweep": self.record_sweep,
            "use_exact_collision_check": self.use_exact_collision_check,
            "spatial_index": self.spatial_index,
        }

    @staticmethod
    def from_dict(data: dict):
        return CGALCollisionConfig(
            record_sweep=data.get("record_sweep", True),
            use_exact_collision_check=data.get("use_exact_collision_check", True),
            spatial_index=data.get("spatial_index", "grid"),
        )


class CGALCollision:
    """Disc / sphere sweeps against a ``GraphSampler`` roadmap's vertices and edges (2D or 3D). Vertex
    indices in the results are the map's node indices; edge keys are directed ``(src, tgt)`` node-index
    pairs, in the directions the roadmap allows (``EdgeDir``)."""

    def __init__(self, config: "CGALCollisionConfig | dict | None" = None, spatial_index: "str | None" = None):
        """``config``: a ``CGALCollisionConfig`` or its dict. ``spatial_index`` (``"grid"`` / ``"rtree"``)
        overrides the config's, e.g. ``CGALCollision(spatial_index="rtree")``."""
        if isinstance(config, dict):
            config = CGALCollisionConfig.from_dict(config)
        self.config = config if config is not None else CGALCollisionConfig()
        if spatial_index is not None:
            if spatial_index not in SPATIAL_INDEXES:
                raise ValueError(f"spatial_index must be one of {SPATIAL_INDEXES}, got {spatial_index!r}")
            self.config.spatial_index = spatial_index
        self.reset()

    @property
    def spatial_index(self) -> str:
        return self.config.spatial_index

    @property
    def record_sweep(self) -> bool:
        return self.config.record_sweep

    @property
    def use_exact_collision_check(self) -> bool:
        return self.config.use_exact_collision_check

    @property
    def num_vertices(self) -> int:
        return 0 if self.vertex_positions is None else int(self.vertex_positions.shape[0])

    def reset(self):
        self.dim = None
        self.vertex_positions = None
        self.vertex_grid = None          # "grid"
        self.edge_grid = None
        self.vertex_kdtree = None        # "rtree"
        self.edge_rtree = None
        self.edge_indices = {}
        # One EdgeDir value per canonical edge, as bytes (``np.frombuffer(self.edge_dir, np.uint8)``).
        self.edge_dir = b""
        self.edge_src_array = None
        self.edge_tgt_array = None
        self.edge_src_positions = None
        self.edge_tgt_positions = None

    def clear_caches(self):
        """Kept for the map's API: the path queries keep no per-query caches."""

    _check_velocity = staticmethod(check_velocity)

    def set_graph(self, vertices=None, edges=None, default_radius: float = 1.0):
        """Build the spatial indices from ``vertices`` (``(N, D)`` node positions, D = 2 or 3, in roadmap
        index order, i.e. ``graph_map.position``) and ``edges`` (the directed ``(src, tgt)`` pairs of
        ``graph_map.edges``, folded back to one entry per undirected edge with its ``EdgeDir``; see
        ``fold_directed_edges``). No self-loops: standing at a vertex is reported as a vertex hit.
        ``default_radius`` is accepted for API compatibility."""
        _ = default_radius
        if vertices is None or edges is None:
            raise ValueError("set_graph: pass vertices and edges")
        self.reset()
        vertices = np.asarray(vertices, dtype=float)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if vertices.ndim != 2 or len(vertices) == 0:
            raise ValueError("set_graph: vertices must be a non-empty (N, D) array; generate the nodes first")
        if len(edges) == 0:
            raise ValueError("set_graph: edges cannot be empty; generate the roadmap first")
        dim = vertices.shape[1]
        if dim not in (2, 3):
            raise ValueError(f"CGALCollision supports 2D and 3D (got dim={dim})")
        if edges.min() < 0 or edges.max() >= len(vertices):
            raise IndexError(f"set_graph: edge index out of range [0, {len(vertices)})")
        self.dim = dim
        self.vertex_positions = vertices
        lo, hi, dirs = fold_directed_edges(edges)
        self.edge_src_array, self.edge_tgt_array = lo, hi
        self.edge_dir = dirs.astype(np.uint8).tobytes()
        self.edge_indices = dict(enumerate(zip(lo.tolist(), hi.tolist())))
        self.edge_src_positions = np.ascontiguousarray(vertices[lo])
        self.edge_tgt_positions = np.ascontiguousarray(vertices[hi])
        bmin = np.minimum(self.edge_src_positions, self.edge_tgt_positions)
        bmax = np.maximum(self.edge_src_positions, self.edge_tgt_positions)
        if self.spatial_index == "grid":
            self.vertex_grid = BoxGrid(vertices)
            self.edge_grid = BoxGrid(bmin, bmax)
        else:
            from rtree import index
            from scipy.spatial import KDTree
            self.vertex_kdtree = KDTree(vertices)
            prop = index.Property()
            prop.dimension = dim
            self.edge_rtree = index.Index(
                ((i, (*bmin[i], *bmax[i]), None) for i in range(len(bmin))), properties=prop)

    # ------------------------------------------------------------ spatial layer

    @staticmethod
    def _grown_boxes(a, b, r):
        """The box of each segment ``a[k] -> b[k]`` grown by ``r[k]``."""
        r = np.asarray(r, dtype=float)[:, None]
        return np.minimum(a, b) - r, np.maximum(a, b) + r

    def _vertex_candidates(self, a, b, r):
        """``(row, vertex)`` pairs, a superset of the vertices within ``r[row]`` of segment
        ``a[row] -> b[row]``. Grid: the vertices in the segment's box grown by ``r[row]`` (one parallel
        query). R-tree mode: the vertices in the ball enclosing the capsule (one bulk, multi-threaded
        KD-tree query)."""
        if len(a) == 0:
            z = np.empty(0, dtype=np.int64)
            return z, z
        if self.vertex_grid is not None:
            return self.vertex_grid.query(*self._grown_boxes(a, b, r))
        center = 0.5 * (a + b)
        R = 0.5 * np.linalg.norm(b - a, axis=1) + np.asarray(r, dtype=float)
        hits = self.vertex_kdtree.query_ball_point(center, R, workers=-1, return_sorted=False)
        counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        rows = np.repeat(np.arange(len(a), dtype=np.int64), counts)
        ids = np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64, count=int(counts.sum()))
        return rows, ids

    def _edge_candidates(self, a, b, r):
        """``(row, edge)`` pairs whose edge box meets the box of segment ``a[row] -> b[row]`` grown by
        ``r[row]`` (one parallel grid query, or one bulk R-tree query): a superset of the edges within
        ``r[row]``."""
        if len(a) == 0:
            z = np.empty(0, dtype=np.int64)
            return z, z
        lo, hi = self._grown_boxes(a, b, r)
        if self.edge_grid is not None:
            return self.edge_grid.query(lo, hi)
        ids, counts = self.edge_rtree.intersection_v(np.ascontiguousarray(lo), np.ascontiguousarray(hi))
        rows = np.repeat(np.arange(len(a), dtype=np.int64), counts.astype(np.int64))
        return rows, ids.astype(np.int64)

    def _segment_candidates(self, a, b, r):
        """Vertices and edges strictly within ``r[k]`` of each segment ``a[k] -> b[k]``:
        ``((seg, vertex), (seg, edge))``."""
        return segment_candidates(self, a, b, r)

    def _point_candidates(self, p, r):
        """Vertices strictly within ``r[k] - 1e-10`` and edges strictly within ``r[k]`` of each point
        ``p[k]``: ``((point, vertex), (point, edge))``."""
        return point_candidates(self, p, r)

    # ------------------------------------------------------------ queries

    def overlapping_interval_paths(self, paths, radius, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                                   as_arrays: bool = True) -> list:
        """When each ball agent, moving along its path, covers each roadmap element:
        ``[((vertex, t1, t2), (src, tgt, t1, t2)), ...]``, one entry per path: NumPy arrays, one row per
        window, sorted by element then time, absolute times from ``t0``, each element's windows merged.
        ``as_arrays=False`` gives dicts instead, ``[({vertex: [(t1, t2), ...]}, {(src, tgt): [(t1, t2), ...]}),
        ...]``.

        ``paths``: one path ``(P, D)``, same-length paths ``(A, P, D)`` or a list of paths. ``radius`` (the
        sweep radius: e.g. ``r_i + r_j`` to keep agent ``j`` clear of agent ``i``), ``velocity``, ``t0``,
        ``turning_speed`` and ``goal_hold`` are one value or one per path. ``velocity=None``: every segment
        takes 1 time unit; ``turning_speed`` (rad/s): the agent waits ``angle / turning_speed`` at each
        waypoint (None, ``inf`` or 0: no wait); ``goal_hold``: it stays at its last waypoint that long
        (``inf``: forever). A vertex or edge is covered while it is strictly within the radius of the
        agent's centre; contacts of zero duration are left out."""
        return sweep_paths(self, paths, radius, velocity, t0, turning_speed, goal_hold, as_arrays)

    @staticmethod
    def agent_conflicts(paths, radius, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0, pairs=None,
                        first_only: bool = False, as_arrays: bool = False):
        """When the balls of every two agents overlap: ``{(i, j): [(t1, t2), ...]}`` (``i < j``), or the
        first contact time with ``first_only=True``; agents ``i`` and ``j`` collide while their centres are
        closer than ``r_i + r_j``. Same inputs as ``overlapping_interval_paths`` (``radius`` is each agent's
        own radius); ``pairs`` limits the check to those agent pairs. See
        ``disc_sweep.disc_agent_conflicts``."""
        return disc_agent_conflicts(paths, radius, velocity, t0, turning_speed, goal_hold, pairs, first_only,
                                    as_arrays)
