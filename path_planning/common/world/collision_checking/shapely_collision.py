"""
Shapely (GEOS) disc-sweep collision checker over a ``GraphSampler`` roadmap (2D).

``ShapelyCollision(config)`` checks sweeps against a roadmap handed to ``set_graph(vertices, edges)``:
the node positions (``graph_map.position``, roadmap index order) and the directed ``(src, tgt)`` pairs of
``graph_map.edges``, so the vertex indices it returns are the map's node indices; the map's
``set_constraint_sweep`` does that call.

Agents are discs, each with its own radius and speed:

- ``overlapping_interval_paths``: sweep whole agent paths, many agents at once, and report when each
  agent covers each roadmap vertex and directed edge.
- ``agent_conflicts``: when the discs of two agents overlap (first contact, or every window), for every
  pair of agents; no roadmap needed.

The spatial layer (which vertices / edges lie within ``r`` of a segment or a point) is one bulk Shapely
``STRtree`` box query per element type for the candidates, then the exact strict distance test (numba,
parallel) in ``disc_sweep``; the timing, assembly and conflicts are the
backend-neutral, N-D code in ``disc_sweep`` shared with ``CGALCollision`` (2D / 3D). Shapely/GEOS is
planar, so this backend is 2D only (``set_graph`` raises for 3D).
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import STRtree

from path_planning.common.world.collision_checking.disc_sweep import (   # noqa: F401  (EdgeDir re-exported)
    EdgeDir, fold_directed_edges, check_velocity, sweep_paths, disc_agent_conflicts, segment_candidates,
    point_candidates)


class ShapelyCollisionConfig:
    """Stored with the map's other configs. ``record_sweep`` and ``use_exact_collision_check`` belonged to
    the former per-move queries (caches and an edge refinement); they are kept for serialisation only and
    no longer change any result."""
    def __init__(self, record_sweep: bool = True, use_exact_collision_check: bool = True):
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check

    def to_dict(self):
        return {
            "record_sweep": self.record_sweep,
            "use_exact_collision_check": self.use_exact_collision_check,
        }

    @staticmethod
    def from_dict(data: dict):
        return ShapelyCollisionConfig(
            record_sweep=data.get("record_sweep", True),
            use_exact_collision_check=data.get("use_exact_collision_check", True),
        )


class ShapelyCollision:
    """Disc sweeps against a ``GraphSampler`` roadmap's vertices and edges using Shapely (GEOS). Vertex
    indices in the results are the map's node indices; edge keys are directed ``(src, tgt)`` node-index
    pairs, in the directions the roadmap allows (``EdgeDir``)."""

    def __init__(self, config: "ShapelyCollisionConfig | dict | None" = None):
        if isinstance(config, dict):
            config = ShapelyCollisionConfig.from_dict(config)
        self.config = config if config is not None else ShapelyCollisionConfig()
        self.reset()

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
        self.vertex_positions = None
        self.edge_indices = {}
        # One EdgeDir value per canonical edge, as bytes (``np.frombuffer(self.edge_dir, np.uint8)``).
        self.edge_dir = b""
        self.vertex_tree = None
        self.edge_tree = None
        self.edge_src_array = None
        self.edge_tgt_array = None
        self.edge_src_positions = None
        self.edge_tgt_positions = None

    def clear_caches(self):
        """Kept for the map's API: the path queries keep no per-query caches."""

    _check_velocity = staticmethod(check_velocity)

    def set_graph(
        self,
        vertices: np.ndarray | None = None,
        edges: np.ndarray | None = None,
        default_radius: float = 1.0,
    ):
        """Build the spatial indices from ``vertices`` (``(N, 2)`` node positions in roadmap index order,
        i.e. ``graph_map.position``) and ``edges`` (the directed ``(src, tgt)`` pairs of
        ``graph_map.edges``: each undirected roadmap edge in the directions it may be traversed, which
        this class folds back to one geometry per undirected edge with its ``EdgeDir``; see
        ``fold_directed_edges``). ``default_radius`` is accepted for API compatibility."""
        _ = default_radius
        if vertices is None or edges is None:
            raise ValueError("set_graph: pass vertices and edges")
        self.reset()
        vertices = np.asarray(vertices, dtype=float)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if vertices.ndim != 2 or vertices.shape[0] == 0:
            raise ValueError("set_graph: vertices must be a non-empty (N, 2) array; generate the nodes first")
        if edges.shape[0] == 0:
            raise ValueError("set_graph: edges cannot be empty; generate the roadmap first")
        dim = vertices.shape[1]
        if dim != 2:
            raise ValueError(
                f"ShapelyCollision supports 2D only (got dim={dim}); Shapely/GEOS is planar. "
                f"Use the CGAL backend for 3D."
            )
        if edges.min() < 0 or edges.max() >= vertices.shape[0]:
            raise IndexError(f"set_graph: edge index out of range [0, {vertices.shape[0]})")

        self.vertex_positions = vertices
        self.vertex_tree = STRtree(shapely.points(self.vertex_positions))
        # One geometry per undirected edge with the directions present (EdgeDir); no self-loops:
        # standing at vertex i is reported as a vertex hit.
        lo, hi, dirs = fold_directed_edges(edges)
        self.edge_src_array, self.edge_tgt_array = lo, hi
        self.edge_dir = dirs.astype(np.uint8).tobytes()
        self.edge_indices = dict(enumerate(zip(lo.tolist(), hi.tolist())))
        self.edge_src_positions = np.ascontiguousarray(self.vertex_positions[lo])
        self.edge_tgt_positions = np.ascontiguousarray(self.vertex_positions[hi])
        self.edge_tree = STRtree(shapely.linestrings(
            np.stack([self.edge_src_positions, self.edge_tgt_positions], axis=1)))

    # ------------------------------------------------------------ spatial layer

    @staticmethod
    def _within(tree, geoms, r):
        """``(row, element)`` pairs with the element strictly within ``r[row]`` of ``geoms[row]`` (GEOS
        ``dwithin`` is inclusive, so the hits are re-filtered with an exact strict distance test)."""
        if tree is None or len(geoms) == 0:
            z = np.empty(0, dtype=np.int64)
            return z, z
        row, el = tree.query(geoms, predicate="dwithin", distance=r)
        if len(row) == 0:
            return row.astype(np.int64), el.astype(np.int64)
        keep = shapely.distance(geoms[row], tree.geometries[el]) < r[row]
        return row[keep].astype(np.int64), el[keep].astype(np.int64)

    def _vertices_within(self, geom, r):
        """Indices of roadmap vertices strictly within distance ``r`` of one geometry."""
        return self._within(self.vertex_tree, np.array([geom], dtype=object), np.array([float(r)]))[1]

    def _edges_within(self, geom, r):
        """Indices (into the edge arrays) of roadmap edges strictly within distance ``r`` of one geometry."""
        return self._within(self.edge_tree, np.array([geom], dtype=object), np.array([float(r)]))[1]

    @staticmethod
    def _box_query(tree, a, b, r):
        """``(row, element)`` pairs whose envelope meets the box of segment ``a[row] -> b[row]`` grown by
        ``r[row]``: a superset of the elements within ``r[row]``."""
        if tree is None or len(a) == 0:
            z = np.empty(0, dtype=np.int64)
            return z, z
        lo = np.minimum(a, b) - r[:, None]
        hi = np.maximum(a, b) + r[:, None]
        row, el = tree.query(shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1]))
        return row.astype(np.int64), el.astype(np.int64)

    def _vertex_candidates(self, a, b, r):
        return self._box_query(self.vertex_tree, a, b, r)

    def _edge_candidates(self, a, b, r):
        return self._box_query(self.edge_tree, a, b, r)

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
        """When each disc agent, moving along its path, covers each roadmap element:
        ``[((vertex, t1, t2), (src, tgt, t1, t2)), ...]``, one entry per path: NumPy arrays, one row per
        window, sorted by element then time, absolute times from ``t0``, each element's windows merged.
        ``as_arrays=False`` gives dicts instead, ``[({vertex: [(t1, t2), ...]}, {(src, tgt): [(t1, t2), ...]}),
        ...]`` (slower for large batches: building the Python objects can cost more than the sweep).

        ``paths``: one path ``(P, 2)``, same-length paths ``(A, P, 2)`` or a list of paths. ``radius`` (the
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
        """When the discs of every two agents overlap: ``{(i, j): [(t1, t2), ...]}`` (``i < j``), or the
        first contact time with ``first_only=True``; agents ``i`` and ``j`` collide while their centres are
        closer than ``r_i + r_j``. Same inputs as ``overlapping_interval_paths`` (``radius`` is each agent's
        own radius); ``pairs`` limits the check to those agent pairs. See
        ``disc_sweep.disc_agent_conflicts``."""
        return disc_agent_conflicts(paths, radius, velocity, t0, turning_speed, goal_hold, pairs, first_only,
                                    as_arrays)
