"""``CollisionSweep``: the base class of the roadmap sweep backends (``ShapelyCollision``, ``CGALCollision``).

It binds the roadmap (``set_graph``) and provides the two queries, ``overlapping_interval_paths`` and
``agent_conflicts``, for ball agents (discs in 2D, spheres in 3D). A backend only builds a spatial index and
supplies candidate queries; the exact distance test, timing and assembly are in ``disc_sweep``.
"""
from __future__ import annotations

import numpy as np

from path_planning.common.world.collision_checking.sweep.disc_sweep import (
    fold_directed_edges, sweep_paths, disc_agent_conflicts)


class CollisionSweep:
    """Ball sweeps against a roadmap. Vertex ids in the results are the map's node indices; edge keys are
    directed ``(src, tgt)`` node-index pairs, in the directions the roadmap allows (``EdgeDir``).

    A subclass sets ``dims`` and implements ``_build_index`` and the candidate queries
    ``_vertex_candidates(a, b, r)`` / ``_edge_candidates(a, b, r)``, which return ``(row, element)`` index
    arrays: a superset of the elements within ``r[row]`` of segment ``a[row] -> b[row]``."""

    dims = (2, 3)                        # supported roadmap dimensions

    def __init__(self):
        self.reset()

    def reset(self):
        self.vertex_positions = None
        self.edge_dir = b""              # one EdgeDir byte per canonical edge
        self.edge_src_array = None       # canonical (undirected) edge endpoints, lo < hi
        self.edge_tgt_array = None
        self.edge_src_positions = None
        self.edge_tgt_positions = None

    def set_graph(self, vertices, edges):
        """Bind the roadmap and build the spatial index. ``vertices``: ``(N, D)`` node positions in roadmap
        index order; ``edges``: the directed ``(src, tgt)`` pairs of ``graph_map.edges``, folded to one
        entry per undirected edge with its ``EdgeDir``. No self-loops: standing at a vertex is a vertex hit."""
        self.reset()
        vertices = np.asarray(vertices, dtype=float)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if vertices.ndim != 2 or len(vertices) == 0:
            raise ValueError("set_graph: vertices must be a non-empty (N, D) array; generate the nodes first")
        if len(edges) == 0:
            raise ValueError("set_graph: edges cannot be empty; generate the roadmap first")
        if vertices.shape[1] not in self.dims:
            raise ValueError(f"{type(self).__name__} supports {' and '.join(f'{d}D' for d in self.dims)} only "
                             f"(got dim={vertices.shape[1]})")
        if edges.min() < 0 or edges.max() >= len(vertices):
            raise IndexError(f"set_graph: edge index out of range [0, {len(vertices)})")
        self.vertex_positions = vertices
        lo, hi, dirs = fold_directed_edges(edges)
        self.edge_src_array, self.edge_tgt_array = lo, hi
        self.edge_dir = dirs.astype(np.uint8).tobytes()
        self.edge_src_positions = np.ascontiguousarray(vertices[lo])
        self.edge_tgt_positions = np.ascontiguousarray(vertices[hi])
        self._build_index()

    def _build_index(self):
        raise NotImplementedError

    def _vertex_candidates(self, a, b, r):
        raise NotImplementedError

    def _edge_candidates(self, a, b, r):
        raise NotImplementedError

    def overlapping_interval_paths(self, paths, radius, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0,
                                   as_arrays: bool = True) -> list:
        """When each ball agent covers each roadmap element along its path. One entry per path:
        ``((vertex, t1, t2), (src, tgt, t1, t2))`` arrays, one row per window, sorted by element then time,
        absolute times, windows merged. ``as_arrays=False``: ``({vertex: [(t1, t2), ...]},
        {(src, tgt): [(t1, t2), ...]})`` dicts instead (slower for large batches).

        ``paths``: one ``(P, D)`` path, ``(A, P, D)`` paths or a list of paths. The other arguments are one
        value or one per path:

        - ``radius``: the sweep radius (e.g. ``r_i + r_j`` to keep agent ``j`` clear of agent ``i``).
        - ``velocity``: speed; ``None`` means every segment takes 1 time unit.
        - ``t0``: start time.
        - ``turning_speed`` (rad/s): wait ``angle / turning_speed`` at each waypoint (``None``, ``inf``
          or 0: no wait).
        - ``goal_hold``: time spent at the last waypoint (``inf``: forever).

        An element is covered while strictly within the radius; zero-duration contacts are left out."""
        return sweep_paths(self, paths, radius, velocity, t0, turning_speed, goal_hold, as_arrays)

    @staticmethod
    def agent_conflicts(paths, radius, velocity=None, t0=0.0, turning_speed=None, goal_hold=0.0, pairs=None,
                        first_only: bool = False, as_arrays: bool = False):
        """When two agents' balls overlap (centres closer than ``r_i + r_j``):
        ``{(i, j): [(t1, t2), ...]}`` with ``i < j``. ``first_only``: ``{(i, j): first contact time}``.
        ``as_arrays``: ``(i, j, t1, t2)`` arrays. ``pairs``: only check these agent pairs. Inputs as in
        ``overlapping_interval_paths``, with ``radius`` each agent's own radius. No roadmap is needed."""
        return disc_agent_conflicts(paths, radius, velocity, t0, turning_speed, goal_hold, pairs, first_only,
                                    as_arrays)
