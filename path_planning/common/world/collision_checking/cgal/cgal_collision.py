"""Disc / sphere sweep collision checker over a ``GraphSampler`` roadmap (2D or 3D).

Agents are balls, each with its own radius and speed. The queries (``overlapping_interval_paths``,
``agent_conflicts``) are those of ``collision_base.CollisionBase``; this class supplies the candidate search
(``spatial_index``, same results either way):

- ``"grid"`` (default): uniform grids (``spatial_grid.BoxGrid``), built and queried in parallel in numba.
  Fastest for roadmaps of roughly even density.
- ``"rtree"``: a SciPy KD-tree over the vertices and an R-tree over the edge boxes. Adapts to uneven
  density (clusters, large empty regions, long edges).

The name is historical: no CGAL bindings are needed. Polygon / polyhedron agents: ``CGALCollisionHetero``.
"""

from __future__ import annotations

import itertools

import numpy as np

from path_planning.common.world.collision_checking.sweep.collision_base import CollisionBase
from path_planning.common.world.collision_checking.sweep.disc_sweep import EdgeDir   # noqa: F401  (re-exported)
from path_planning.common.world.collision_checking.sweep.spatial_grid import BoxGrid


SPATIAL_INDEXES = ("grid", "rtree")


class CGALCollision(CollisionBase):
    """Disc / sphere sweeps against a roadmap (2D or 3D). ``spatial_index``: ``"grid"`` or ``"rtree"``."""

    def __init__(self, spatial_index: str = "grid"):
        if spatial_index not in SPATIAL_INDEXES:
            raise ValueError(f"spatial_index must be one of {SPATIAL_INDEXES}, got {spatial_index!r}")
        self.spatial_index = spatial_index
        super().__init__()

    def reset(self):
        super().reset()
        self.vertex_grid = None          # "grid"
        self.edge_grid = None
        self.vertex_kdtree = None        # "rtree"
        self.edge_rtree = None

    def _build_index(self):
        bmin = np.minimum(self.edge_src_positions, self.edge_tgt_positions)
        bmax = np.maximum(self.edge_src_positions, self.edge_tgt_positions)
        if self.spatial_index == "grid":
            self.vertex_grid = BoxGrid(self.vertex_positions)
            self.edge_grid = BoxGrid(bmin, bmax)
        else:
            from rtree import index
            from scipy.spatial import KDTree
            self.vertex_kdtree = KDTree(self.vertex_positions)
            prop = index.Property()
            prop.dimension = self.vertex_positions.shape[1]
            self.edge_rtree = index.Index(
                ((i, (*bmin[i], *bmax[i]), None) for i in range(len(bmin))), properties=prop)

    @staticmethod
    def _grown_boxes(a, b, r):
        """The box of each segment ``a[k] -> b[k]`` grown by ``r[k]``."""
        r = np.asarray(r, dtype=float)[:, None]
        return np.minimum(a, b) - r, np.maximum(a, b) + r

    def _vertex_candidates(self, a, b, r):
        """Grid: the vertices in each segment's grown box. KD-tree: those in the ball enclosing the capsule."""
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
        """The edges whose box meets each segment's grown box."""
        if len(a) == 0:
            z = np.empty(0, dtype=np.int64)
            return z, z
        lo, hi = self._grown_boxes(a, b, r)
        if self.edge_grid is not None:
            return self.edge_grid.query(lo, hi)
        ids, counts = self.edge_rtree.intersection_v(np.ascontiguousarray(lo), np.ascontiguousarray(hi))
        rows = np.repeat(np.arange(len(a), dtype=np.int64), counts.astype(np.int64))
        return rows, ids.astype(np.int64)
