"""Disc sweep collision checker over a ``GraphSampler`` roadmap (2D, Shapely / GEOS).

Agents are discs, each with its own radius and speed. The queries (``overlapping_interval_paths``,
``agent_conflicts``) are those of ``collision_base.CollisionBase``; this class supplies the candidate search,
one bulk ``STRtree`` box query per element type. For 3D use ``CGALCollision``.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import STRtree

from path_planning.common.world.collision_checking.sweep.collision_base import CollisionBase
from path_planning.common.world.collision_checking.sweep.disc_sweep import EdgeDir   # noqa: F401  (re-exported)


class ShapelyCollision(CollisionBase):
    """Disc sweeps against a roadmap (2D), with Shapely ``STRtree`` candidate queries."""

    dims = (2,)

    def reset(self):
        super().reset()
        self.vertex_tree = None
        self.edge_tree = None

    def _build_index(self):
        self.vertex_tree = STRtree(shapely.points(self.vertex_positions))
        self.edge_tree = STRtree(shapely.linestrings(
            np.stack([self.edge_src_positions, self.edge_tgt_positions], axis=1)))

    @staticmethod
    def _box_query(tree, a, b, r):
        """``(row, element)`` pairs whose envelope meets the box of segment ``a[row] -> b[row]`` grown by
        ``r[row]``."""
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
