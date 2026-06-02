"""
GridOverlay for Neural-ATTF on arbitrary graphs.

The neural encoder (U-Net) consumes a 2D image. To use it on non-grid graphs,
we project the graph onto an axis-aligned grid defined by ``graph_map.bounds``
at a chosen resolution. Each cell is marked free / obstacle via
``graph_map.in_collision_point``. World <-> cell conversion lets us:

  * build the encoder input from agent / task point tuples,
  * wrap the encoder's per-cell output as a callable A* guidance cost
    indexed by world point.

For lattice grid maps (``use_discrete_space=True``) this collapses to an
identity mapping (since node points are already grid indices in the
discrete-space convention used by ``Environment``).
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import math
import numpy as np


class GridOverlay:
    def __init__(self, graph_map, resolution: float | None = None):
        self.graph_map = graph_map
        bounds = np.asarray(graph_map.bounds, dtype=float)
        if bounds.shape[0] < 2:
            raise ValueError("GridOverlay requires 2D bounds")
        self.bounds_lo = bounds[:, 0].astype(float).copy()
        self.bounds_hi = bounds[:, 1].astype(float).copy()
        self.resolution = float(resolution or getattr(graph_map, "resolution", 1.0))

        # Width / height of the overlay in cells. Use ceil so the upper bound is
        # included. Pad to a multiple of 16 so a U-Net with 4 downsampling
        # stages can ingest the image without padding hacks.
        span = self.bounds_hi - self.bounds_lo
        w = max(1, int(math.ceil(float(span[0]) / self.resolution)))
        h = max(1, int(math.ceil(float(span[1]) / self.resolution)))
        target = max(w, h)
        enc = int(math.ceil(target / 16.0)) * 16
        self.shape: Tuple[int, int] = (enc, enc)

        # Build a binary maze: 1 = free, 0 = obstacle. Each overlay cell is
        # sampled at its world-frame center.
        maze = np.ones(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                world = self.cell_to_point((i, j))
                if not (
                    self.bounds_lo[0] <= world[0] <= self.bounds_hi[0]
                    and self.bounds_lo[1] <= world[1] <= self.bounds_hi[1]
                ):
                    maze[i, j] = 0.0
                    continue
                if graph_map.in_collision_point(world):
                    maze[i, j] = 0.0
        self.maze: np.ndarray = maze

    def point_to_cell(self, point: Sequence[float]) -> Tuple[int, int]:
        i = int(math.floor((float(point[0]) - self.bounds_lo[0]) / self.resolution))
        j = int(math.floor((float(point[1]) - self.bounds_lo[1]) / self.resolution))
        i = min(max(i, 0), self.shape[0] - 1)
        j = min(max(j, 0), self.shape[1] - 1)
        return i, j

    def cell_to_point(self, cell: Sequence[int]) -> Tuple[float, float]:
        x = self.bounds_lo[0] + (float(cell[0]) + 0.5) * self.resolution
        y = self.bounds_lo[1] + (float(cell[1]) + 0.5) * self.resolution
        return float(x), float(y)

    def make_cost_map_lookup(self, cost_array: np.ndarray) -> Callable[[Sequence[float]], float]:
        """Wrap a 2D ``cost_array`` shaped like ``self.shape`` as a point lookup."""
        arr = np.asarray(cost_array)
        if arr.shape != self.shape:
            raise ValueError(f"cost_array shape {arr.shape} != overlay shape {self.shape}")

        def lookup(point: Sequence[float]) -> float:
            i, j = self.point_to_cell(point)
            return float(arr[i, j])

        return lookup
