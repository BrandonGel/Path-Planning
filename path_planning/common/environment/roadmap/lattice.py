import warnings
from typing import List, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array

from path_planning.common.environment.node import Node
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase, RoadmapConfig

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


class Lattice(RoadmapBase):
    """4-connected (2D) / 6-connected (3D) lattice over cell-centre nodes."""

    name = "grid"

    def __init__(self, graph_map: "GraphSamplerBase", config: dict | RoadmapConfig | None = None):
        super().__init__(graph_map, config)

    def generate_lattice(self, snap_off_grid: bool | None = None) -> csr_array:
        """Cardinal-neighbour roadmap for nodes on the Cartesian cell grid.

        Each node is linked to the nodes in the adjacent cells one step along
        each axis in both directions, so at most ``2 * dim`` neighbours (4 in
        2D). All edges have the uniform length ``resolution``; the min/max
        edge-length band does not apply. The segment between two adjacent
        free-cell centres stays inside those two cells, so no line-of-sight
        check is needed.

        Nodes off the cell centres (typically start/goal anchors) are snapped
        to the centre of their cell when ``snap_off_grid`` (default from the
        config) is True; a node that
        lands in a cell already holding a node is merged into it. Every move is
        reported in a warning and recorded in ``self.snapped``
        (original position -> position of the node it became). With
        ``snap_off_grid`` False an off-grid node raises ``ValueError``.
        """
        snap_off_grid = self.config.snap_off_grid if snap_off_grid is None else bool(snap_off_grid)
        self._snapped = {}
        n = self.num_nodes
        if n == 0:
            self._adj = csr_array((0, 0), dtype=float)
            return self.adj

        m = self.graph_map
        shape = tuple(int(s) for s in m.shape)
        snapped = self.centres_of(self.cells_of(self.nodes))      # every node's cell centre
        off_grid = ~np.all(np.isclose(snapped, self.nodes, rtol=0.0, atol=1e-6 * m.resolution), axis=1)
        if off_grid.any():
            if not snap_off_grid:
                raise ValueError(f"generate_lattice: {int(off_grid.sum())} node(s) are not at cell centres")
            self._snap_off_grid(off_grid, snapped)
            n = self.num_nodes
        cells = self.cells_of(self.nodes)

        flat = np.ravel_multi_index(tuple(cells.T), shape)
        if len(np.unique(flat)) != n:
            raise ValueError("generate_lattice: several nodes share a cell")
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
        self._adj = csr_array((w, (rows, cols)), shape=(n, n))
        self.adj.sort_indices()
        return self.adj

    def _snap_off_grid(self, off_grid: np.ndarray, snapped: np.ndarray) -> None:
        """Replace each off-centre node by its cell centre ``snapped[i]``; a node
        whose cell already holds a node is merged into that node. Fills
        ``self.snapped`` (original -> final position) and warns with the list."""
        key = [tuple(np.round(p, 9)) for p in snapped]            # cell centre as dict key
        owner = {k: i for i, k in enumerate(key) if not off_grid[i]}  # cells already occupied
        keep = np.ones(self.num_nodes, dtype=bool)
        report = []
        for i in np.nonzero(off_grid)[0]:
            src = tuple(float(x) for x in self.nodes[i])
            j = owner.setdefault(key[i], int(i))                   # first node in the cell wins
            if j != i:
                keep[i] = False
            dst = tuple(float(x) for x in snapped[j])
            self._snapped[src] = dst
            report.append(f"  {np.round(src, 6).tolist()} -> {np.round(dst, 6).tolist()}"
                          + (" (merged into existing node)" if j != i else " (snapped)"))
        self._nodes = np.ascontiguousarray(snapped[keep])
        self._clearance = self.clearance[keep]
        self._num_nodes = len(self.nodes)
        warnings.warn(f"Lattice: {len(report)} node(s) were not at cell centres and were snapped:\n"
                      + "\n".join(report), stacklevel=3)

    generate = generate_lattice
