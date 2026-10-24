from typing import List, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array

from path_planning.common.environment.node import Node
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


class Lattice(RoadmapBase):
    """4-connected (2D) / 6-connected (3D) lattice over cell-centre nodes."""

    name = "grid"

    def __init__(self, graph_map: "GraphSamplerBase"):
        super().__init__(graph_map)

    def generate_lattice(self) -> csr_array:
        """Cardinal-neighbour roadmap for nodes on the Cartesian cell grid.

        Every node must sit at a cell centre and no two nodes may share a cell.
        Each node is linked to the nodes in the adjacent cells one step along
        each axis in both directions, so at most ``2 * dim`` neighbours (4 in
        2D). All edges have the uniform length ``resolution``; the min/max
        edge-length band does not apply. The segment between two adjacent
        free-cell centres stays inside those two cells, so no line-of-sight
        check is needed.
        """
        n = self.num_nodes
        if n == 0:
            self.adj = csr_array((0, 0), dtype=float)
            return self.adj

        m = self.graph_map
        shape = tuple(int(s) for s in m.shape)
        cells = np.rint(np.asarray(m.path_world_to_map(self.nodes), dtype=float)).astype(np.int64)
        cells = cells.reshape(n, self.dim)
        centres = np.asarray(m.path_map_to_world(cells), dtype=float).reshape(n, self.dim)
        off_grid = ~np.all(np.isclose(centres, self.nodes, rtol=0.0, atol=1e-6 * m.resolution), axis=1)
        if off_grid.any():
            raise ValueError(f"generate_grid: {int(off_grid.sum())} node(s) are not at cell centres")

        flat = np.ravel_multi_index(tuple(cells.T), shape)
        if len(np.unique(flat)) != n:
            raise ValueError("generate_grid: several nodes share a cell")
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
        self.adj = csr_array((w, (rows, cols)), shape=(n, n))
        self.adj.sort_indices()
        return self.adj

    generate = generate_lattice
