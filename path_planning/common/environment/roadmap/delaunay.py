from itertools import combinations
from typing import List, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import Delaunay

from path_planning.common.environment.node import Node
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase, RoadmapConfig

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


class DelaunayRoadmap(RoadmapBase):
    """Delaunay edges between the given samples, any dimension, no obstacle
    constraints; edges filtered by the length band and line of sight."""

    name = "dt"

    def __init__(self, graph_map: "GraphSamplerBase", config: dict | RoadmapConfig | None = None):
        super().__init__(graph_map, config)

    def generate_dt(self, filter_collision: bool | None = None) -> csr_array:
        n = self.num_nodes
        if n <= self.dim:
            self._adj = csr_array((n, n), dtype=float)
            return self.adj
        simplices = np.asarray(Delaunay(self.nodes).simplices, dtype=np.int64)
        pairs = list(combinations(range(self.dim + 1), 2))   # all simplex edges (6 per tetrahedron)
        u = np.concatenate([simplices[:, a] for a, _ in pairs])
        v = np.concatenate([simplices[:, b] for _, b in pairs])
        return self.filter_edges(u, v, filter_collision)

    generate = generate_dt
