from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array

from path_planning.common.environment.node import Node
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


class CustomRoadmap(RoadmapBase):
    """Roadmap from an explicit edge list over the given samples, any
    dimension. Duplicate, reversed and self-loop entries collapse to one
    undirected edge; the length band and (optionally) line of sight filter
    the rest. Edge weight is the Euclidean length."""

    name = "custom"

    def __init__(self, graph_map: "GraphSamplerBase"):
        super().__init__(graph_map)

    def generate_custom(self, edge_indices: List[Tuple[int, int]] | np.ndarray,
                        filter_collision: bool = True) -> csr_array:
        edges = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 2)
        if edges.size and (edges.min() < 0 or edges.max() >= self.num_nodes):
            raise IndexError(f"edge index out of range for {self.num_nodes} nodes")
        return self.filter_edges(edges[:, 0], edges[:, 1], filter_collision)

    generate = generate_custom

    def generate_roadmap(self, samples: List[Node], edge_indices: List[Tuple[int, int]] | np.ndarray,
                         filter_collision: bool = True) -> csr_array:
        """``edge_indices`` index into ``samples``."""
        self.set_nodes(samples)
        return self.generate_custom(edge_indices, filter_collision)
