from typing import TYPE_CHECKING
from scipy.sparse import csr_array

from path_planning.common.environment.roadmap.roadmap_base import RoadmapConfig
from path_planning.common.environment.roadmap.roadmap_planar import RoadmapPlanar

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


class Centroids(RoadmapPlanar):
    """Nodes are the triangle centroids, linked when the triangles share an
    edge; anchors are linked to their incident triangles."""

    name = "centroids"

    def __init__(self, graph_map: "GraphSamplerBase", config: dict | RoadmapConfig | None = None):
        super().__init__(graph_map, config)

    def generate_centroids(self, anchor_points=None, filter_collision: bool | None = None) -> csr_array:
        return self._dual_graph(lambda tp: tp.mean(axis=1), anchor_points, filter_collision)

    generate = generate_centroids
