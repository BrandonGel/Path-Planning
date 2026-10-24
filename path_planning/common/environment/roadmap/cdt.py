from typing import TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array

from path_planning.common.environment.roadmap.roadmap_base import RoadmapConfig
from path_planning.common.environment.roadmap.roadmap_planar import RoadmapPlanar

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


class CDT(RoadmapPlanar):
    """Nodes are the triangulation vertices (obstacle corners, anchors and
    the given samples), edges are the triangle edges."""

    name = "cdt"

    def __init__(self, graph_map: "GraphSamplerBase", config: dict | RoadmapConfig | None = None):
        super().__init__(graph_map, config)

    def generate_cdt(self, anchor_points=None, filter_collision: bool | None = None) -> csr_array:
        cdt, _ = self._triangulate(self.nodes, anchor_points)
        tri = np.asarray(cdt["triangles"], dtype=np.int64)
        eu, ev, _ = self._triangle_edges(tri)
        return self._finish_planar(cdt["vertices"], eu, ev, filter_collision)

    generate = generate_cdt
