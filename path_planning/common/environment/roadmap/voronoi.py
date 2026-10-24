from typing import TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array

from path_planning.common.environment.roadmap.roadmap_planar import RoadmapPlanar

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


def _circumcentres(tp: np.ndarray) -> np.ndarray:
    """Circumcentre of each triangle in ``tp`` (T, 3, 2) via the intersection
    of two perpendicular bisectors (Cramer's rule)."""
    p1, p2, p3 = tp[:, 0], tp[:, 1], tp[:, 2]
    a1, a2 = p2 - p1, p3 - p2                        # bisector normals
    c1 = (a1 * (p1 + p2) * 0.5).sum(axis=1)
    c2 = (a2 * (p2 + p3) * 0.5).sum(axis=1)
    det = a1[:, 0] * a2[:, 1] - a2[:, 0] * a1[:, 1]
    bad = np.abs(det) < 1e-10
    if bad.any():
        raise ValueError(f"Voronoi: {int(bad.sum())} degenerate (collinear) triangle(s)")
    return np.stack([(c1 * a2[:, 1] - c2 * a1[:, 1]) / det,
                     (a1[:, 0] * c2 - a2[:, 0] * c1) / det], axis=1)


class Voronoi(RoadmapPlanar):
    """Nodes are the triangle circumcentres (the Voronoi vertices of the
    triangulation), linked when the triangles share an edge; anchors are
    linked to their incident triangles. Co-circular triangles share a
    circumcentre and collapse to one node."""

    name = "voronoi"

    def __init__(self, graph_map: "GraphSamplerBase"):
        super().__init__(graph_map)

    def generate_voronoi(self, anchor_points=None, filter_collision: bool = True) -> csr_array:
        return self._dual_graph(_circumcentres, anchor_points, filter_collision)

    generate = generate_voronoi
