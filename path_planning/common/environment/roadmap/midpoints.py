from typing import TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_array

from path_planning.common.environment.roadmap.roadmap_planar import RoadmapPlanar

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


class Midpoints(RoadmapPlanar):
    """Nodes are the midpoints of the non-constraint triangle edges, linked
    when their edges share a triangle. Each anchor becomes a node linked to
    the midpoints of its incident edges."""

    name = "midpoints"

    def __init__(self, graph_map: "GraphSamplerBase"):
        super().__init__(graph_map)

    def generate_midpoints(self, anchor_points=None, filter_collision: bool = True) -> csr_array:
        cdt, anchor_vidx = self._triangulate(self.nodes, anchor_points)
        tri = np.asarray(cdt["triangles"], dtype=np.int64)
        verts = np.asarray(cdt["vertices"], dtype=float)
        eu, ev, slot_edge = self._triangle_edges(tri)

        # Constraint (obstacle outline) edges get no midpoint node.
        V = len(verts)
        segs = np.asarray(cdt["segments"], dtype=np.int64)
        constraint_keys = np.minimum(segs[:, 0], segs[:, 1]) * V + np.maximum(segs[:, 0], segs[:, 1])
        kept = ~np.isin(eu * V + ev, constraint_keys)
        mid_id = np.full(len(eu), -1, dtype=np.int64)
        mid_id[kept] = np.arange(int(kept.sum()))
        M = int(kept.sum())
        points = 0.5 * (verts[eu[kept]] + verts[ev[kept]])

        us, vs = [], []
        for a, b in ((0, 1), (1, 2), (2, 0)):        # edge pairs within each triangle
            ma, mb = mid_id[slot_edge[:, a]], mid_id[slot_edge[:, b]]
            ok = (ma >= 0) & (mb >= 0)
            us.append(ma[ok])
            vs.append(mb[ok])
        if len(anchor_vidx):
            hit = ((eu[None, :] == anchor_vidx[:, None]) | (ev[None, :] == anchor_vidx[:, None])) & kept[None, :]
            k, e = np.nonzero(hit)
            us.append(M + k)
            vs.append(mid_id[e])
            points = np.vstack([points, np.asarray(anchor_points, dtype=float).reshape(-1, 2)])
        return self._finish_planar(points, np.concatenate(us), np.concatenate(vs), filter_collision)

    generate = generate_midpoints
