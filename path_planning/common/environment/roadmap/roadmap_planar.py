from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import triangle as tr
from scipy.ndimage import label
from scipy.sparse import csr_array
from python_motion_planning.common.env.map.grid import TYPES

from path_planning.common.environment.node import Node
from path_planning.common.environment.roadmap.roadmap_base import RoadmapBase, _unique_undirected

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase



def _dedupe_points(points: np.ndarray, nd: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Drop coordinate duplicates (rounded to ``nd`` decimals), keeping the
    first occurrence and its order. Returns ``(points, old_to_new)``."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, points.shape[-1] if points.ndim == 2 else 2), np.zeros(0, np.int64)
    _, first, inverse = np.unique(np.round(points, nd), axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")          # unique ids by first occurrence
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return np.ascontiguousarray(points[first[order]]), rank[inverse.ravel()]



class RoadmapPlanar(RoadmapBase):
    """Shared machinery of the triangulation-based roadmaps (2D only).

    The free space is triangulated with a constrained Delaunay triangulation
    whose constraint outline sits on the true obstacle cell corners. Subclasses
    derive nodes and edges from the triangles and finish with
    ``_finish_planar``, which **replaces** ``nodes`` / ``clearance``.
    ``anchor_points`` (e.g. start and goal) are forced to become nodes.
    """

    def __init__(self, graph_map: "GraphSamplerBase"):
        super().__init__(graph_map)

    # -------------------------------------------------------------- outline
    def obstacle_boundary(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Outline of the blocked region as a planar straight-line graph.

        Returns ``(points (B, 2), segments (S, 2), holes (H, 2))`` in world
        coordinates: the corners of the obstacle/inflation cells where the
        outline turns or branches, the straight segments between them (every
        cell face touching free space or the map edge, plus the closed map
        rectangle, with collinear runs merged), and one seed point inside each
        connected blocked component. Vertices lie on the cell corner lattice
        ``bounds_lo + k * resolution``, i.e. exactly where the collision
        checker and ESDF see the obstacles.
        """
        if self.dim != 2:
            raise NotImplementedError("obstacle_boundary needs a 2D map")
        m = self.graph_map
        tm = m.type_map.data
        mask = (tm == TYPES.OBSTACLE) | (tm == TYPES.INFLATION)
        H, W = mask.shape
        lo = np.asarray(m.bounds, dtype=float)[:, 0]
        res = float(m.resolution)
        Wc = W + 1                                   # corners per lattice row

        # A face lies between two cells whose blocked state differs (outside
        # counts as free); every face on the map rectangle is always present.
        hface = np.diff(np.pad(mask, ((1, 1), (0, 0))), axis=0) != 0   # (H+1, W): between rows r-1, r
        hface[0, :] = True
        hface[H, :] = True
        vface = np.diff(np.pad(mask, ((0, 0), (1, 1))), axis=1) != 0   # (H, W+1): between cols c-1, c
        vface[:, 0] = True
        vface[:, W] = True
        hr, hc = np.nonzero(hface)
        vr, vc = np.nonzero(vface)
        h_a, h_b = hr * Wc + hc, hr * Wc + hc + 1          # horizontal face (r, c): corners (r,c)-(r,c+1)
        v_a, v_b = vr * Wc + vc, (vr + 1) * Wc + vc        # vertical face (r, c): corners (r,c)-(r+1,c)
        deg = np.bincount(np.concatenate([h_a, h_b, v_a, v_b]), minlength=(H + 1) * Wc)

        def merge_runs(a, b, major, minor):
            """Faces sorted along their line; consecutive faces fuse when the
            shared corner has degree 2 (a pass-through vertex)."""
            order = np.lexsort((minor, major))
            a, b, major, minor = a[order], b[order], major[order], minor[order]
            join = (major[1:] == major[:-1]) & (minor[1:] == minor[:-1] + 1) & (deg[b[:-1]] == 2)
            starts = np.nonzero(np.concatenate(([True], ~join)))[0]
            ends = np.concatenate((starts[1:] - 1, [len(a) - 1]))
            return a[starts], b[ends]

        hs, he = merge_runs(h_a, h_b, hr, hc)
        vs, ve = merge_runs(v_a, v_b, vc, vr)
        seg_a = np.concatenate([hs, vs])
        seg_b = np.concatenate([he, ve])
        ids, inverse = np.unique(np.concatenate([seg_a, seg_b]), return_inverse=True)
        segments = inverse.ravel().reshape(2, -1).T
        r, c = np.divmod(ids, Wc)
        points = lo + np.stack([r, c], axis=1) * res

        labeled, _ = label(mask)
        labs, first = np.unique(labeled.ravel(), return_index=True)
        first = first[labs > 0]
        hr0, hc0 = np.divmod(first, W)
        holes = lo + (np.stack([hr0, hc0], axis=1) + 0.5) * res
        return points, segments, holes

    # -------------------------------------------------------- triangulation
    def _triangulate(self, interior: np.ndarray, anchor_points) -> Tuple[dict, np.ndarray]:
        """Constrained Delaunay triangulation of the free space containing the
        boundary vertices, the anchors and the interior points. Returns the
        ``triangle`` result and the vertex id of every anchor."""
        bnd_pts, bnd_segs, holes = self.obstacle_boundary()
        anchors = (np.zeros((0, 2)) if anchor_points is None
                   else np.asarray(anchor_points, dtype=float).reshape(-1, 2))
        interior = np.asarray(interior, dtype=float).reshape(-1, 2)
        all_pts, old_to_new = _dedupe_points(np.vstack([bnd_pts, anchors, interior]))
        segs = old_to_new[bnd_segs]
        segs = segs[segs[:, 0] != segs[:, 1]]
        su, sv, _ = _unique_undirected(segs[:, 0], segs[:, 1])
        pslg = dict(vertices=all_pts, segments=np.stack([su, sv], axis=1))
        if len(holes):
            pslg["holes"] = holes
        cdt = tr.triangulate(pslg, "pc")
        verts = np.asarray(cdt["vertices"], dtype=float)
        if len(verts) < len(all_pts) or not np.allclose(verts[: len(all_pts)], all_pts):
            raise RuntimeError("triangle did not preserve the input vertex order")
        anchor_vidx = old_to_new[len(bnd_pts): len(bnd_pts) + len(anchors)]
        return cdt, anchor_vidx

    @staticmethod
    def _triangle_edges(tri: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unique undirected triangle edges and the edge id of each of the
        three slots ``(0,1), (1,2), (2,0)`` of every triangle, shape (T, 3)."""
        eu, ev, inv = _unique_undirected(tri[:, [0, 1, 2]].ravel(), tri[:, [1, 2, 0]].ravel())
        return eu, ev, inv.reshape(len(tri), 3)

    def _triangle_adjacency(self, tri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pairs of triangles that share an edge."""
        eu, _, slot_edge = self._triangle_edges(tri)
        inv = slot_edge.ravel()
        order = np.argsort(inv, kind="stable")
        tri_of_slot = np.repeat(np.arange(len(tri)), 3)[order]
        counts = np.bincount(inv, minlength=len(eu))
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        two = counts == 2
        return tri_of_slot[starts[two]], tri_of_slot[starts[two] + 1]

    def _dual_graph(self, centres_of, anchor_points, filter_collision: bool) -> csr_array:
        """Shared body of the centroid and Voronoi graphs: one node per
        triangle, edges between triangles sharing an edge, anchors linked to
        the triangles that contain them."""
        cdt, anchor_vidx = self._triangulate(self.nodes, anchor_points)
        tri = np.asarray(cdt["triangles"], dtype=np.int64)
        verts = np.asarray(cdt["vertices"], dtype=float)
        T = len(tri)
        points = centres_of(verts[tri])
        ta, tb = self._triangle_adjacency(tri)
        us, vs = [ta], [tb]
        if len(anchor_vidx):
            hit = (tri[None, :, :] == anchor_vidx[:, None, None]).any(axis=2)   # (K, T)
            k, t = np.nonzero(hit)
            us.append(T + k)
            vs.append(t)
            points = np.vstack([points, np.asarray(anchor_points, dtype=float).reshape(-1, 2)])
        return self._finish_planar(points, np.concatenate(us), np.concatenate(vs), filter_collision)

    # --------------------------------------------------------------- finish
    def _finish_planar(self, points: np.ndarray, u: np.ndarray, v: np.ndarray,
                       filter_collision: bool) -> csr_array:
        """Dedupe the points, remap the edges onto them, install them as the
        nodes and keep the edges passing the length band and (optionally) the
        line-of-sight filter."""
        pts, old_to_new = _dedupe_points(points)
        u = old_to_new[np.asarray(u, dtype=np.int64)]
        v = old_to_new[np.asarray(v, dtype=np.int64)]
        self._set_points(pts)
        return self.filter_edges(u, v, filter_collision)

    # -------------------------------------------------------------- builder
    def generate(self, anchor_points=None, filter_collision: bool = True) -> csr_array:
        """Build the graph from the current nodes; implemented per subclass."""
        raise NotImplementedError

    def generate_roadmap(self, samples: List[Node], anchor_points=None,
                         filter_collision: bool = True) -> csr_array:
        """``samples`` become interior triangulation points; ``anchor_points``
        (e.g. start and goal) are forced to be nodes of the result."""
        self.set_nodes(samples)
        return self.generate(anchor_points, filter_collision)
