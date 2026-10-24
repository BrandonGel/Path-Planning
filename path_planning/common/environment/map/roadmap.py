from itertools import combinations
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import triangle as tr
from numba import njit, prange
from scipy.ndimage import label
from scipy.sparse import csr_array
from scipy.spatial import KDTree, Delaunay
from python_motion_planning.common.env.map.grid import TYPES

from path_planning.common.environment.node import Node
from path_planning.common.environment.map.numbaUtil import _world_in_collision

if TYPE_CHECKING:
    from path_planning.common.environment.map.graph_sampler_base import GraphSamplerBase


@njit(parallel=True, cache=True)
def _prm_edges_kernel(q, pts, dist, idx, min_len, max_len, k_cap,
                      bounds, res, shape, type_map, esdf, obs_t, inf_t, strict):
    """Per query row: walk the distance-sorted candidates, keep collision-free
    ones inside [min_len, max_len], stop after k_cap accepted. Rows are
    independent. ``q[ii]`` is the row's own point, ``pts[idx[ii, jj]]`` a
    candidate; ``dist`` may hold ``inf`` past a ``distance_upper_bound``.

    ``complete[ii]`` is True when no candidate outside the row could still be
    chosen: the walk ran past ``max_len``, or the quota was filled at a
    distance strictly below the row's last distance (so no tie is cut off).
    """
    n, m = dist.shape
    K = k_cap if k_cap > 0 else m
    out_idx = np.full((n, K), -1, np.int64)
    out_w = np.zeros((n, K))
    count = np.zeros(n, np.int64)
    complete = np.zeros(n, np.bool_)
    for ii in prange(n):
        c = 0
        d_stop = -1.0
        hit_max = False
        for jj in range(1, m):              # column 0 is the node itself
            d = dist[ii, jj]
            if d > max_len:
                hit_max = True              # sorted; inf (beyond bound) lands here too
                break
            if d < min_len:
                continue
            j = idx[ii, jj]
            if _world_in_collision(q[ii], pts[j], bounds, res, shape,
                                   type_map, esdf, obs_t, inf_t, strict):
                continue
            out_idx[ii, c] = j
            out_w[ii, c] = d
            c += 1
            if c >= K:
                d_stop = d
                break
        count[ii] = c
        complete[ii] = hit_max or (c >= K and d_stop < dist[ii, m - 1])
    return out_idx, out_w, count, complete


def _to_csr(out_idx: np.ndarray, out_w: np.ndarray, count: np.ndarray) -> csr_array:
    """Pack fixed-width kernel output (first ``count[i]`` slots of row i) into CSR."""
    n = len(count)
    keep = np.arange(out_idx.shape[1])[None, :] < count[:, None]
    indptr = np.concatenate(([0], np.cumsum(count)))
    return csr_array((out_w[keep], out_idx[keep], indptr), shape=(n, n))


def _canonical_rows(dist: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort each row by (distance, node index) so ties are deterministic
    and independent of the KD-tree and of the query size."""
    order = np.lexsort((idx, dist), axis=-1)
    return (np.ascontiguousarray(np.take_along_axis(dist, order, 1)),
            np.ascontiguousarray(np.take_along_axis(idx, order, 1)))


@njit(parallel=True, cache=True)
def _filter_edges_kernel(pts, u, v, min_len, max_len, check_collision,
                         bounds, res, shape, type_map, esdf, obs_t, inf_t, strict):
    """Keep edge ``e`` when ``min_len <= |pts[u[e]] - pts[v[e]]| <= max_len``
    and (if ``check_collision``) the segment is free in both directions, so
    an endpoint outside the map or inside an obstacle kills the edge
    whichever end it is. Returns the keep mask and every edge's length.
    Edges are independent."""
    E = len(u)
    keep = np.zeros(E, np.bool_)
    w = np.zeros(E)
    for e in prange(E):
        d2 = 0.0
        for k in range(pts.shape[1]):
            diff = pts[u[e], k] - pts[v[e], k]
            d2 += diff * diff
        d = np.sqrt(d2)
        w[e] = d
        if d < min_len or d > max_len:
            continue
        if check_collision and (
            _world_in_collision(pts[u[e]], pts[v[e]], bounds, res, shape, type_map, esdf, obs_t, inf_t, strict)
            or _world_in_collision(pts[v[e]], pts[u[e]], bounds, res, shape, type_map, esdf, obs_t, inf_t, strict)
        ):
            continue
        keep[e] = True
    return keep, w


def _unique_undirected(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected edges ``(eu < ev)`` of the pair list and, for every
    input slot, the id of its edge. Callers drop self-loops beforehand."""
    u = np.asarray(u, dtype=np.int64).ravel()
    v = np.asarray(v, dtype=np.int64).ravel()
    pairs = np.stack([np.minimum(u, v), np.maximum(u, v)], axis=1)
    if len(pairs) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return uniq[:, 0], uniq[:, 1], inverse.ravel()


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


def _symmetric_csr(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> csr_array:
    """CSR adjacency holding every undirected edge in both directions."""
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.concatenate([w, w])
    adj = csr_array((data, (rows, cols)), shape=(n, n))
    adj.sort_indices()
    return adj


# Numpy based roadmap class
class Roadmap:
    """Roadmap as an (N, dim) point array plus a CSR adjacency.

    ``adj.indices[adj.indptr[u]:adj.indptr[u+1]]`` are the neighbours of ``u``
    and ``adj.data`` the matching edge lengths. ``road_map`` / ``edge_weights``
    give the list-of-lists view the rest of the codebase reads.

    Builders: ``generate_prm`` / ``generate_grid`` / ``generate_dt`` connect the
    nodes given to ``set_nodes``. The planar builders ``generate_cdt``,
    ``generate_midpoints``, ``generate_centroids`` and ``generate_voronoi`` are
    2D only and **replace** ``nodes`` and ``clearance`` with points derived from
    a constrained Delaunay triangulation of the free space whose constraint
    outline sits on the true obstacle cell corners. ``anchor_points`` (e.g.
    start and goal) are forced to become nodes of those graphs.
    """

    def __init__(self, graph_map: "GraphSamplerBase"):
        self.graph_map = graph_map

        self._dim = self.graph_map.dim
        self._num_nodes = 0

        self.min_edge_length = graph_map.min_edge_length
        self.max_edge_length = graph_map.max_edge_length
        self.num_neighbors = int(graph_map.num_neighbors)

        self.nodes = np.zeros((0, self.dim))
        self.clearance = np.zeros(0)
        self.adj = csr_array((0, 0), dtype=float)

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_edges(self):
        return int(self.adj.nnz)

    @property
    def dim(self):
        return self._dim

    def _rows(self, values: np.ndarray) -> list:
        if self.num_nodes == 0:
            return []
        return [r.tolist() for r in np.split(values, self.adj.indptr[1:-1])]

    @property
    def road_map(self) -> List[List[int]]:
        return self._rows(self.adj.indices)

    @property
    def edge_weights(self) -> List[List[float]]:
        return self._rows(self.adj.data)

    def neighbors(self, u: int) -> np.ndarray:
        return self.adj.indices[self.adj.indptr[u]:self.adj.indptr[u + 1]]

    def in_collision(self, p1, p2=None) -> bool:
        return self.graph_map.in_collision(p1, p2)

    def set_nodes(self, samples: List[Node]):
        n = len(samples)
        self._num_nodes = n
        self.nodes = np.asarray([s.current for s in samples], dtype=float).reshape(n, self.dim)
        self.clearance = np.asarray([s.max_clearance for s in samples], dtype=float)
        self.adj = csr_array((n, n), dtype=float)

    def generate_roadmap(self, samples: List[Node]) -> csr_array:
        self.set_nodes(samples)
        return self.generate_prm()

    def generate_grid(self) -> csr_array:
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

    def _collision_args(self) -> tuple:
        m = self.graph_map
        return (np.asarray(m.bounds, dtype=np.float64), float(m.resolution),
                np.asarray(m.shape, dtype=np.int64), m._type_map_flat(), m._esdf_flat(),
                int(TYPES.OBSTACLE), int(TYPES.INFLATION), bool(m.strict_collision))

    def generate_prm(self, k_query: int | None = None, grow: int = 4) -> csr_array:
        """k-nearest PRM edges with an adaptive candidate query.

        Each round queries ``k`` candidates per unfinished node within
        ``max_edge_length``, runs the edge kernel, and keeps the rows the
        kernel proves complete. The rest are re-queried with ``k * grow``
        until ``k`` reaches the node count. Memory is O(N * k) instead of
        O(N^2) for a full query. ``k_query`` sets the first ``k``; default
        ``4 * num_neighbors + 1``. Ties are broken by node index.
        """
        n = self.num_nodes
        if n == 0:
            self.adj = csr_array((0, 0), dtype=float)
            return self.adj

        K = self.num_neighbors if self.num_neighbors > 0 else n
        k = n if k_query is None and self.num_neighbors <= 0 else (k_query or 4 * self.num_neighbors + 1)
        k = int(min(max(k, 2), n))
        args = (float(self.min_edge_length), float(self.max_edge_length), int(K)) + self._collision_args()

        tree = KDTree(self.nodes)
        # KDTree's bound is exclusive; nudge it so edges exactly at
        # max_edge_length are returned (the kernel applies the inclusive test).
        upper = np.nextafter(float(self.max_edge_length), np.inf)
        rows = np.arange(n)
        out_idx = np.full((n, K), -1, dtype=np.int64)
        out_w = np.zeros((n, K))
        count = np.zeros(n, dtype=np.int64)
        while True:
            dist, idx = tree.query(self.nodes[rows], k=k, workers=-1,
                                   distance_upper_bound=upper)
            dist = np.atleast_2d(np.asarray(dist, dtype=np.float64))
            idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
            idx[idx >= n] = 0                       # placeholder for inf entries, never read
            dist, idx = _canonical_rows(dist, idx)

            oi, ow, c, done = _prm_edges_kernel(self.nodes[rows], self.nodes, dist, idx, *args)
            if k >= n:
                done[:] = True                      # nothing left to grow into
            out_idx[rows[done]] = oi[done]
            out_w[rows[done]] = ow[done]
            count[rows[done]] = c[done]
            rows = rows[~done]
            if len(rows) == 0:
                break
            k = int(min(n, k * grow))

        self.adj = _to_csr(out_idx, out_w, count)
        return self.adj

    # ------------------------------------------------------------------ planar
    def obstacle_boundary(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Outline of the blocked region as a planar straight-line graph.

        Returns ``(points (B, 2), segments (S, 2), holes (H, 2))`` in world
        coordinates: the corners of the obstacle/inflation cells where the
        outline turns or branches, the straight segments between them (every
        cell face touching free space or the map edge, plus the closed map
        rectangle, with collinear runs merged), and one seed point inside each
        connected blocked component. Vertices lie on the cell corner lattice
        ``bounds_lo + k * resolution``, i.e. exactly where the collision
        checker and ESDF see the obstacles. 2D only.
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

    def _clearance_at(self, points: np.ndarray) -> np.ndarray:
        """Max object radius at each point's cell; 0 outside the map."""
        m = self.graph_map
        n = len(points)
        if n == 0:
            return np.zeros(0)
        cells = np.rint(np.asarray(m.path_world_to_map(points), dtype=float)).astype(np.int64).reshape(n, self.dim)
        shape = np.asarray(m.shape, dtype=np.int64)
        inside = np.all((cells >= 0) & (cells < shape), axis=1)
        out = np.zeros(n)
        if inside.any():
            out[inside] = m.inflated_esdf[tuple(cells[inside].T)]
        return out

    def _finish_planar(self, points: np.ndarray, u: np.ndarray, v: np.ndarray,
                       filter_collision: bool) -> csr_array:
        """Dedupe the points, remap and unique the edges, apply the length band
        and (optionally) the line-of-sight filter, install nodes and adjacency."""
        pts, old_to_new = _dedupe_points(points)
        u = old_to_new[np.asarray(u, dtype=np.int64)]
        v = old_to_new[np.asarray(v, dtype=np.int64)]
        loops = u != v
        eu, ev, _ = _unique_undirected(u[loops], v[loops])
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        keep, w = _filter_edges_kernel(pts, eu, ev, float(self.min_edge_length), float(self.max_edge_length),
                                       bool(filter_collision), *self._collision_args())
        self.nodes = pts
        self._num_nodes = len(pts)
        self.clearance = self._clearance_at(pts)
        self.adj = _symmetric_csr(len(pts), eu[keep], ev[keep], w[keep])
        return self.adj

    @staticmethod
    def _triangle_edges(tri: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unique undirected triangle edges and the edge id of each of the
        three slots ``(0,1), (1,2), (2,0)`` of every triangle, shape (T, 3)."""
        eu, ev, inv = _unique_undirected(tri[:, [0, 1, 2]].ravel(), tri[:, [1, 2, 0]].ravel())
        return eu, ev, inv.reshape(len(tri), 3)

    def generate_cdt(self, anchor_points=None, filter_collision: bool = True) -> csr_array:
        """Nodes are the triangulation vertices (obstacle corners, anchors and
        the current nodes), edges are the triangle edges."""
        cdt, _ = self._triangulate(self.nodes, anchor_points)
        tri = np.asarray(cdt["triangles"], dtype=np.int64)
        eu, ev, _ = self._triangle_edges(tri)
        return self._finish_planar(cdt["vertices"], eu, ev, filter_collision)

    def generate_midpoints(self, anchor_points=None, filter_collision: bool = True) -> csr_array:
        """Nodes are the midpoints of the non-constraint triangle edges, linked
        when their edges share a triangle. Each anchor becomes a node linked to
        the midpoints of its incident edges."""
        cdt, anchor_vidx = self._triangulate(self.nodes, anchor_points)
        tri = np.asarray(cdt["triangles"], dtype=np.int64)
        verts = np.asarray(cdt["vertices"], dtype=float)
        eu, ev, slot_edge = self._triangle_edges(tri)

        V = len(verts)
        segs = np.asarray(cdt["segments"], dtype=np.int64)
        constraint_keys = np.minimum(segs[:, 0], segs[:, 1]) * V + np.maximum(segs[:, 0], segs[:, 1])
        kept = ~np.isin(eu * V + ev, constraint_keys)
        mid_id = np.full(len(eu), -1, dtype=np.int64)
        mid_id[kept] = np.arange(int(kept.sum()))
        M = int(kept.sum())
        points = 0.5 * (verts[eu[kept]] + verts[ev[kept]])

        us, vs = [], []
        for a, b in ((0, 1), (1, 2), (2, 0)):
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

    def generate_centroids(self, anchor_points=None, filter_collision: bool = True) -> csr_array:
        """Nodes are the triangle centroids, linked when the triangles share an
        edge; anchors are linked to their incident triangles."""
        return self._dual_graph(lambda tp: tp.mean(axis=1), anchor_points, filter_collision)

    def generate_voronoi(self, anchor_points=None, filter_collision: bool = True) -> csr_array:
        """Nodes are the triangle circumcentres (the Voronoi vertices of the
        triangulation), linked when the triangles share an edge; anchors are
        linked to their incident triangles. Co-circular triangles share a
        circumcentre and collapse to one node."""
        def circumcentres(tp):
            p1, p2, p3 = tp[:, 0], tp[:, 1], tp[:, 2]
            a1, a2 = p2 - p1, p3 - p2                        # bisector normals
            c1 = (a1 * (p1 + p2) * 0.5).sum(axis=1)
            c2 = (a2 * (p2 + p3) * 0.5).sum(axis=1)
            det = a1[:, 0] * a2[:, 1] - a2[:, 0] * a1[:, 1]
            bad = np.abs(det) < 1e-10
            if bad.any():
                raise ValueError(f"generate_voronoi: {int(bad.sum())} degenerate (collinear) triangle(s)")
            return np.stack([(c1 * a2[:, 1] - c2 * a1[:, 1]) / det,
                             (a1[:, 0] * c2 - a2[:, 0] * c1) / det], axis=1)
        return self._dual_graph(circumcentres, anchor_points, filter_collision)

    def generate_dt(self, filter_collision: bool = True) -> csr_array:
        """Delaunay edges between the current nodes (any dimension, no
        obstacle constraints), filtered by the length band and line of sight."""
        n = self.num_nodes
        if n <= self.dim:
            self.adj = csr_array((n, n), dtype=float)
            return self.adj
        simplices = np.asarray(Delaunay(self.nodes).simplices, dtype=np.int64)
        pairs = list(combinations(range(self.dim + 1), 2))
        u = np.concatenate([simplices[:, a] for a, _ in pairs])
        v = np.concatenate([simplices[:, b] for _, b in pairs])
        eu, ev, _ = _unique_undirected(u, v)
        pts = np.ascontiguousarray(self.nodes, dtype=np.float64)
        keep, w = _filter_edges_kernel(pts, eu, ev, float(self.min_edge_length), float(self.max_edge_length),
                                       bool(filter_collision), *self._collision_args())
        self.adj = _symmetric_csr(n, eu[keep], ev[keep], w[keep])
        return self.adj

    def generate_planar(self, use_option: str = "cdt", anchor_points=None,
                        filter_collision: bool = True) -> csr_array:
        """Dispatch to ``generate_cdt`` / ``generate_midpoints`` /
        ``generate_centroids`` / ``generate_voronoi`` / ``generate_dt``."""
        if use_option == "dt":
            return self.generate_dt(filter_collision)
        builders = {"cdt": self.generate_cdt, "midpoints": self.generate_midpoints,
                    "centroids": self.generate_centroids, "voronoi": self.generate_voronoi}
        if use_option not in builders:
            raise ValueError(f"Invalid use_option: {use_option}")
        return builders[use_option](anchor_points, filter_collision)

