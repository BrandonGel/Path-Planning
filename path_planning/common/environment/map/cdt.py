"""
Run graph sampler for 2D and 3D maps.
python scripts/roadmaps/run_graph_sampler.py
"""

from python_motion_planning.common import TYPES
import os
import time
import triangle as tr
import matplotlib.pyplot as plt
import numpy as np
from python_motion_planning.common import TYPES
from scipy.ndimage import label
from path_planning.common.environment.node import Node

def _pt_key(p, nd=8):
    return (round(float(p[0]), nd), round(float(p[1]), nd))

def _edge_key_by_coords(p, q):
    a = _pt_key(p)
    b = _pt_key(q)
    return (a, b) if a < b else (b, a)


def dedupe_points_and_neighbors(
    points: np.ndarray,
    neighbors,
    nd: int = 8,
):
    """
    Deduplicate `points` (by rounded coordinate keys) and remap `neighbors`.

    Args:
        points: (N,2) array of point coordinates.
        neighbors: adjacency list where `neighbors[i]` contains indices into `points`.
        nd: rounding decimals for deduplication.

    Returns:
        new_points: (M,2) array
        new_neighbors: list[list[int]] adjacency list over new point indices.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must have shape (N,2) or (N,D>=2)")

    if isinstance(neighbors, np.ndarray):
        nbrs_list = neighbors.tolist()
    else:
        nbrs_list = neighbors

    N = len(pts)
    if len(nbrs_list) != N:
        raise ValueError(f"neighbors length ({len(nbrs_list)}) must match points length ({N})")

    old_to_new = np.empty(N, dtype=int)
    key_to_new = {}  # pt_key -> new index
    new_points_list: list[list[float]] = []

    # Stable first-occurrence dedupe
    for i in range(N):
        key = _pt_key(pts[i], nd=nd)
        j = key_to_new.get(key)
        if j is None:
            j = len(new_points_list)
            key_to_new[key] = j
            new_points_list.append([float(pts[i, 0]), float(pts[i, 1])])
        old_to_new[i] = j

    M = len(new_points_list)
    new_neighbors_sets = [set() for _ in range(M)]

    for i, nbrs in enumerate(nbrs_list):
        ni = int(old_to_new[i])
        for j in nbrs:
            nj = int(old_to_new[int(j)])
            if nj == ni:
                continue  # remove self loops
            new_neighbors_sets[ni].add(nj)

    new_points = np.asarray(new_points_list, dtype=float)
    new_neighbors = [sorted(list(s)) for s in new_neighbors_sets]
    return new_points, new_neighbors

def get_circumcenter(p1:np.ndarray, p2:np.ndarray, p3:np.ndarray):
    x1, y1 = p1[:,0], p1[:,1]
    x2, y2 = p2[:,0], p2[:,1]
    x3, y3 = p3[:,0], p3[:,1]

    # Calculate midpoints of two sides (p1-p2 and p2-p3)
    mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    mid2 = ((x2 + x3) / 2, (y2 + y3) / 2)

    # Perpendicular bisector of p1-p2: Ax + By = C
    # Slope of p1-p2 is (y2-y1)/(x2-x1). 
    # Perpendicular slope is -(x2-x1)/(y2-y1).
    A1 = x2 - x1
    B1 = y2 - y1
    C1 = A1 * mid1[0] + B1 * mid1[1]

    # Perpendicular bisector of p2-p3
    A2 = x3 - x2
    B2 = y3 - y2
    C2 = A2 * mid2[0] + B2 * mid2[1]

    # Solve the 2x2 system using Cramer's rule
    det = A1 * B2 - A2 * B1
    if np.any(abs(det) < 1e-10):
        return None  # Points are collinear

    x = (C1 * B2 - C2 * B1) / det
    y = (A1 * C2 - A2 * C1) / det

    return np.array([x, y]).T


def _merge_collinear_axis_segments(coord_segs):
    """Merge maximal runs of collinear axis-aligned unit segments into single segments.

    ``get_boundary`` emits one constraint segment per obstacle *cell face*, so a straight
    wall of N cells becomes N unit segments and N+1 vertices. Feeding that to the CDT
    produces a roadmap with one node per boundary cell (tens of thousands), which makes the
    triangulation's dense cost matrix explode. This collapses each straight run to one
    segment by contracting degree-2 "pass-through" vertices whose two neighbours are
    collinear. Corner / junction vertices (degree != 2, or a direction change) are
    preserved, so the obstacle geometry is unchanged.

    ``coord_segs`` is a list of ``(p1, p2)`` coordinate-tuple pairs. Returns the merged
    list of ``(p1, p2)`` pairs.
    """
    from collections import defaultdict

    adj = defaultdict(set)
    for p1, p2 in coord_segs:
        if p1 == p2:
            continue
        adj[p1].add(p2)
        adj[p2].add(p1)

    def collinear(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) <= 1e-7

    def removable(v):
        nb = adj[v]
        if len(nb) != 2:
            return False
        a, c = tuple(nb)
        return collinear(a, v, c)

    def ek(a, b):
        return (a, b) if a <= b else (b, a)

    visited: set = set()
    out = []
    anchors = [v for v in adj if not removable(v)]
    for a in anchors:
        for nb in list(adj[a]):
            if ek(a, nb) in visited:
                continue
            prev, cur = a, nb
            visited.add(ek(prev, cur))
            while removable(cur):
                nxt = next(w for w in adj[cur] if w != prev)
                if ek(cur, nxt) in visited:
                    break
                visited.add(ek(cur, nxt))
                prev, cur = cur, nxt
            if a != cur:
                out.append((a, cur))
    # Any edges not reached from an anchor belong to a fully-collinear loop (no corners);
    # keep them as-is (does not occur for closed rectilinear boundaries, but be safe).
    for p1, p2 in coord_segs:
        if p1 == p2 or ek(p1, p2) in visited:
            continue
        visited.add(ek(p1, p2))
        out.append((p1, p2))
    return out


def get_boundary(map_,mask:np.ndarray):
    dim = getattr(map_, "dim", 2)
    if dim != 2 or mask.ndim != 2:
        raise NotImplementedError(
            "Planar/CDT boundary extraction requires a 2D occupancy grid (dim=2, mask H×W)."
        )
    use_discrete_space = map_.use_discrete_space
    offset = map_.map_to_world((0, 0)) if use_discrete_space else (0.0, 0.0)

    mask = (map_.type_map.data == TYPES.OBSTACLE) | (map_.type_map.data == TYPES.INFLATION)
    H, W = mask.shape
    res = float(map_.resolution)
    b = map_.bounds

    # --- 0) Get the holes where there are no connecting edges inside ---
    # One seed point per connected obstacle component: its first cell in row-major order
    # (what np.argwhere(labeled == comp)[0] returned, found here in a single pass instead of
    # one full-grid scan per component).
    labeled, num = label(mask)
    holes = []
    if num > 0:
        labs, first = np.unique(labeled.ravel(), return_index=True)
        for comp, idx in zip(labs.tolist(), first.tolist()):
            if comp == 0:
                continue
            r0, c0 = divmod(int(idx), W)
            hx = float(b[0, 0]) + (r0 + 0.5) * res + offset[0]
            hy = float(b[1, 0]) + (c0 + 0.5) * res + offset[1]
            holes.append((hx, hy))

    # --- 1) boundary segments (world-space) ---
    # A cell face is on the boundary if it touches free space or the map boundary. Done per
    # face direction on the whole grid (a Python loop over the ~5M masked cells took ~4 s);
    # the corner coordinates are computed with the same float operations as before so the
    # vertices are bit-identical.
    b00, b10 = float(b[0, 0]), float(b[1, 0])
    off0, off1 = offset[0], offset[1]
    padded = np.zeros((H + 2, W + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask

    def corners_of(rr, cc):
        x0 = b00 + rr * res + off0
        y0 = b10 + cc * res + off1
        return [(x0, y0), (x0 + res, y0), (x0 + res, y0 + res), (x0, y0 + res)]

    # (dr, dc, corner_idx_1, corner_idx_2) for each face of cell (r,c)
    faces = [(-1, 0, 0, 3), (1, 0, 1, 2), (0, -1, 0, 1), (0, 1, 2, 3)]
    seg_rows = []
    for dr, dc, ci1, ci2 in faces:
        neighbour = padded[1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc]  # False outside the grid
        rr, cc = np.nonzero(mask & ~neighbour)
        if rr.size == 0:
            continue
        cs = corners_of(rr, cc)
        (x1, y1), (x2, y2) = cs[ci1], cs[ci2]
        swap = (x1 > x2) | ((x1 == x2) & (y1 > y2))  # tuple order: p1 < p2
        seg_rows.append(np.stack([np.where(swap, x2, x1), np.where(swap, y2, y1),
                                  np.where(swap, x1, x2), np.where(swap, y1, y2)], axis=1))
    if seg_rows:
        segs = np.unique(np.concatenate(seg_rows, axis=0), axis=0)  # == sorted(seg_set)
    else:
        segs = np.empty((0, 4), dtype=float)

    # --- 2) unique boundary vertices ---
    endpoints = np.concatenate([segs[:, :2], segs[:, 2:]], axis=0)
    bnd_pts, inverse = np.unique(endpoints, axis=0, return_inverse=True)  # sorted (N,2)
    inverse = np.asarray(inverse).reshape(-1)
    bnd_pts = np.asarray(bnd_pts, dtype=float).reshape(-1, 2)

    # segments as pairs of vertex indices
    n_seg = len(segs)
    bnd_segs = np.stack([inverse[:n_seg], inverse[n_seg:]], axis=1).astype(int)

    # --- 3) add outer map rectangle so the domain is enclosed ---
    x_min, x_max = float(b[0, 0]) + offset[0], float(b[0, 1]) + offset[0]
    y_min, y_max = float(b[1, 0]) + offset[1], float(b[1, 1]) + offset[1]


    nx = int(round((x_max - x_min) / res))
    ny = int(round((y_max - y_min) / res))

    outer_edge_pts = []
    # bottom
    for i in range(nx + 1):
        outer_edge_pts.append((x_min + i * res, y_min))
    # right (exclude bottom-right corner)
    for j in range(1, ny + 1):
        outer_edge_pts.append((x_max, y_min + j * res))
    # top (exclude top-right corner)
    for i in range(nx - 1, -1, -1):
        outer_edge_pts.append((x_min + i * res, y_max))
    # left (exclude top-left and bottom-left corners)
    for j in range(ny - 1, 0, -1):
        outer_edge_pts.append((x_min, y_min + j * res))

    key_to_idx = {_pt_key(bnd_pts[i]): i for i in range(len(bnd_pts))}

    outer_edge_indices = []
    outer_new_pts = []

    base_n = len(bnd_pts)
    for p in outer_edge_pts:
        k = _pt_key(p)
        if k in key_to_idx:
            outer_edge_indices.append(key_to_idx[k])
        else:
            outer_edge_indices.append(base_n + len(outer_new_pts))
            outer_new_pts.append(p)
            key_to_idx[k] = outer_edge_indices[-1]

    if len(outer_new_pts) > 0:
        bnd_pts = np.vstack([bnd_pts, np.array(outer_new_pts, dtype=float)])

    # --- 4) Build chain segments between consecutive outer edge points (closed loop) ---
    outer_chain_segs = [
        (outer_edge_indices[i], outer_edge_indices[i + 1])
        for i in range(len(outer_edge_indices) - 1)
    ]
    outer_chain_segs.append((outer_edge_indices[-1], outer_edge_indices[0]))

    # --- 5) Remove the old 4 corner-to-corner outer segments (if present) ---
    corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    c_corner_idx = [key_to_idx[_pt_key(p)] for p in corners]
    corner_pairs = {
        tuple(sorted((c_corner_idx[0], c_corner_idx[1]))),
        tuple(sorted((c_corner_idx[1], c_corner_idx[2]))),
        tuple(sorted((c_corner_idx[2], c_corner_idx[3]))),
        tuple(sorted((c_corner_idx[3], c_corner_idx[0]))),
    }

    # --- 7) Get the unique boundary segments ---
    seg_set = set()
    for u, v in bnd_segs.tolist():
        if int(u) == int(v):
            continue
        uu, vv = int(u), int(v)
        key = tuple(sorted((uu, vv)))
        if key in corner_pairs:
            continue
        seg_set.add(key)

    for u, v in outer_chain_segs:
        if int(u) == int(v):
            continue
        seg_set.add(tuple(sorted((int(u), int(v)))))

    bnd_segs = np.array(list(seg_set), dtype=int)

    # --- 8) Merge collinear runs: collapse each straight wall (many unit cell-faces)
    # into a single constraint segment, so the CDT roadmap has one vertex per corner /
    # junction instead of one per boundary cell (~tens of thousands -> hundreds).
    def _rk(p):
        return (round(float(p[0]), 8), round(float(p[1]), 8))

    coord_segs = [(_rk(bnd_pts[u]), _rk(bnd_pts[v])) for u, v in bnd_segs.tolist()]
    merged = _merge_collinear_axis_segments(coord_segs)

    pt_index: dict = {}
    new_pts: list = []
    new_segs: list = []
    seen_seg: set = set()
    for p1, p2 in merged:
        for p in (p1, p2):
            if p not in pt_index:
                pt_index[p] = len(new_pts)
                new_pts.append([float(p[0]), float(p[1])])
        a, bb = pt_index[p1], pt_index[p2]
        key = (a, bb) if a < bb else (bb, a)
        if a != bb and key not in seen_seg:
            seen_seg.add(key)
            new_segs.append([a, bb])
    bnd_pts = np.array(new_pts, dtype=float)
    bnd_segs = np.array(new_segs, dtype=int)

    return bnd_pts,bnd_segs,holes

def get_all_points(interior_points:np.ndarray,bnd_pts:np.ndarray):
    """
    Return all unique points from interior + boundary.

    - Preserves a stable order: boundary points first, then interior points.
      This matches how constrained CDT segment indices (`bnd_segs`) are defined
      against `bnd_pts`.
    - Uniqueness is determined by rounding with `_pt_key` (nd=8), consistent with
      other geometry keying in this module.
    """
    interior_points = np.asarray(interior_points, dtype=float)
    bnd_pts = np.asarray(bnd_pts, dtype=float)
    if interior_points.ndim != 2 or interior_points.shape[1] < 2:
        raise ValueError("interior_points must have shape (N,2) or (N,D>=2)")
    if bnd_pts.ndim != 2 or bnd_pts.shape[1] < 2:
        raise ValueError("bnd_pts must have shape (M,2) or (M,D>=2)")

    seen: set[tuple[float, float]] = set()
    uniq: list[list[float]] = []
    for p in np.vstack((bnd_pts,interior_points)):
        k = _pt_key(p, nd=8)
        if k in seen:
            continue
        seen.add(k)
        uniq.append([float(p[0]), float(p[1])])
    return np.asarray(uniq, dtype=float)


def get_all_points_and_segments(
    interior_points: np.ndarray,
    bnd_pts: np.ndarray,
    bnd_segs: np.ndarray,
    nd: int = 8,
):
    """Deduplicate boundary + interior vertices (boundary first) AND remap the boundary
    constraint segments to the deduplicated indices.

    ``get_boundary`` can emit duplicate boundary coordinates. Deduplicating the vertex
    array alone (as :func:`get_all_points` does) silently shifts vertex indices, so
    ``bnd_segs`` — defined against the *original* ``bnd_pts`` order — ends up referencing
    the wrong vertices. The resulting PSLG has crossing/overlapping constraints, which
    makes ``triangle`` abort with "Topological inconsistency after splitting a segment".
    Remapping the segments through the same dedup keeps the PSLG consistent. Degenerate
    (now zero-length) and duplicate segments are dropped.

    Returns ``(all_points, segments)`` ready for the constrained triangulation.
    """
    bnd_pts = np.asarray(bnd_pts, dtype=float)
    interior_points = np.asarray(interior_points, dtype=float)
    if bnd_pts.ndim != 2 or bnd_pts.shape[1] < 2:
        raise ValueError("bnd_pts must have shape (M,2) or (M,D>=2)")

    key_to_new: dict = {}
    uniq: list[list[float]] = []
    old_bnd_to_new = np.empty(len(bnd_pts), dtype=int)
    for i in range(len(bnd_pts)):
        k = _pt_key(bnd_pts[i], nd=nd)
        j = key_to_new.get(k)
        if j is None:
            j = len(uniq)
            key_to_new[k] = j
            uniq.append([float(bnd_pts[i, 0]), float(bnd_pts[i, 1])])
        old_bnd_to_new[i] = j
    if interior_points.size:
        for p in interior_points:
            k = _pt_key(p, nd=nd)
            if k not in key_to_new:
                key_to_new[k] = len(uniq)
                uniq.append([float(p[0]), float(p[1])])

    all_points = np.asarray(uniq, dtype=float)
    seen_seg: set = set()
    remapped: list[list[int]] = []
    for u, v in np.asarray(bnd_segs, dtype=int):
        a, b = int(old_bnd_to_new[u]), int(old_bnd_to_new[v])
        if a == b:
            continue  # collapsed to a single vertex by dedup -> degenerate
        key = (a, b) if a < b else (b, a)
        if key in seen_seg:
            continue
        seen_seg.add(key)
        remapped.append([a, b])
    return all_points, np.asarray(remapped, dtype=int)


def get_constrained_delaunay_triangulation(bnd_pts:np.ndarray,bnd_segs:np.ndarray,holes:list):
    A_cdt = dict(vertices=bnd_pts, segments=bnd_segs,holes=holes)
    cdt = tr.triangulate(A_cdt,'pc')
    return cdt

def connect_cdt(cdt:dict):
    """
    Build a neighbor adjacency list from a CDT triangulation result.

    Uses unique undirected edges extracted from `cdt['triangles']`.

    Returns:
        vertices: (V,2) float array
        neighbors: list[list[int]] adjacency list over V vertices
    """
    vertices = np.asarray(cdt["vertices"], dtype=float)
    triangles = np.asarray(cdt["triangles"], dtype=int)
    V = len(vertices)

    seen = set()
    edges = []
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for x, y in ((a, b), (b, c), (c, a)):
            if x == y:
                continue
            u, v = (x, y) if x < y else (y, x)
            key = (u, v)
            if key in seen:
                continue
            seen.add(key)
            edges.append(key)
    
    neighbors = [[] for _ in range(V)]
    for u, v in edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        neighbors[u].append(v)
        neighbors[v].append(u)
    # Deduplicate adjacency lists (in case edges repeated)
    neighbors = [sorted(list(set(nbrs))) for nbrs in neighbors]
    return vertices, neighbors

def connect_midpoints(triangles:np.ndarray,verts:np.ndarray,bnd_pts:np.ndarray,bnd_segs:np.ndarray,start_goal_indices={}):
    triangles = np.asarray(triangles, dtype=int)
    verts = np.asarray(verts, dtype=float)
    # start_goal_indices is expected to be dict-like: {Node -> vertex_index_in_map_nodes}.
    # For connectivity in this CDT-based graph, we instead map each start/goal position
    # to the corresponding CDT vertex index inside `verts` (coordinate matching).
    start_goal_items = list(start_goal_indices.items()) if hasattr(start_goal_indices, "items") else []
    K = len(start_goal_items)
    extra_neighbors = [[] for _ in range(K)]  # extra node id (0..K-1) -> list of midpoint node ids
    if K > 0:
        verts_key_to_idx = {}
        for i in range(len(verts)):
            verts_key_to_idx[_pt_key(verts[i])] = i

    # Boundary edges as coordinate keys, using bnd_pts/bnd_segs from the PSLG input.
    boundary_edge_keys = set()
    for u, v in np.asarray(bnd_segs, dtype=int):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        boundary_edge_keys.add(_edge_key_by_coords(bnd_pts[u], bnd_pts[v]))

    # 1) unique undirected triangulation edges excluding boundary edges
    edge_to_mid = {}  # (u_idx, v_idx) in `verts` -> midpoint id
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for x, y in ((a, b), (b, c), (c, a)):
            if x == y:
                continue
            lo, hi = (x, y) if x < y else (y, x)
            if (lo, hi) in edge_to_mid:
                continue
            if _edge_key_by_coords(verts[x], verts[y]) in boundary_edge_keys:
                continue
            edge_to_mid[(lo, hi)] = len(edge_to_mid)

    num_edges = len(edge_to_mid)
    midpoints = np.zeros((num_edges + K, 2), dtype=float)
    for ii, (node, _) in enumerate(start_goal_items):
        midpoints[num_edges + ii] = np.asarray(node.current, dtype=float)[:2]
    for (u, v), eid in edge_to_mid.items():
        midpoints[eid] = 0.5 * (verts[u] + verts[v])

        # Note: start/goal -> midpoint connectivity is added after we build
        # the sg-vertex-to-extra-node mapping (see `sg_vidx_to_extra` below).
    # Build vertex-index -> extra-node-id mapping once (after verts_key_to_idx exists).
    if K > 0:
        sg_vidx_to_extra = {}
        for ii, (node, _) in enumerate(start_goal_items):
            sg_key = _pt_key(node.current)
            if sg_key in verts_key_to_idx:
                sg_vidx_to_extra[int(verts_key_to_idx[sg_key])] = ii
    else:
        sg_vidx_to_extra = {}

    # Re-run incident detection now that we have a robust sg_vidx_to_extra mapping.
    # (This keeps the logic simple and avoids relying on map-nodes indexing.)
    if K > 0:
        for (u, v), eid in edge_to_mid.items():
            if u in sg_vidx_to_extra:
                extra_neighbors[sg_vidx_to_extra[u]].append(eid)
            if v in sg_vidx_to_extra:
                extra_neighbors[sg_vidx_to_extra[v]].append(eid)

    # 2) midpoint adjacency: connect midpoints of edges that co-occur in triangles
    adj_edges = set()
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        e0 = (a, b) if a < b else (b, a)
        e1 = (b, c) if b < c else (c, b)
        e2 = (c, a) if c < a else (a, c)

        # Pairwise within triangle, but only if both edges were kept.
        for ea, eb in ((e0, e1), (e1, e2), (e2, e0)):
            if ea not in edge_to_mid or eb not in edge_to_mid:
                continue
            ma = edge_to_mid[ea]
            mb = edge_to_mid[eb]
            if ma == mb:
                continue
            adj_edges.add((ma, mb) if ma < mb else (mb, ma))
    if K > 0:
        for ii in range(K):
            extra_id = num_edges + ii
            for neighbor_mid in extra_neighbors[ii]:
                a, b = extra_id, int(neighbor_mid)
                adj_edges.add((a, b) if a < b else (b, a))
    midpoint_edges = np.array(sorted(adj_edges), dtype=int)
    midpoint_edges = [(int(edge[0]), int(edge[1])) for edge in midpoint_edges]

    total_nodes = num_edges + K
    midpoint_neighbors = [[] for _ in range(total_nodes)]
    for u, v in midpoint_edges:
        midpoint_neighbors[u].append(int(v))
        midpoint_neighbors[v].append(int(u))
    return midpoints,midpoint_neighbors

def connect_centroids(triangles: np.ndarray, verts: np.ndarray, bnd_pts:np.ndarray, bnd_segs:np.ndarray, start_goal_indices={}):
    """
    Compute triangle centroids and connect neighboring triangles' centroids

    Two triangles are considered neighbors if they share an undirected edge.

    Returns:
        voronoi_points: (T + K, 2) float array of Voronoi points, optionally appended with K start/goal points.
        voronoi_neighbors: list[list[int]] adjacency list over (triangles + start/goal nodes).
    """
    triangles = np.asarray(triangles, dtype=int)
    verts = np.asarray(verts, dtype=float)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")
    if verts.ndim != 2 or verts.shape[1] < 2:
        raise ValueError("verts must have shape (V, 2) or (V, D>=2)")
    tri_pts = verts[triangles]  # (T,3,2)
    centroids = tri_pts.mean(axis=1)  # (T,2)
    num_bdry_pts = len(bnd_pts)

    # Build triangle adjacency via shared edges
    edge_to_tri = {}  # (min_vi, max_vi) -> first triangle index that had it
    T = len(triangles)
    adj = [set() for _ in range(T)]
    for ti, tri in enumerate(triangles):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for x, y in ((a, b), (b, c), (c, a)):
            if x == y:
                continue
            e = (x, y) if x < y else (y, x)
            other = edge_to_tri.get(e)
            if other is None:
                edge_to_tri[e] = ti
            else:
                if other != ti:
                    adj[ti].add(int(other))
                    adj[int(other)].add(int(ti))


    # Optionally append start/goal nodes and connect them to incident triangle centroids.
    if start_goal_indices is None or len(start_goal_indices) == 0:
        centroid_neighbors = [sorted(list(nbrs)) for nbrs in adj]
        return centroids, centroid_neighbors

    # Accept either:
    # - dict-like: {Node -> vertex_index} (used elsewhere in this repo), or
    # - iterable of vertex indices
    sg_positions = np.asarray([node.current for node in start_goal_indices.keys()], dtype=float)
    centroids_all = np.vstack([centroids, sg_positions])  # (T+K,2)
    adj_all = [set(n) for n in adj]
    adj_all.extend([set() for _ in range(len(start_goal_indices))])

    # # Map each vertex index -> triangles that include it
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    for i, (node,index) in enumerate(start_goal_indices.items()):
        vidx = int(num_bdry_pts + index)
        incident = np.where((v0 == vidx) | (v1 == vidx) | (v2 == vidx))[0]
        pass
        for ti in incident.tolist():
            adj_all[T+i].add(int(ti))
            adj_all[int(ti)].add(T+i)
    
    centroid_neighbors = [sorted(list(nbrs)) for nbrs in adj_all]
    return centroids_all, centroid_neighbors

def connect_voronoi(triangles: np.ndarray, verts: np.ndarray, bnd_pts:np.ndarray, bnd_segs:np.ndarray, start_goal_indices={}):
    """
    Compute triangle circumcenters and connect neighboring triangles' circumcenters, creating a Voronoi graph.

    Two triangles are considered neighbors if they share an undirected edge.

    Returns:
        voronoi_points: (T + K, 2) float array of Voronoi points, optionally appended with K start/goal points.
        voronoi_neighbors: list[list[int]] adjacency list over (triangles + start/goal nodes).
    """
    triangles = np.asarray(triangles, dtype=int)
    verts = np.asarray(verts, dtype=float)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")
    if verts.ndim != 2 or verts.shape[1] < 2:
        raise ValueError("verts must have shape (V, 2) or (V, D>=2)")
    tri_pts = verts[triangles]  # (T,3,2)
    voronoi_points = get_circumcenter(tri_pts[:,0],tri_pts[:,1],tri_pts[:,2])  # (T,2)
    num_bdry_pts = len(bnd_pts)

    # Build triangle adjacency via shared edges
    edge_to_tri = {}  # (min_vi, max_vi) -> first triangle index that had it
    T = len(triangles)
    adj = [set() for _ in range(T)]
    for ti, tri in enumerate(triangles):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for x, y in ((a, b), (b, c), (c, a)):
            if x == y:
                continue
            e = (x, y) if x < y else (y, x)
            other = edge_to_tri.get(e)
            if other is None:
                edge_to_tri[e] = ti
            else:
                if other != ti:
                    adj[ti].add(int(other))
                    adj[int(other)].add(int(ti))


    # Optionally append start/goal nodes and connect them to incident triangle centroids.
    if start_goal_indices is None or len(start_goal_indices) == 0:
        voronoi_neighbors = [sorted(list(nbrs)) for nbrs in adj]
        return voronoi_points, voronoi_neighbors

    # Accept either:
    # - dict-like: {Node -> vertex_index} (used elsewhere in this repo), or
    # - iterable of vertex indices
    sg_positions = np.asarray([node.current for node in start_goal_indices.keys()], dtype=float)
    voronoi_points_all = np.vstack([voronoi_points, sg_positions])  # (T+K,2)
    adj_all = [set(n) for n in adj]
    adj_all.extend([set() for _ in range(len(start_goal_indices))])

    # # Map each vertex index -> triangles that include it
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    for i, (node,index) in enumerate(start_goal_indices.items()):
        vidx = int(num_bdry_pts + index)
        incident = np.where((v0 == vidx) | (v1 == vidx) | (v2 == vidx))[0]
        pass
        for ti in incident.tolist():
            adj_all[T+i].add(int(ti))
            adj_all[int(ti)].add(T+i)
    
    voronoi_neighbors = [sorted(list(nbrs)) for nbrs in adj_all]
    return voronoi_points_all, voronoi_neighbors

def get_planar_graph(map_,mask:np.ndarray, use_option:str = 'cdt'):
    interior_points = np.array([p.current for p in map_.nodes])
    start_goal_indices = map_.start_nodes_index | map_.goal_nodes_index
    
    bnd_pts,bnd_segs,holes = get_boundary(map_,mask)
    # Deduplicate vertices AND remap the constraint segments together: get_boundary can
    # emit duplicate coordinates, and deduping vertices without remapping segments leaves
    # the segments pointing at the wrong vertices (triangulation then aborts with a
    # topological inconsistency). See get_all_points_and_segments.
    all_points, tri_segs = get_all_points_and_segments(interior_points, bnd_pts, bnd_segs)
    cdt = get_constrained_delaunay_triangulation(all_points,tri_segs,holes)
    if use_option == 'cdt':
        points,neighbors = connect_cdt(cdt)
    elif use_option == 'midpoints':
        points,neighbors = connect_midpoints(cdt['triangles'],cdt['vertices'],bnd_pts,bnd_segs,start_goal_indices)
    elif use_option == 'centroids':
        points,neighbors = connect_centroids(cdt['triangles'],cdt['vertices'],bnd_pts,bnd_segs,start_goal_indices)
    elif use_option == 'voronoi':
        points,neighbors = connect_voronoi(cdt['triangles'],cdt['vertices'],bnd_pts,bnd_segs,start_goal_indices)
    else:
        raise ValueError(f"Invalid use_option: {use_option}")
    points, neighbors = dedupe_points_and_neighbors(points,neighbors)
    return  points,neighbors
