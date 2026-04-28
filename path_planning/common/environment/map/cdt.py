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


def get_boundary(map_,mask:np.ndarray):
    use_discrete_space = map_.use_discrete_space
    offset = map_.map_to_world((0,0)) if use_discrete_space else (0,0) 
    
    mask = (map_.type_map.data == TYPES.OBSTACLE) | (map_.type_map.data == TYPES.INFLATION)
    H, W = mask.shape
    res = float(map_.resolution)
    b = map_.bounds

    # --- 0) Get the holes where there are no connecting edges inside ---
    labeled, num = label(mask)
    holes = []
    for comp in range(1, num + 1):
        coords = np.argwhere(labeled == comp)
        if coords.shape[0] == 0:
            continue
        r0, c0 = int(coords[0, 0]), int(coords[0, 1])
        hx = float(b[0, 0]) + (r0 + 0.5) * res + offset[0]
        hy = float(b[1, 0]) + (c0 + 0.5) * res + offset[1]
        holes.append((hx, hy))

    # --- 1) boundary segments (world-space) ---
    # A cell face is on the boundary if it touches free space or the map boundary.
    rows, cols = np.where(mask)
    seg_set = set()

    # cell corners (world coordinates)
    def cell_corners(r: int, c: int):
        x0 = float(b[0, 0]) + r * res + offset[0]
        y0 = float(b[1, 0]) + c * res + offset[1]
        return [
            (x0, y0),
            (x0 + res, y0),
            (x0 + res, y0 + res),
            (x0, y0 + res),
        ]

    # (dr, dc, corner_idx_1, corner_idx_2) for each face of cell (r,c)
    faces = [(-1, 0, 0, 3), (1, 0, 1, 2), (0, -1, 0, 1), (0, 1, 2, 3)]
    for r, c in zip(rows.tolist(), cols.tolist()):
        corners = cell_corners(r, c)
        for dr, dc, ci1, ci2 in faces:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W or (not mask[nr, nc]):
                p1, p2 = corners[ci1], corners[ci2]
                if p1 > p2:
                    p1, p2 = p2, p1
                seg_set.add((p1, p2))

    # --- 2) unique boundary vertices ---
    pt_set = set()
    for p1, p2 in seg_set:
        pt_set.add(p1)
        pt_set.add(p2)

    bnd_pts = np.array(sorted(pt_set), dtype=float)  # (N,2)

    # map point -> index
    pt_index = {pt: i for i, pt in enumerate(sorted(pt_set))}

    # segments as pairs of vertex indices
    bnd_segs = np.array(
        [(pt_index[p1], pt_index[p2]) for p1, p2 in sorted(seg_set)],
        dtype=int,
    )

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
    all_points = get_all_points(interior_points, bnd_pts)
    cdt = get_constrained_delaunay_triangulation(all_points,bnd_segs,holes)
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
