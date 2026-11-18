"""
Head-to-head benchmark: CGAL vs Shapely disc-sweep backends.

Builds a real 2D PRM roadmap, then runs an identical batch of moves and waits through
``overlapping_interval_paths`` of both ``CGALCollision`` and ``ShapelyCollision`` (same ``set_graph``, same
queries): one call per move, and all moves in one batched call. Reports per-backend timing and result
agreement (vertex / edge key sets equal; time windows within tolerance).

Usage:
    python scripts/roadmaps/benchmark_sweep.py [n_queries] [radius]
"""

from __future__ import annotations

import os
import random
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from path_planning.common.world.collision_checking.cgal.cgal_collision import CGALCollision
from path_planning.common.world.collision_checking.shapely.shapely_collision import ShapelyCollision
from path_planning.utils.util import read_graph_sampler_from_yaml, set_global_seed


def _build_roadmap(map_yaml: str, sample_num: int, seed: int):
    """Build a 2D continuous PRM roadmap; return (vertices, edges)."""
    set_global_seed(seed)
    m = read_graph_sampler_from_yaml(map_yaml, use_discrete_space=False)
    m.set_roadmap_config(sample_num=sample_num, num_neighbors=10.0, min_edge_length=0.0, max_edge_length=4.0)
    m.set_start([]); m.set_goal([])
    nodes = m.generateRandomNodes(generate_grid_nodes=False)
    m.generate_roadmap("prm")
    vertices = [tuple(float(c) for c in n.current) for n in m.nodes]
    edges = [tuple(e) for e in m.edges]
    return vertices, edges, m.road_map


def _make_queries(vertices, road_map, n_queries, rng):
    """Random query segments (u, v) over real roadmap geometry, plus some point
    queries (u == v). Returns list of (u, v) world-coord tuples."""
    n = len(vertices)
    queries = []
    for _ in range(n_queries):
        i = rng.randrange(n)
        u = vertices[i]
        roll = rng.random()
        if roll < 0.2:
            v = u  # stationary point query
        elif road_map[i]:
            v = vertices[rng.choice(road_map[i])]  # move along an edge
        else:
            v = vertices[rng.randrange(n)]
        queries.append((u, v))
    return queries


def _run_backend(sweep, vertices, edges, queries, velocity, r):
    """Time the queries one call each, then all in one batched call; return (t_single, t_batch, results)."""
    sweep.set_graph(vertices, edges)
    paths = [[u] if u == v else [u, v] for (u, v) in queries]
    holds = [float("inf") if u == v else 0.0 for (u, v) in queries]

    t0 = time.perf_counter()
    results = [sweep.overlapping_interval_paths(p, r, velocity, goal_hold=h, as_arrays=False)[0] for p, h in zip(paths, holds)]
    t_single = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched = sweep.overlapping_interval_paths(paths, r, velocity, goal_hold=holds, as_arrays=False)
    t_batch = time.perf_counter() - t0
    assert batched == results, "batched and one-by-one results differ"
    return t_single, t_batch, results


def _agreement(cgal_res, shp_res, atol=1e-6):
    """Compare per-query results between the two backends."""
    vtx_mismatch = edge_key_mismatch = interval_mismatch = 0
    for (cv, ce), (sv, se) in zip(cgal_res, shp_res):
        if set(cv) != set(sv):
            vtx_mismatch += 1
        if set(ce) != set(se):
            edge_key_mismatch += 1
        for k in (set(ce) & set(se)) | (set(cv) & set(sv)):
            a, b = (ce, se) if k in ce else (cv, sv)
            if any(abs(x0 - y0) > atol or abs(x1 - y1) > atol
                   for (x0, x1), (y0, y1) in zip(a[k], b[k]) if not (x1 == y1 == float("inf"))):
                interval_mismatch += 1
                break
    return vtx_mismatch, edge_key_mismatch, interval_mismatch


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    r = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    velocity = 1.0
    seed = 42
    map_yaml = "path_planning/maps/2d/2d_mapd.yaml"

    print(f"=== Sweep backend benchmark | r={r} velocity={velocity} n_queries={n_queries} ===")
    vertices, edges, road_map = _build_roadmap(map_yaml, sample_num=800, seed=seed)
    print(f"roadmap: {len(vertices)} vertices, {len(edges)} edges")

    rng = random.Random(seed)
    queries = _make_queries(vertices, road_map, n_queries, rng)

    cgal_s, cgal_b, cgal_res = _run_backend(CGALCollision(), vertices, edges, queries, velocity, r)
    shp_s, shp_b, shp_res = _run_backend(ShapelyCollision(), vertices, edges, queries, velocity, r)

    print("\n--- timing (total over all queries) ---")
    print(f"  one call per query : CGAL {cgal_s*1e3:8.1f} ms | Shapely {shp_s*1e3:8.1f} ms | x{cgal_s/shp_s:.2f}")
    print(f"  one batched call   : CGAL {cgal_b*1e3:8.1f} ms | Shapely {shp_b*1e3:8.1f} ms | x{cgal_b/shp_b:.2f}")
    print(f"  per query (batched): CGAL {cgal_b/n_queries*1e6:7.1f} us | Shapely {shp_b/n_queries*1e6:7.1f} us")

    vm, ekm, im = _agreement(cgal_res, shp_res)
    print("\n--- agreement (# queries differing, out of {}) ---".format(n_queries))
    print(f"  vertex keys : {vm}")
    print(f"  edge keys   : {ekm}")
    print(f"  windows     : {im}")
    print("  (small counts are expected from float rounding at the strict < r boundary)")


if __name__ == "__main__":
    main()
