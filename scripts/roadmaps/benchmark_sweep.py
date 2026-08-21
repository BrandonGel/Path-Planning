"""
Head-to-head benchmark: CGAL vs Shapely spatial-sweep backends.

Builds a real 2D PRM roadmap, then runs an identical batch of swept-collision
queries through both ``CGAL_Sweep`` and ``ShapelySweep`` (same ``set_graph``,
same queries), reporting per-backend timing and result agreement (overlapping
vertex/edge sets equal; time intervals within tolerance).

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

from path_planning.utils.cgal_sweep import CGAL_Sweep
from path_planning.utils.shapely_sweep import ShapelySweep
from path_planning.utils.util import read_graph_sampler_from_yaml, set_global_seed


def _build_roadmap(map_yaml: str, sample_num: int, seed: int):
    """Build a 2D continuous PRM roadmap; return (vertices, edges)."""
    set_global_seed(seed)
    m = read_graph_sampler_from_yaml(map_yaml, use_discrete_space=False)
    m.set_parameters(sample_num=sample_num, num_neighbors=10.0, min_edge_len=0.0, max_edge_len=4.0)
    m.set_start([]); m.set_goal([])
    nodes = m.generateRandomNodes(generate_grid_nodes=False, roadmap_type="prm")
    m.generate_map("prm", nodes)
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
    """Time both query methods over all queries; return (t_elems, t_interval, results)."""
    sweep.set_graph(vertices, edges)
    sweep.record_sweep = False  # measure raw work, not cache hits

    results = []
    t0 = time.perf_counter()
    for (u, v) in queries:
        results.append(sweep.overlapping_graph_elements_cgal(u, v, velocity, r))
    t_elems = time.perf_counter() - t0

    interval_results = []
    t0 = time.perf_counter()
    for (u, v) in queries:
        interval_results.append(sweep.overlapping_interval_cgal(u, v, velocity, r, get_time_interval=True))
    t_interval = time.perf_counter() - t0

    return t_elems, t_interval, results, interval_results


def _agreement(cgal_res, shp_res, cgal_iv, shp_iv, atol=1e-6):
    """Compare per-query results between the two backends."""
    elem_mismatch = sum(1 for a, b in zip(cgal_res, shp_res) if set(a) != set(b))

    vtx_mismatch = edge_key_mismatch = interval_mismatch = 0
    for (cv, ce), (sv, se) in zip(cgal_iv, shp_iv):
        if set(cv.keys()) != set(sv.keys()):
            vtx_mismatch += 1
        if set(ce.keys()) != set(se.keys()):
            edge_key_mismatch += 1
        for k in set(ce.keys()) & set(se.keys()):
            (a0, a1), (b0, b1) = ce[k], se[k]
            if abs(a0 - b0) > atol or abs(a1 - b1) > atol:
                interval_mismatch += 1
                break
    return elem_mismatch, vtx_mismatch, edge_key_mismatch, interval_mismatch


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

    cgal_e, cgal_i, cgal_res, cgal_iv = _run_backend(CGAL_Sweep(), vertices, edges, queries, velocity, r)
    shp_e, shp_i, shp_res, shp_iv = _run_backend(ShapelySweep(), vertices, edges, queries, velocity, r)

    print("\n--- timing (total over all queries) ---")
    print(f"  overlapping_graph_elements : CGAL {cgal_e*1e3:8.1f} ms | Shapely {shp_e*1e3:8.1f} ms | x{cgal_e/shp_e:.2f}")
    print(f"  overlapping_interval       : CGAL {cgal_i*1e3:8.1f} ms | Shapely {shp_i*1e3:8.1f} ms | x{cgal_i/shp_i:.2f}")
    print(f"  per-query (interval)       : CGAL {cgal_i/n_queries*1e6:7.1f} us | Shapely {shp_i/n_queries*1e6:7.1f} us")

    em, vm, ekm, im = _agreement(cgal_res, shp_res, cgal_iv, shp_iv)
    print("\n--- agreement (# queries differing, out of {}) ---".format(n_queries))
    print(f"  edge-element set : {em}")
    print(f"  interval vtx keys: {vm}")
    print(f"  interval edge keys: {ekm}")
    print(f"  interval values  : {im}")
    print("  (small counts are expected from boundary <= vs < and float rounding)")


if __name__ == "__main__":
    main()
