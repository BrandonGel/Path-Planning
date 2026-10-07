"""Consistency checks for the unlearned-embedding controls (and gnn6).

python scripts/test/check_controls.py --gnn-repro
    Re-run the gnn path through embeddings.compute_embedding on 5 stored gnn6
    instances (CPU) and compare with the stored artifacts: embeddings within
    atol 1e-6 (the GPU forward is not bit-reproducible), K-means/medoids on
    the stored embedding reproduce the stored seeds exactly, and the stored
    sparse pickle contains every start/goal anchor.

python scripts/test/check_controls.py --all [--methods ...]
    For every (method, cf, N, generator, case): all start/goal anchors of every
    permutation are nodes of the sparse roadmap; num_clusters vs K
    (shortfalls listed); embedding rows == source vertices; candidates ==
    rows - anchors - dropped. Tallies -> results/controls/checks.json.
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = Path(os.environ.get("CONTROLS_DATA",
                           "/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/test_cluster"))
PARAM_DIR = "map64.0x64.0_resolution1.0"
GNN6_RUN = REPO_ROOT / "logs/cluster/gatv2_compile/wandb/offline-run-20260915_070451-miszrtsa"
GNN6_EPOCH = 100
CFS = (0.1, 0.2, 0.3, 0.4, 0.5)
AGENTS = (4, 8, 16, 32, 64)
RMTS = ("grid", "prm", "cdt", "halton")
NUM_CASES = 25
NUM_PERMS = 4


def cf_root(cf: float) -> Path:
    return DATA / f"test_samples1500_cf{int(round(cf * 10)):02d}"


def agents_dir(cf: float, n: int) -> Path:
    return cf_root(cf) / PARAM_DIR / f"agents{n}_obst0.025" / "radius0.5"


def cluster_dir(cf, n, rmt, case) -> Path:
    return agents_dir(cf, n) / f"case_{case}" / "maps" / rmt / "cluster"


def perm_endpoints(cf, n, case, perm):
    """Start/goal world coordinates of one permutation's input.yaml."""
    with open(agents_dir(cf, n) / f"case_{case}" / "perm" / f"perm_{perm}" / "input.yaml") as f:
        agents = yaml.safe_load(f)["agents"]
    pts = []
    for a in agents:
        pts.append(tuple(float(v) for v in a["start"]))
        pts.append(tuple(float(v) for v in a["goal"]))
    return pts


_FRAME_CACHE = {}


def to_roadmap_frame(rmt: str, pts, inpt_path: Path):
    """World -> roadmap-native frame (grid maps index cells)."""
    if rmt != "grid":
        return [tuple(round(float(v), 9) for v in p) for p in pts]
    from path_planning.common.environment.map.graph_sampler import GraphSampler
    from path_planning.utils.util import agents_yaml_to_roadmap_frame
    key = str(inpt_path)
    if key not in _FRAME_CACHE:
        with open(inpt_path) as f:
            inpt = yaml.safe_load(f)
        _FRAME_CACHE[key] = GraphSampler(bounds=inpt["map"]["bounds"],
                                         resolution=inpt["map"]["resolution"],
                                         start=[], goal=[], use_discrete_space=True)
    m = _FRAME_CACHE[key]
    agents = [{"start": list(p), "goal": list(p)} for p in pts]
    return [tuple(round(float(v), 9) for v in a["start"])
            for a in agents_yaml_to_roadmap_frame(m, agents)]


def pickle_node_set(pkl: Path):
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return {tuple(round(float(v), 9) for v in n.current) for n in data["nodes"]}, len(data["nodes"])


def check_instance(method, cf, n, rmt, case):
    """One (method, cf, N, rmt, case): returns a dict of findings or None if absent."""
    cdir = cluster_dir(cf, n, rmt, case)
    rt_file = cdir / f"graph_map_{method}_runtime.yaml"
    pkl = cdir / f"graph_map_{method}.pkl"
    if not rt_file.exists() or not pkl.exists():
        return None
    with open(rt_file) as f:
        rt = yaml.safe_load(f)
    emb = rt.get("embedding") or {}
    K = int(rt["min_clusters"])
    n_clusters = int(rt["num_clusters"])
    n_nodes = int(rt["num_nodes"])
    src = int(rt["source_num_nodes"])
    n_embedded = emb.get("n_embedded", (rt.get("embedding_shape") or [None])[0])
    centroid_rows = (rt.get("centroid_shape") or [None])[0]
    nodes, n_pkl = pickle_node_set(pkl)
    inpt_path = agents_dir(cf, n) / f"case_{case}" / "input.yaml"
    missing = []
    for perm in range(NUM_PERMS):
        pts = to_roadmap_frame(rmt, perm_endpoints(cf, n, case, perm), inpt_path)
        for p in pts:
            if p not in nodes:
                missing.append((perm, p))
    out = {
        "K": K, "n_clusters": n_clusters, "n_nodes": n_nodes, "n_pkl_nodes": n_pkl,
        "source_nodes": src, "n_embedded": n_embedded,
        "n_anchors": emb.get("n_anchors"), "n_dropped": emb.get("n_dropped"),
        "n_candidates": emb.get("n_candidates"), "n_components": emb.get("n_components"),
        "centroid_rows": centroid_rows,
        "anchors_missing": missing,
        "pkl_matches_yaml": n_pkl == n_nodes,
        "rows_match_source": (n_embedded == src) if n_embedded is not None else None,
        "candidates_consistent": (
            emb["n_candidates"] == n_embedded - emb["n_anchors"] - emb["n_dropped"]
            if emb else None),
        "cluster_shortfall": max(K - n_clusters, 0),
        "cluster_excess": max(n_clusters - K, 0),
        "cache_hit": emb.get("cache_hit"),
    }
    return out


def run_all(methods, out_file: Path):
    tallies = {}
    for method in methods:
        t = {"instances": 0, "anchor_failures": [], "pkl_yaml_mismatch": [],
             "rows_mismatch": [], "candidates_inconsistent": [],
             "cluster_shortfall": [], "cluster_excess": [],
             "dropped_by_generator": defaultdict(list),
             "components_by_generator": defaultdict(list),
             "waypoint_surplus_ratio": [],
             "cells_present": 0}
        for cf in CFS:
            for n in AGENTS:
                for rmt in RMTS:
                    any_present = False
                    for case in range(NUM_CASES):
                        r = check_instance(method, cf, n, rmt, case)
                        if r is None:
                            continue
                        any_present = True
                        t["instances"] += 1
                        key = f"cf{cf}/N{n}/{rmt}/case_{case}"
                        if r["anchors_missing"]:
                            t["anchor_failures"].append((key, r["anchors_missing"][:4]))
                        if not r["pkl_matches_yaml"]:
                            t["pkl_yaml_mismatch"].append(key)
                        if r["rows_match_source"] is False:
                            t["rows_mismatch"].append(key)
                        if r["candidates_consistent"] is False:
                            t["candidates_inconsistent"].append(key)
                        if r["cluster_shortfall"]:
                            t["cluster_shortfall"].append((key, r["K"], r["n_clusters"]))
                        if r["cluster_excess"]:
                            t["cluster_excess"].append((key, r["K"], r["n_clusters"]))
                        if r["n_dropped"] is not None:
                            t["dropped_by_generator"][rmt].append(int(r["n_dropped"]))
                        if r["n_components"] is not None:
                            t["components_by_generator"][rmt].append(int(r["n_components"]))
                        t["waypoint_surplus_ratio"].append(r["n_nodes"] / max(r["n_clusters"], 1))
                    t["cells_present"] += int(any_present)
        summary = {
            "instances": t["instances"],
            "cells_present": t["cells_present"],
            "anchor_failures": t["anchor_failures"],
            "pkl_yaml_mismatch": t["pkl_yaml_mismatch"],
            "rows_mismatch": t["rows_mismatch"],
            "candidates_inconsistent": t["candidates_inconsistent"],
            "n_cluster_shortfall": len(t["cluster_shortfall"]),
            "cluster_shortfall_examples": t["cluster_shortfall"][:10],
            "max_cluster_shortfall": max((k - c for _, k, c in t["cluster_shortfall"]), default=0),
            "n_cluster_excess": len(t["cluster_excess"]),
            "max_cluster_excess": max((c - k for _, k, c in t["cluster_excess"]), default=0),
            "dropped_by_generator": {g: {"mean": float(np.mean(v)), "max": int(max(v)),
                                         "min": int(min(v)), "n": len(v)}
                                     for g, v in t["dropped_by_generator"].items()},
            "components_by_generator": {g: {"mean": float(np.mean(v)), "max": int(max(v)),
                                            "min": int(min(v))}
                                        for g, v in t["components_by_generator"].items()},
            "sparse_nodes_over_clusters": {
                "mean": float(np.mean(t["waypoint_surplus_ratio"])) if t["waypoint_surplus_ratio"] else None,
                "max": float(np.max(t["waypoint_surplus_ratio"])) if t["waypoint_surplus_ratio"] else None},
        }
        tallies[method] = summary
        print(f"[{method}] instances={summary['instances']} cells={summary['cells_present']} "
              f"anchor_failures={len(summary['anchor_failures'])} "
              f"pkl/yaml mismatch={len(summary['pkl_yaml_mismatch'])} "
              f"rows_mismatch={len(summary['rows_mismatch'])} "
              f"candidates_inconsistent={len(summary['candidates_inconsistent'])} "
              f"clusters<K: {summary['n_cluster_shortfall']} (max {summary['max_cluster_shortfall']}) "
              f"clusters>K: {summary['n_cluster_excess']} (max {summary['max_cluster_excess']}) "
              f"nodes/clusters={summary['sparse_nodes_over_clusters']['mean']}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(tallies, f, indent=1, default=str)
    print(f"wrote {out_file}")
    bad = [m for m, s in tallies.items()
           if s["anchor_failures"] or s["pkl_yaml_mismatch"] or s["rows_mismatch"]
           or s["candidates_inconsistent"]]
    return 1 if bad else 0


def run_gnn_repro(cf=0.1, n=16, rmt="prm", cases=range(5), method="gnn6"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from path_planning.cluster.embeddings import compute_embedding
    from path_planning.cluster.gnn_cluster_map import (
        _kmeans_medoids, ablation_flags_from_config, gnn_seed_indices, load_cluster_encoder,
        read_run_config, sampler_to_cluster_heterodata)
    from path_planning.data_generation.dataset_ground_truth_map import create_map
    from path_planning.data_generation.dataset_util import (
        generate_base_case_path, get_graph_file_path, get_input_file_path)
    device = torch.device("cpu")
    flags = ablation_flags_from_config(read_run_config(GNN6_RUN))
    model = None
    ok_all = True
    print(f"gnn6 reproduction through compute_embedding (CPU) on cf{cf} N{n} {rmt} cases {list(cases)}")
    for case in cases:
        case_path, map_path = generate_base_case_path(agents_dir(cf, n), case, rmt)
        with open(get_input_file_path(case_path)) as f:
            inpt = yaml.safe_load(f)
        m = create_map(inpt, graph_file=get_graph_file_path(map_path), verbose=False,
                       args={"use_constraint_sweep": False})
        data = sampler_to_cluster_heterodata(m, *flags)
        if model is None:
            model, _ = load_cluster_encoder(GNN6_RUN, data, epoch=GNN6_EPOCH, device=device)
        z_new, info = compute_embedding(m, "gnn", model=model, data=data, device=device)
        cdir = cluster_dir(cf, n, rmt, case)
        z_st = np.load(cdir / f"graph_map_{method}_embedding.npy")
        s_st = np.load(cdir / f"graph_map_{method}_seeds.npy")
        c_st = np.load(cdir / f"graph_map_{method}_centroids.npy")
        with open(cdir / f"graph_map_{method}_runtime.yaml") as f:
            rt = yaml.safe_load(f)
        K = int(rt["min_clusters"])
        sg = set(m.start_nodes_index.values()) | set(m.goal_nodes_index.values())
        free = np.array([i for i in range(len(z_st)) if i not in sg], dtype=np.int64)
        k_free = min(max(K - len(sg), 1), len(free) - 1)
        s_re, c_re = _kmeans_medoids(z_st, free, k_free)
        seeds_new, _, _ = gnn_seed_indices(model, data, m, K, device)
        nodes, _ = pickle_node_set(cdir / f"graph_map_{method}.pkl")
        inpt_path = case_path / "input.yaml"
        anchors_ok = all(p in nodes for perm in range(NUM_PERMS)
                         for p in to_roadmap_frame(rmt, perm_endpoints(cf, n, case, perm), inpt_path))
        maxabs = float(np.abs(z_st - z_new).max())
        emb_ok = np.allclose(z_st, z_new, atol=1e-6)
        seeds_ok = np.array_equal(np.asarray(s_re), s_st) and np.allclose(c_re, c_st, atol=1e-5)
        jac = len(set(seeds_new) & set(s_st.tolist())) / len(set(seeds_new) | set(s_st.tolist()))
        ok = emb_ok and seeds_ok and anchors_ok
        ok_all &= ok
        print(f"  case_{case}: K={K} embedding max|d|={maxabs:.2e} allclose={emb_ok} "
              f"kmeans(stored z)==stored seeds: {seeds_ok} "
              f"refactored-path seed Jaccard vs stored={jac:.3f} anchors in pkl: {anchors_ok} -> {'OK' if ok else 'FAIL'}")
    print("gnn repro:", "OK" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-repro", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--methods", nargs="+",
                    default=["gnn6", "euclid", "isomap_32", "spectral_32"])
    ap.add_argument("--out", default=str(REPO_ROOT / "results/controls/checks.json"))
    args = ap.parse_args()
    rc = 0
    if args.gnn_repro:
        rc |= run_gnn_repro()
    if args.all:
        rc |= run_all(args.methods, Path(args.out))
    if not (args.gnn_repro or args.all):
        ap.print_help()
    sys.exit(rc)
