"""Collect the unlearned-embedding control results into results/controls/.

python scripts/test/build_controls_results.py [--methods gnn6 euclid isomap_32 spectral_32]
        [--sweep-method isomap_32 --sweep-agents 64] [--diag-cells 0.1:64 0.3:64 ...]

Outputs (under --out, default results/controls):
  <method>_<d>.csv           one row per (generator, N, cf, case, perm)
  baseline.csv               dense-roadmap solutions (no clustering)
  summary.md                 sweep table, gnn6-vs-controls comparison, diagnostics, deviations
  isomap_explained_variance.png
  diagnostics.json, tables.json
"""
import argparse
import csv
import json
import math
import os
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
CFS = (0.1, 0.2, 0.3, 0.4, 0.5)
AGENTS = (4, 8, 16, 32, 64)
RMTS = ("grid", "prm", "cdt", "halton")
POOL_RMTS = ("prm", "cdt", "halton")   # the three generators pooled in the comparison
NUM_CASES = 25
NUM_PERMS = 4
GNN6_RUN = REPO_ROOT / "logs/cluster/gatv2_compile/wandb/offline-run-20260915_070451-miszrtsa"
GNN6_EPOCH = 100
BOOT = 2000
BOOT_SEED = 0
CSV_COLUMNS = ["generator", "N", "cf", "case", "perm", "success", "flowtime", "planner_time",
               "embed_time", "cluster_time", "reconstruct_time", "n_vertices_sparse",
               "n_free_representatives", "K", "n_clusters", "n_anchors", "n_dropped",
               "n_components", "explained_variance", "negative_mass", "embed_cache_hit",
               "build_time_total", "n_vertices_source", "n_edges_sparse"]


# ---------------------------------------------------------------------------
# paths / loading
# ---------------------------------------------------------------------------

def cf_root(cf):
    return DATA / f"test_samples1500_cf{int(round(cf * 10)):02d}"


def agents_dir(cf, n):
    return cf_root(cf) / PARAM_DIR / f"agents{n}_obst0.025" / "radius0.5"


def csv_name(method):
    if method == "gnn6" or method.startswith("gnn"):
        return f"{method}_32.csv"
    if method == "euclid":
        return "euclid_2.csv"
    return f"{method}.csv"


def read_solution(path):
    """Only the top-level scalar keys (a line scan; the indented schedule
    block, which precedes `success:` alphabetically, is skipped)."""
    out = {}
    with open(path) as f:
        for line in f:
            if line[:1] in (" ", "-", "\n") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            if v:
                out[k.strip()] = v
    return out


def load_rows(method):
    """All (generator, N, cf, case, perm) rows for one method (None = baseline)."""
    rows = []
    for cf in CFS:
        for n in AGENTS:
            d = agents_dir(cf, n)
            for rmt in RMTS:
                for case in range(NUM_CASES):
                    cdir = d / f"case_{case}" / "maps" / rmt / "cluster"
                    rt = None
                    if method is not None:
                        rt_file = cdir / f"graph_map_{method}_runtime.yaml"
                        if not rt_file.exists():
                            continue
                        with open(rt_file) as f:
                            rt = yaml.safe_load(f)
                    for perm in range(NUM_PERMS):
                        suffix = "solution_graph_map" if method is None else f"solution_graph_map_{method}"
                        sol = d / f"case_{case}" / "perm" / f"perm_{perm}" / "sipp" / rmt / f"{suffix}_velocity1.0.yaml"
                        if not sol.exists():
                            continue
                        s = read_solution(sol)
                        row = {"generator": rmt, "N": n, "cf": cf, "case": case, "perm": perm,
                               "success": int(s.get("success", "false").lower() == "true"),
                               "flowtime": float(s["flowtime"]) if "flowtime" in s else float("nan"),
                               "planner_time": float(s["runtime"]) if "runtime" in s else float("nan"),
                               "n_vertices_sparse": int(s["num_nodes"]) if "num_nodes" in s else None,
                               "n_edges_sparse": int(s["num_edges"]) if "num_edges" in s else None}
                        if rt is not None:
                            b = rt.get("runtime_breakdown", {})
                            e = rt.get("embedding") or {}
                            row.update({
                                "embed_time": float(b.get("heterodata", 0.0)) + float(b.get("inference", 0.0)),
                                "cluster_time": float(b.get("kmeans", 0.0)),
                                "reconstruct_time": float(b.get("cluster", 0.0)) + float(b.get("distill", 0.0)) + float(b.get("build", 0.0)),
                                "n_vertices_sparse": int(rt["num_nodes"]),
                                "n_edges_sparse": int(rt["num_edges"]),
                                "n_free_representatives": (rt.get("centroid_shape") or [None])[0],
                                "K": int(rt["min_clusters"]),
                                "n_clusters": int(rt["num_clusters"]),
                                "n_anchors": e.get("n_anchors"),
                                "n_dropped": e.get("n_dropped"),
                                "n_components": e.get("n_components"),
                                "explained_variance": e.get("explained_variance"),
                                "negative_mass": e.get("negative_mass"),
                                "embed_cache_hit": e.get("cache_hit"),
                                "build_time_total": float(rt.get("runtime", float("nan"))),
                                "n_vertices_source": int(rt["source_num_nodes"]),
                            })
                        else:
                            row.update({"embed_time": 0.0, "cluster_time": 0.0, "reconstruct_time": 0.0,
                                        "build_time_total": 0.0})
                        rows.append(row)
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, centre - half, centre + half)


def case_bootstrap(values_by_case, stat=np.mean, n_boot=BOOT, seed=BOOT_SEED):
    """Cluster bootstrap: resample cases with replacement, pool their values.
    values_by_case: {case: [v, ...]}. Returns (point, lo, hi)."""
    cases = sorted(values_by_case)
    if not cases:
        return (float("nan"), float("nan"), float("nan"))
    pooled = np.concatenate([np.asarray(values_by_case[c], dtype=float) for c in cases])
    point = float(stat(pooled))
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(values_by_case[c], dtype=float) for c in cases]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(cases), len(cases))
        stats[b] = stat(np.concatenate([arrays[i] for i in pick]))
    return (point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def fmt_ci(t, scale=1.0, nd=1):
    p, lo, hi = t
    if any(map(lambda v: v != v, (p, lo, hi))):
        return "—"
    return f"{p*scale:.{nd}f} [{lo*scale:.{nd}f}, {hi*scale:.{nd}f}]"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def index_rows(rows):
    """{(generator, N, cf, case, perm): row}"""
    return {(r["generator"], r["N"], r["cf"], r["case"], r["perm"]): r for r in rows}


def success_cell(rows, cf, n, rmts):
    sel = [r for r in rows if r["cf"] == cf and r["N"] == n and r["generator"] in rmts]
    k = sum(r["success"] for r in sel)
    by_case = defaultdict(list)
    for r in sel:
        by_case[r["case"]].append(r["success"])
    return {"n": len(sel), "k": k, "wilson": wilson_ci(k, len(sel)),
            "boot": case_bootstrap(by_case) if sel else (float("nan"),) * 3,
            "n_cases": len(by_case)}


def sweep_table(by_method, sweep_methods, agents, cfs, rmt="prm"):
    lines = [f"| method | N | cf | success % (Wilson 95%) | n | explained variance mean ± std (25 roadmaps) |",
             "|---|---|---|---|---|---|"]
    for m in sweep_methods:
        rows = by_method.get(m, [])
        for n in agents:
            for cf in cfs:
                c = success_cell(rows, cf, n, (rmt,))
                if c["n"] == 0:
                    continue
                ev = {r["case"]: r["explained_variance"] for r in rows
                      if r["cf"] == cf and r["N"] == n and r["generator"] == rmt
                      and r.get("explained_variance") is not None}
                evs = np.array([v for v in ev.values() if v == v], dtype=float)
                ev_s = f"{evs.mean():.3f} ± {evs.std():.3f}" if evs.size else "—"
                lines.append(f"| {m} | {n} | {cf} | {fmt_ci(c['wilson'], 100)} | {c['n']} | {ev_s} |")
    return "\n".join(lines)


def comparison_tables(by_method, methods, baseline_rows, rmts, label):
    """Success (Wilson + case bootstrap), flowtime increase and total runtime
    (case bootstrap) on the common solved set, per (N, cf)."""
    base = index_rows(baseline_rows)
    idx = {m: index_rows(by_method.get(m, [])) for m in methods}
    out = [f"### {label}", "",
           "| N | cf | generators | method | success % (Wilson) | success % (case boot) | flowtime +% (case boot) | total runtime s (case boot) | n | n_common | n_cases |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    records = []
    for n in AGENTS:
        for cf in CFS:
            present = [m for m in methods if any(k[1] == n and k[2] == cf and k[0] in rmts for k in idx[m])]
            if not present or present == ["gnn6"]:
                continue   # nothing to compare in this cell yet
            # pool only the generators every compared method has in this cell
            gens = tuple(g for g in rmts
                         if all(any(k[0] == g and k[1] == n and k[2] == cf for k in idx[m]) for m in present))
            if not gens:
                continue
            keys = [k for k in base if k[1] == n and k[2] == cf and k[0] in gens]
            common = [k for k in keys if base[k]["success"]
                      and all(k in idx[m] and idx[m][k]["success"] for m in present)]
            n_cases = len({k[3] for k in common})
            for m in present:
                c = success_cell(by_method[m], cf, n, gens)
                fl_by_case, rt_by_case = defaultdict(list), defaultdict(list)
                for k in common:
                    r = idx[m][k]
                    fl_by_case[k[3]].append(100.0 * (r["flowtime"] / base[k]["flowtime"] - 1.0))
                    rt_by_case[k[3]].append(r["embed_time"] + r["cluster_time"] + r["reconstruct_time"] + r["planner_time"])
                fl = case_bootstrap(fl_by_case) if common else (float("nan"),) * 3
                rt = case_bootstrap(rt_by_case) if common else (float("nan"),) * 3
                out.append(f"| {n} | {cf} | {'+'.join(gens)} | {m} | {fmt_ci(c['wilson'], 100)} | {fmt_ci(c['boot'], 100)} | "
                           f"{fmt_ci(fl, 1, 2)} | {fmt_ci(rt, 1, 2)} | {c['n']} | {len(common)} | {n_cases} |")
                records.append({"N": n, "cf": cf, "generators": gens, "method": m, "success": c["wilson"], "success_boot": c["boot"],
                                "n": c["n"], "flowtime_increase": fl, "total_runtime": rt,
                                "n_common": len(common), "n_cases": n_cases})
    return "\n".join(out), records


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def _load_source_map(cf, n, rmt, case):
    from path_planning.data_generation.dataset_ground_truth_map import create_map
    from path_planning.data_generation.dataset_util import (
        generate_base_case_path, get_graph_file_path, get_input_file_path)
    case_path, map_path = generate_base_case_path(agents_dir(cf, n), case, rmt)
    with open(get_input_file_path(case_path)) as f:
        inpt = yaml.safe_load(f)
    m = create_map(inpt, graph_file=get_graph_file_path(map_path), verbose=False,
                   args={"use_constraint_sweep": False})
    return m


def diagnostics(methods, cells, rmt="prm", n_sources=200, n_targets=100, noise_floor=True):
    """Spearman of pairwise embedding distances vs gnn6 (and vs geodesic), Jaccard
    of representative sets vs gnn6, on 20k = 200x100 vertex pairs per roadmap
    seeded by case id; plus gnn6's own re-embed noise floor (CPU forward)."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.stats import spearmanr
    from path_planning.cluster.embeddings import symmetric_roadmap_csr
    from path_planning.cluster.gnn_cluster_map import _kmeans_medoids

    model = None
    torch = None
    res = {}
    for cf, n in cells:
        d = agents_dir(cf, n)
        per = defaultdict(list)
        for case in range(NUM_CASES):
            cdir = d / f"case_{case}" / "maps" / rmt / "cluster"
            g_emb = cdir / "graph_map_gnn6_embedding.npy"
            if not g_emb.exists():
                continue
            z_g = np.load(g_emb).astype(np.float64)
            s_g = set(np.load(cdir / "graph_map_gnn6_seeds.npy").tolist())
            embs = {}
            seeds = {}
            for m in methods:
                f = cdir / f"graph_map_{m}_embedding.npy"
                if f.exists():
                    embs[m] = np.load(f).astype(np.float64)
                    seeds[m] = set(np.load(cdir / f"graph_map_{m}_seeds.npy").tolist())
            if not embs:
                continue
            src_map = _load_source_map(cf, n, rmt, case)
            sg = set(src_map.start_nodes_index.values()) | set(src_map.goal_nodes_index.values())
            valid = np.ones(len(z_g), dtype=bool)
            for z in embs.values():
                valid &= ~np.isnan(z[:, 0])
            for i in sg:
                valid[i] = False
            cand = np.flatnonzero(valid)
            rng = np.random.default_rng(case)
            sources = rng.choice(cand, size=min(n_sources, len(cand)), replace=False)
            targets = rng.choice(cand, size=(len(sources), n_targets), replace=True)
            csr, _ = symmetric_roadmap_csr(src_map)
            geo_rows = shortest_path(csr, method="D", directed=False, indices=sources)
            geo = geo_rows[np.arange(len(sources))[:, None], targets].ravel()
            keep = np.isfinite(geo) & (targets != sources[:, None]).ravel()

            def pair_d(z):
                return np.linalg.norm(z[sources][:, None, :] - z[targets], axis=-1).ravel()[keep]

            dg = pair_d(z_g)
            geo = geo[keep]
            per["gnn6_vs_geodesic"].append(spearmanr(dg, geo).correlation)
            for m, z in embs.items():
                dm = pair_d(z)
                per[f"{m}_vs_gnn6_spearman"].append(spearmanr(dm, dg).correlation)
                per[f"{m}_vs_geodesic_spearman"].append(spearmanr(dm, geo).correlation)
                per[f"{m}_vs_gnn6_jaccard"].append(len(seeds[m] & s_g) / len(seeds[m] | s_g))
                per[f"{m}_K"].append(len(seeds[m]))
            if "isomap_32" in embs and "euclid" in embs:
                per["isomap_32_vs_euclid_spearman"].append(
                    spearmanr(pair_d(embs["isomap_32"]), pair_d(embs["euclid"])).correlation)
                per["isomap_32_vs_euclid_jaccard"].append(
                    len(seeds["isomap_32"] & seeds["euclid"]) / len(seeds["isomap_32"] | seeds["euclid"]))
            if noise_floor and (GNN6_RUN / "files/model" / f"epoch_{GNN6_EPOCH}.pth").exists():
                if torch is None:
                    import torch as _torch
                    torch = _torch
                    from path_planning.cluster.gnn_cluster_map import (
                        ablation_flags_from_config, load_cluster_encoder, read_run_config,
                        sampler_to_cluster_heterodata)
                    flags = ablation_flags_from_config(read_run_config(GNN6_RUN))
                data = sampler_to_cluster_heterodata(src_map, *flags)
                if model is None:
                    model, _ = load_cluster_encoder(GNN6_RUN, data, epoch=GNN6_EPOCH,
                                                    device=torch.device("cpu"))
                with torch.no_grad():
                    ds = data.to(torch.device("cpu"))
                    z_cpu = model(dict(ds.x_dict), ds.edge_index_dict, ds.edge_attr_dict)['node'].cpu().numpy()
                free = np.array([i for i in range(len(z_cpu)) if i not in sg], dtype=np.int64)
                with open(cdir / "graph_map_gnn6_runtime.yaml") as f:
                    K = int(yaml.safe_load(f)["min_clusters"])
                k_free = min(max(K - len(sg), 1), len(free) - 1)
                s_cpu, _ = _kmeans_medoids(z_cpu, free, k_free)
                s_cpu = set(s_cpu)
                per["gnn6_noise_floor_jaccard"].append(len(s_cpu & s_g) / len(s_cpu | s_g))
                per["gnn6_noise_floor_spearman"].append(spearmanr(pair_d(z_cpu.astype(np.float64)), dg).correlation)
                per["gnn6_noise_floor_maxabs"].append(float(np.abs(z_cpu - z_g).max()))
        res[f"cf{cf}_N{n}_{rmt}"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                                     for k, v in per.items()}
        print(f"diagnostics cf{cf} N{n} {rmt}: {len(per.get('gnn6_vs_geodesic', []))} roadmaps")
    return res


def diagnostics_table(diag, methods):
    lines = ["| cell | quantity | " + " | ".join(methods) + " | gnn6 re-embed floor |",
             "|---|---|" + "---|" * len(methods) + "---|"]
    for cell, d in diag.items():
        def g(key):
            v = d.get(key)
            return f"{v['mean']:.3f} ± {v['std']:.3f}" if v else "—"
        lines.append(f"| {cell} | Spearman ρ vs gnn6 (pair dist.) | " +
                     " | ".join(g(f"{m}_vs_gnn6_spearman") for m in methods) +
                     f" | {g('gnn6_noise_floor_spearman')} |")
        lines.append(f"| {cell} | Spearman ρ vs geodesic | " +
                     " | ".join(g(f"{m}_vs_geodesic_spearman") for m in methods) +
                     f" | {g('gnn6_vs_geodesic')} (gnn6) |")
        lines.append(f"| {cell} | Jaccard of representatives vs gnn6 | " +
                     " | ".join(g(f"{m}_vs_gnn6_jaccard") for m in methods) +
                     f" | {g('gnn6_noise_floor_jaccard')} |")
        lines.append(f"| {cell} | K | " + " | ".join(g(f"{m}_K") for m in methods) + " | |")
        if "isomap_32_vs_euclid_spearman" in d:
            lines.append(f"| {cell} | isomap_32 vs euclid: Spearman / Jaccard | "
                         f"{g('isomap_32_vs_euclid_spearman')} / {g('isomap_32_vs_euclid_jaccard')}" +
                         " | " * (len(methods)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# explained-variance plot
# ---------------------------------------------------------------------------

def explained_variance_plot(out_png, method="isomap_32", cf=0.1, n=64, rmt="prm", d_max=64,
                            marks=(2, 4, 8, 16, 32)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = []
    for case in range(NUM_CASES):
        f = agents_dir(cf, n) / f"case_{case}" / "maps" / rmt / "cluster" / f"graph_map_{method}_spectrum.npy"
        if not f.exists():
            continue
        spec = np.load(f)
        pos = spec[spec > 0]
        cum = np.cumsum(pos)[:d_max] / pos.sum()
        if len(cum) < d_max:
            cum = np.pad(cum, (0, d_max - len(cum)), constant_values=1.0)
        curves.append(cum)
    if not curves:
        print("explained-variance plot skipped: no spectrum files")
        return None
    arr = np.array(curves)
    mean, std = arr.mean(0), arr.std(0)
    ds = np.arange(1, d_max + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.fill_between(ds, mean - std, mean + std, color="#2f6fed", alpha=0.15, lw=0)
    ax.plot(ds, mean, color="#2f6fed", lw=2)
    mk = [d for d in marks if d <= d_max]
    ax.plot(mk, mean[np.array(mk) - 1], "o", color="#2f6fed", ms=6, mec="white", mew=1.5)
    for d in mk:
        ax.annotate(f"{mean[d-1]:.3f}", (d, mean[d-1]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#374151")
    ax.set_xscale("log", base=2)
    ax.set_xticks(mk + ([64] if d_max >= 64 else []))
    ax.set_xticklabels([str(v) for v in ax.get_xticks()])
    ax.set_xlabel("isomap dimension d")
    ax.set_ylabel("explained variance (positive MDS spectrum)")
    ax.set_title(f"Isomap explained variance vs d ({rmt}, N={n}, cf={cf}, {len(curves)} roadmaps, mean ± std)",
                 fontsize=9)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return {"d": ds.tolist(), "mean": mean.tolist(), "std": std.tolist(), "n_roadmaps": len(curves)}


# ---------------------------------------------------------------------------
# deviations paragraph
# ---------------------------------------------------------------------------

def deviations_text(by_method, checks, methods):
    parts = []
    comp = {}
    for m in methods:
        rows = by_method.get(m, [])
        per_gen = defaultdict(list)
        for r in rows:
            if r.get("n_dropped") is not None and r["perm"] == 0:
                per_gen[r["generator"]].append((r["n_dropped"], r.get("n_components") or 0, r["n_vertices_source"]))
        if per_gen:
            comp[m] = {g: (np.mean([a for a, _, _ in v]), np.max([a for a, _, _ in v]),
                           np.mean([b for _, b, _ in v]), np.mean([c for _, _, c in v]))
                       for g, v in per_gen.items()}
    if comp:
        parts.append("**Disconnected roadmaps.** The source roadmaps are not connected: the boundary-node "
                     "resampling leaves isolated (degree-0) boundary vertices, so every generator has "
                     "many components. Per generator, mean/max vertices outside the largest component "
                     "(dropped from the isomap/spectral candidate set, option (a)), mean component count, "
                     "and mean source vertex count: " +
                     "; ".join(f"{m}: " + ", ".join(f"{g} {a:.0f}/{b:.0f} dropped, {c:.0f} comps, |V|={d:.0f}"
                                                    for g, (a, b, c, d) in v.items())
                               for m, v in comp.items() if any(a > 0 for a, _, _, _ in v.values())) +
                     ". gnn6 and euclid keep those vertices as candidates (the gnn path is unchanged), so the "
                     "candidate sets differ by that count; no start/goal was outside the component "
                     "(an instance with one raises and would be listed under failures).")
    if checks:
        for m, c in checks.items():
            bits = []
            if c["anchor_failures"]:
                bits.append(f"{len(c['anchor_failures'])} instances with a missing anchor")
            if c["n_cluster_shortfall"]:
                bits.append(f"{c['n_cluster_shortfall']} instances with num_clusters < K (max shortfall {c['max_cluster_shortfall']})")
            if c["n_cluster_excess"]:
                bits.append(f"{c['n_cluster_excess']} instances with num_clusters > K (max excess {c['max_cluster_excess']})")
            r = c.get("sparse_nodes_over_clusters", {}).get("mean")
            parts.append(f"**{m} checks** ({c['instances']} instances): " +
                         ("; ".join(bits) if bits else "all anchors present, num_clusters == K everywhere") +
                         (f"; sparse vertex count / cluster count = {r:.2f} on average" if r else "") + ".")
    parts.append("**Vertex sets.** Every method embeds all source roadmap vertices (samples, registered "
                 "boundary nodes, start/goal) — the same set the GNN embeds — and clusters that set minus the "
                 "start/goal anchors (minus the disconnected vertices for isomap/spectral). The sparse roadmap "
                 "contains the K cluster representatives plus the interior waypoints of the CTopPRM cluster "
                 "tours, so `n_vertices_sparse` > K by design for every method, including gnn6.")
    parts.append("**Timing.** `embed_time` is the encoder forward on the GPU for gnn6 (plus HeteroData "
                 "assembly) and CPU time for the controls (all-pairs Dijkstra + eigensolve for isomap; "
                 "sparse eigensolve for spectral; ~0 for euclid). Isomap/spectral solves are content-cached "
                 "across cluster-fraction roots; a cache hit reports the stored compute time.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["gnn6", "euclid", "isomap_32", "spectral_32"],
                    help="methods for the comparison table (gnn6 first)")
    ap.add_argument("--sweep-methods", nargs="+", default=None,
                    help="isomap_<d> methods for the sweep table (default: every isomap_* found among --methods and on disk)")
    ap.add_argument("--sweep-agents", nargs="+", type=int, default=[64])
    ap.add_argument("--sweep-cfs", nargs="+", type=float, default=list(CFS))
    ap.add_argument("--diag-cells", nargs="+", default=["0.1:64", "0.3:64", "0.1:16", "0.3:16"])
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "results/controls"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    methods = list(args.methods)
    # any isomap_<d> present on disk joins the sweep table
    found = set()
    for cf in CFS:
        for n in AGENTS:
            cd = agents_dir(cf, n) / "case_0" / "maps" / "prm" / "cluster"
            if cd.exists():
                found |= {p.stem[len("graph_map_"):-len("_runtime")] for p in cd.glob("graph_map_isomap_*_runtime.yaml")}
    sweep_methods = args.sweep_methods or sorted(found, key=lambda s: int(s.split("_")[1]))
    all_methods = list(dict.fromkeys(methods + sweep_methods))

    by_method = {}
    for m in all_methods:
        rows = load_rows(m)
        by_method[m] = rows
        write_csv(rows, out / csv_name(m))
        print(f"{m}: {len(rows)} rows -> {csv_name(m)}")
    baseline = load_rows(None)
    write_csv(baseline, out / "baseline.csv")
    print(f"baseline: {len(baseline)} rows")

    controls = [m for m in methods if m != "gnn6"]
    checks_file = out / "checks.json"
    checks = json.load(open(checks_file)) if checks_file.exists() else {}

    md = ["# Unlearned embedding controls — summary", "",
          f"Data root: `{DATA}`. Instances: 25 cases × 4 permutations per (generator, N, cf). "
          f"Planner SIPP, 60 s limit. Methods: {', '.join(all_methods)}. "
          "Success CIs: Wilson (instances treated as independent) and a case-level cluster bootstrap "
          f"({BOOT} resamples, seed {BOOT_SEED}; permutations and generators of a case are resampled together). "
          "Flowtime increase = mean per-instance % increase over the dense-roadmap SIPP solution on the "
          "common solved set (dense baseline ∩ every listed method); total runtime = embed + cluster + "
          "reconstruct + planner time on the same set.", ""]
    md += ["## 1. Isomap sweep (prm)", "",
           sweep_table(by_method, sweep_methods, args.sweep_agents, args.sweep_cfs), ""]
    t_pool, rec_pool = comparison_tables(by_method, methods, baseline, POOL_RMTS,
                                         "gnn6 vs controls — pooled over prm + cdt + halton")
    t_grid, rec_grid = comparison_tables(by_method, methods, baseline, ("grid",), "grid only")
    md += ["## 2. Comparison at every (N, cf)", "", t_pool, "", t_grid, ""]

    diag = {}
    if not args.no_diagnostics:
        cells = [(float(c.split(":")[0]), int(c.split(":")[1])) for c in args.diag_cells]
        diag = diagnostics(controls, cells)
        with open(out / "diagnostics.json", "w") as f:
            json.dump(diag, f, indent=1)
        md += ["## 3. Diagnostics vs gnn6 (prm; 20,000 sampled vertex pairs per roadmap, seed = case id)", "",
               diagnostics_table(diag, controls), "",
               "The gnn6 re-embed floor is the same quantity between the stored (GPU) gnn6 embedding and a "
               "fresh CPU forward of the same checkpoint (max |Δ| ≈ 4e-7): differences smaller than it are "
               "K-means seed churn, not embedding differences.", ""]
    ev = explained_variance_plot(out / "isomap_explained_variance.png",
                                 cf=args.sweep_cfs[0], n=args.sweep_agents[0])
    if ev:
        md += ["## Explained variance vs d", "", "![explained variance](isomap_explained_variance.png)", "",
               "| d | " + " | ".join(str(d) for d in (2, 4, 8, 16, 32, 64) if d <= len(ev["d"])) + " |",
               "|---|" + "---|" * len([d for d in (2, 4, 8, 16, 32, 64) if d <= len(ev["d"])]),
               "| mean EV | " + " | ".join(f"{ev['mean'][d-1]:.3f}" for d in (2, 4, 8, 16, 32, 64) if d <= len(ev["d"])) + " |", ""]
    md += ["## 4. Deviations from the plan", "", deviations_text(by_method, checks, all_methods), ""]
    with open(out / "summary.md", "w") as f:
        f.write("\n".join(md))
    with open(out / "tables.json", "w") as f:
        json.dump({"pooled": rec_pool, "grid": rec_grid, "explained_variance": ev}, f, indent=1, default=float)
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
