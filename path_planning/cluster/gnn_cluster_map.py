"""GNN-based roadmap distillation (cluster.md milestone 2, Method A).

Pipeline per case: trained encoder -> node embeddings -> sklearn K-means over
the non-start/goal embeddings -> embedding-space medoid nodes -> those medoids
become the seed list of CTopPRM(clustering='custom'), which reuses the exact
wavefront / connection / tour / shortening machinery of the ctopprm/kmeans/em
baselines to reconstruct and validate the distilled roadmap.

Kept separate from cluster_map.py because its Pool-based worker plumbing
cannot carry a torch model; GNN inference runs sequentially in-process
(seconds per case). File naming mirrors the classic methods:
``maps/<rmt>/cluster/graph_map_gnn.pkl`` (+ ``_runtime.yaml`` sidecar), so
run_all_cluster_solvers-style phase-2 loops and the notebooks pick the results
up as method name ``gnn`` with no changes.
"""
import math
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch_geometric.nn import to_hetero

from path_planning.cluster.cluster_map import (
    _build_setup_sampler,
    _snap_points,
    save_cluster_map,
)
from path_planning.cluster.CTopPRMpy.ctopprm import CTopPRM
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.data_generation.cluster_dataset_generate import (
    transform_graph_map_to_gnn,
)
from path_planning.data_generation.dataset_ground_truth_map import create_map
from path_planning.data_generation.dataset_util import (
    generate_base_case_path,
    generate_cluster_path,
    get_graph_file_path,
    get_graph_runtime_file_path,
    get_input_file_path,
    write_runtime_yaml,
)
from path_planning.gnn.dataloader_cluster import (
    HeteroData,
    _min_max_normalize_columns,
    get_dummy_sample_data,
)
from path_planning.gnn.model import get_model
from path_planning.utils.util import agents_yaml_to_roadmap_frame, set_global_seed

GNN_METHOD = "gnn"
GNN_GRAPH_NAME = "graph_map_gnn.pkl"


def _unwrap_wandb_config(config):
    """Unwrap wandb config nesting: newer wandb writes {'desc': ..., 'value': x}
    per key, older writes {'value': x} (dataset_prune._flatten_wandb_config
    handles only the latter)."""
    if isinstance(config, dict):
        keys = set(config.keys())
        if keys == {"value"} or keys == {"desc", "value"}:
            return _unwrap_wandb_config(config["value"])
        return {k: _unwrap_wandb_config(v) for k, v in config.items()}
    if isinstance(config, list):
        return [_unwrap_wandb_config(v) for v in config]
    return config


def sampler_to_cluster_heterodata(map_: GraphSampler) -> HeteroData:
    """Live GraphSampler -> the cluster-training HeteroData layout.

    Composes transform_graph_map_to_gnn (5-wide one-hot + boundary self-loop
    arrays) with the _load_single_graph normalization convention the encoder
    was trained on: positions divided by max(bounds) with NO origin shift,
    per-relation min-max normalized edge_attr, AddSelfLoops on to/approx
    first, boundary relation attached after. Maps without registered boundary
    nodes yield an all-zero BOUNDARY column and an empty boundary relation
    (their sample nodes were FREE in training too).
    """
    import torch_geometric
    from torch_geometric.transforms import AddSelfLoops

    (ndata, e_idx, e_w, sg_idx, sg_w, b_idx, b_w) = transform_graph_map_to_gnn(map_)
    dim = map_.dim
    bounds_max = float(np.asarray(map_.bounds, dtype=float).max())
    x = ndata.astype(np.float32).copy()
    x[:, :dim] = x[:, :dim] / bounds_max

    data = HeteroData()
    data['node'].x = torch.tensor(x, dtype=torch.float)
    data['node', 'to', 'node'].edge_index = torch.tensor(e_idx.T, dtype=torch.long)
    data['node', 'to', 'node'].edge_attr = torch.tensor(
        _min_max_normalize_columns(e_w.reshape(-1, 1).astype(np.float64)), dtype=torch.float)
    data['node', 'to', 'node'].edge_weight = None
    data['node', 'approx', 'node'].edge_index = torch.tensor(sg_idx.T, dtype=torch.long)
    data['node', 'approx', 'node'].edge_attr = torch.tensor(
        _min_max_normalize_columns(sg_w.reshape(-1, 1).astype(np.float64)), dtype=torch.float)
    data['node', 'approx', 'node'].edge_weight = None
    transform = torch_geometric.transforms.Compose([AddSelfLoops('edge_attr', fill_value=0.0)])
    data = transform(data)
    # After AddSelfLoops so the boundary relation keeps only its own loops.
    if len(b_idx):
        data['node', 'boundary', 'node'].edge_index = torch.tensor(
            b_idx.reshape(-1, 2).T, dtype=torch.long)
        data['node', 'boundary', 'node'].edge_attr = torch.tensor(
            _min_max_normalize_columns(b_w.reshape(-1, 1).astype(np.float64)), dtype=torch.float)
    else:
        data['node', 'boundary', 'node'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['node', 'boundary', 'node'].edge_attr = torch.zeros((0, 1), dtype=torch.float)
    data['node', 'boundary', 'node'].edge_weight = None
    return data


def load_cluster_encoder(run_folder, sample_data: HeteroData, epoch: Optional[int] = None,
                         device: Optional[torch.device] = None):
    """Load a train_cluster.py wandb run's encoder for inference.

    Reads files/config.yaml (unwrapping wandb's {value: ...} nesting), builds
    the model from config['encoder']['model'], to_hetero's it on the
    get_dummy_sample_data metadata (to/approx/boundary — the layout the
    checkpoint was trained with), lazy-inits on sample_data, then loads
    files/model/epoch_{n}.pth (highest epoch when epoch is None).
    Returns (model.eval(), resolved_epoch).
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_folder = Path(run_folder)
    files = run_folder / "files" if (run_folder / "files").exists() else run_folder
    with open(files / "config.yaml") as f:
        cfg = _unwrap_wandb_config(yaml.safe_load(f))
    model_cfg = dict(cfg["encoder"]["model"])

    ckpts = sorted((files / "model").glob("epoch_*.pth"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no epoch_*.pth checkpoints under {files/'model'}")
    if epoch is None:
        ckpt = ckpts[-1]
    else:
        ckpt = files / "model" / f"epoch_{epoch}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found (have: {[p.name for p in ckpts]})")

    homogeneous = get_model(model_type=model_cfg['type'], **model_cfg)
    dim = sample_data['node'].x.shape[1] - 3
    metadata = get_dummy_sample_data(dim=dim).metadata()
    model = to_hetero(homogeneous, metadata, aggr=model_cfg['to_hetero_aggr']).to(device)
    with torch.no_grad():  # materialize lazy modules before load_state_dict
        ds = sample_data.to(device)
        model(dict(ds.x_dict), ds.edge_index_dict, ds.edge_attr_dict)
    state = torch.load(ckpt, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        stripped = { (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
                     for k, v in state.items() }
        model.load_state_dict(stripped)
    model.eval()
    return model, int(ckpt.stem.split("_")[1])


@torch.no_grad()
def gnn_seed_indices(model, data: HeteroData, map_: GraphSampler, k: int,
                     device: torch.device) -> Tuple[List[int], Dict[str, float]]:
    """Embedding K-means anchor nodes: cluster.md §16A + §17 medoids in latent
    space. Start/goal nodes are excluded (protected singletons, §5) — CTopPRM
    adds them back as the leading seeds. Returns (medoid node indices,
    timing dict with 'inference' and 'kmeans')."""
    from sklearn.cluster import KMeans

    t0 = time.perf_counter()
    ds = data.to(device)
    z = model(dict(ds.x_dict), ds.edge_index_dict, ds.edge_attr_dict)['node'].cpu().numpy()
    t_inf = time.perf_counter() - t0

    sg = set(map_.start_nodes_index.values()) | set(map_.goal_nodes_index.values())
    free = np.array([i for i in range(len(z)) if i not in sg], dtype=np.int64)
    k_free = min(max(k - len(sg), 1), len(free) - 1)
    t1 = time.perf_counter()
    km = KMeans(n_clusters=k_free, n_init=4, random_state=0).fit(z[free])
    seeds: List[int] = []
    for c in range(k_free):
        members = free[km.labels_ == c]
        if members.size == 0:
            continue
        d = np.linalg.norm(z[members] - km.cluster_centers_[c], axis=1)
        seeds.append(int(members[np.argmin(d)]))
    t_km = time.perf_counter() - t1
    return seeds, {"inference": t_inf, "kmeans": t_km}


def build_gnn_cluster_map(source_map: GraphSampler, agents: List[dict], model,
                          device: torch.device, cluster_fraction: float = 0.05
                          ) -> Tuple[GraphSampler, dict]:
    """GNN analogue of cluster_map.build_cluster_map: same K formula, same
    distilled-sampler assembly and alignment assertion; the cluster anchors
    come from the encoder instead of a graph clusterer."""
    agents_rt = agents_yaml_to_roadmap_frame(source_map, agents)
    starts = [tuple(a["start"]) for a in agents_rt]
    goals = [tuple(a["goal"]) for a in agents_rt]
    num_endpoints = len(dict.fromkeys(starts + goals))
    k = max(num_endpoints + 2, math.ceil(cluster_fraction * len(source_map.nodes)))

    t0 = time.perf_counter()
    data = sampler_to_cluster_heterodata(source_map)
    t_hetero = time.perf_counter()
    seeds, seed_times = gnn_seed_indices(model, data, source_map, k, device)
    t_seeds = time.perf_counter()

    planner = CTopPRM(source_map, clustering="custom", custom_seeds=seeds,
                      min_clusters=k)
    planner.set_up_distinct_paths(list(zip(starts, goals)))
    t_cluster = time.perf_counter()
    points, edges = planner.get_roadmap()
    t_distill = time.perf_counter()

    fresh = _build_setup_sampler(source_map)
    fresh.set_start(starts)
    fresh.set_goal(goals)
    nodes = fresh.generate_custom_nodes(points, filter_points=False)
    expected = _snap_points(fresh, points)
    got = np.asarray([n.current for n in nodes[: len(points)]], dtype=float)
    if len(nodes) < len(points) or not np.allclose(got, expected, atol=1e-9):
        raise RuntimeError(
            f"{GNN_METHOD}: distilled points rejected by point_expandable "
            f"({len(nodes)} nodes for {len(points)} points); refusing to save "
            f"a roadmap with misaligned edge indices"
        )
    fresh.generate_custom_roadmap(edges)
    t_build = time.perf_counter()

    stats = {
        "method": GNN_METHOD,
        "clustering_mode": "custom(gnn-embedding-kmeans)",
        "min_clusters": k,
        "num_clusters": len(planner.seed_indices),
        "num_nodes": len(fresh.nodes),
        "num_edges": len(fresh.edges),
        "source_num_nodes": len(source_map.nodes),
        "source_num_edges": len(source_map.edges),
        "runtime_breakdown": {
            "heterodata": t_hetero - t0,
            "inference": seed_times["inference"],
            "kmeans": seed_times["kmeans"],
            "cluster": t_cluster - t_seeds,
            "distill": t_distill - t_cluster,
            "build": t_build - t_distill,
        },
    }
    return fresh, stats


def create_gnn_cluster_maps(path: Path, num_cases: int, config: Dict,
                            run_folder, epoch: Optional[int],
                            cluster_fraction: float,
                            overwrite: bool = False, verbose: bool = True):
    """Sequential per-case GNN distillation (the model lives in-process).
    Skips existing pkls unless overwrite; raises listing per-case failures so
    downstream graph_files stay complete. Returns the resolved epoch."""
    path = Path(path)
    road_map_type = config["road_map_type"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    resolved_epoch = None
    failures = []
    for case_id in range(num_cases):
        set_global_seed(config.get("seed", 42) + case_id)
        case_path, map_path = generate_base_case_path(path, case_id, road_map_type)
        cluster_dir = generate_cluster_path(map_path)
        graph_file = get_graph_file_path(cluster_dir, GNN_GRAPH_NAME)
        if graph_file.exists() and not overwrite:
            continue
        try:
            with open(get_input_file_path(case_path), "r") as f:
                inpt = yaml.safe_load(f)
            source_graph_file = get_graph_file_path(map_path)
            if not source_graph_file.exists():
                failures.append((case_id, f"source roadmap missing: {source_graph_file}"))
                continue
            t0 = time.perf_counter()
            source_map = create_map(inpt, graph_file=source_graph_file, verbose=False,
                                    args={"use_constraint_sweep": False})
            t_load = time.perf_counter() - t0
            if model is None:
                sample = sampler_to_cluster_heterodata(source_map)
                model, resolved_epoch = load_cluster_encoder(run_folder, sample,
                                                             epoch=epoch, device=device)
                if verbose:
                    print(f"loaded encoder from {run_folder} (epoch {resolved_epoch}, {device})")
            cluster_map, stats = build_gnn_cluster_map(
                source_map, inpt["agents"], model, device, cluster_fraction)
            t_save0 = time.perf_counter()
            save_cluster_map(cluster_map, graph_file)
            stats["runtime_breakdown"]["load"] = t_load
            stats["runtime_breakdown"]["save"] = time.perf_counter() - t_save0
            stats["runtime"] = sum(stats["runtime_breakdown"].values())
            stats["source_graph"] = str(source_graph_file)
            stats["cluster_fraction"] = cluster_fraction
            stats["run_folder"] = str(run_folder)
            stats["epoch"] = resolved_epoch
            write_runtime_yaml(get_graph_runtime_file_path(cluster_dir, GNN_GRAPH_NAME), stats)
            if verbose:
                print(f"case_{case_id} {road_map_type}/{GNN_METHOD}: "
                      f"{stats['source_num_nodes']} -> {stats['num_nodes']} nodes, "
                      f"{stats['num_edges']} edges ({stats['runtime']:.2f}s)")
        except Exception as exc:  # noqa: BLE001 - reported to the driver
            failures.append((case_id, f"{exc}\n{traceback.format_exc()}"))
    if failures:
        details = "\n".join(f"case_{cid}: {msg}" for cid, msg in failures)
        raise RuntimeError(
            f"{len(failures)}/{num_cases} GNN cluster maps failed for "
            f"road_map_type={road_map_type}:\n{details}"
        )
    return resolved_epoch


def summarize_gnn_cluster_runtimes(path: Path, num_cases: int, config: Dict,
                                   output_file=None):
    """gnn analogue of summarize_cluster_runtimes; writes
    <path>/gnn_cluster_runtime.yaml keyed road_map_type -> gnn -> stats."""
    path = Path(path)
    road_map_type = config["road_map_type"]
    runtimes, breakdowns, cases = [], [], {}
    for case_id in range(num_cases):
        _, map_path = generate_base_case_path(path, case_id, road_map_type)
        runtime_file = get_graph_runtime_file_path(
            generate_cluster_path(map_path), GNN_GRAPH_NAME)
        if not runtime_file.exists():
            continue
        with open(runtime_file, "r") as f:
            data = yaml.safe_load(f) or {}
        if "runtime" not in data:
            continue
        runtime = float(data["runtime"])
        cases[f"case_{case_id}"] = round(runtime, 6)
        runtimes.append(runtime)
        breakdowns.append(data.get("runtime_breakdown", {}) or {})
    if output_file is None:
        output_file = path / "gnn_cluster_runtime.yaml"
    output_file = Path(output_file)
    existing = {}
    if output_file.exists():
        with open(output_file, "r") as f:
            existing = yaml.safe_load(f) or {}
    if runtimes:
        arr = np.asarray(runtimes, dtype=float)
        stage_keys = sorted({k for b in breakdowns for k in b})
        existing[road_map_type] = {GNN_METHOD: {
            "num_cases": len(runtimes),
            "total": round(float(arr.sum()), 6),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "mean_breakdown": {
                k: round(float(np.mean([b.get(k, 0.0) for b in breakdowns])), 6)
                for k in stage_keys
            },
            "cases": cases,
        }}
    with open(output_file, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)
    return output_file
