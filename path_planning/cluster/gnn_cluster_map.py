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
    apply_edge_ablation,
    get_dummy_sample_data,
)
from path_planning.gnn.model import get_model
from path_planning.utils.util import agents_yaml_to_roadmap_frame, set_global_seed

GNN_METHOD = "gnn"
GNN_GRAPH_NAME = "graph_map_gnn.pkl"


def gnn_embedding_name(method_name: str = GNN_METHOD) -> str:
    """Per-case node-embedding sidecar next to the pickle: row i is the
    encoder output for source_map.nodes[i]; shape (num_source_nodes, dim)."""
    return f"graph_map_{method_name}_embedding.npy"


def gnn_centroid_name(method_name: str = GNN_METHOD) -> str:
    """K-means centroids in embedding space, shape (num_seeds, dim); row j is
    the centroid whose latent-space medoid is seed j of gnn_seed_name."""
    return f"graph_map_{method_name}_centroids.npy"


def gnn_seed_name(method_name: str = GNN_METHOD) -> str:
    """Medoid seed node indices (into the source roadmap), shape (num_seeds,),
    in the order handed to CTopPRM (obstacle-aware: boundary then open space)."""
    return f"graph_map_{method_name}_seeds.npy"


def gnn_graph_name(method_name: str = GNN_METHOD) -> str:
    """Pickle name for a gnn-family method (e.g. gnn2 for the stage-2
    encoder) so multiple checkpoints' results can coexist per case."""
    return f"graph_map_{method_name}.pkl"


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


def sampler_to_cluster_heterodata(map_: GraphSampler, use_boundary_edges: bool = True,
                                  use_task_edges: bool = True,
                                  use_node_type_features: bool = True) -> HeteroData:
    """Live GraphSampler -> the cluster-training HeteroData layout.

    The three flags mirror the training-time ablation (dataset.config in the
    run's config.yaml, see ablation_flags_from_config): relations are dropped
    / the node one-hot flattened after assembly exactly as GraphDataset does,
    so the sample matches what the checkpoint was trained on.

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
    return apply_edge_ablation(data, use_boundary_edges, use_task_edges, use_node_type_features)


def read_run_config(run_folder) -> dict:
    """The train_cluster.py run's files/config.yaml with wandb's nesting unwrapped."""
    run_folder = Path(run_folder)
    files = run_folder / "files" if (run_folder / "files").exists() else run_folder
    with open(files / "config.yaml") as f:
        return _unwrap_wandb_config(yaml.safe_load(f))


def ablation_flags_from_config(cfg: dict) -> Tuple[bool, bool, bool]:
    """(use_boundary_edges, use_task_edges, use_node_type_features) recorded in
    the run config under dataset.config; missing keys (runs that predate the
    ablations, e.g. gnn6) mean the full relation set and the real one-hot."""
    dcfg = ((cfg.get("dataset") or {}).get("config") or {})
    return (bool(dcfg.get("use_boundary_edges", True)), bool(dcfg.get("use_task_edges", True)),
            bool(dcfg.get("use_node_type_features", True)))


def edge_flags_from_config(cfg: dict) -> Tuple[bool, bool]:
    """(use_boundary_edges, use_task_edges) — see ablation_flags_from_config."""
    return ablation_flags_from_config(cfg)[:2]


def load_cluster_encoder(run_folder, sample_data: HeteroData, epoch: Optional[int] = None,
                         device: Optional[torch.device] = None):
    """Load a train_cluster.py wandb run's encoder for inference.

    Reads files/config.yaml (unwrapping wandb's {value: ...} nesting), builds
    the model from config['encoder']['model'], to_hetero's it on the
    get_dummy_sample_data metadata (to + approx/boundary per the run's
    edge-ablation flags — the layout the checkpoint was trained with),
    lazy-inits on sample_data, then loads
    files/model/epoch_{n}.pth (highest epoch when epoch is None).
    Returns (model.eval(), resolved_epoch).
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_folder = Path(run_folder)
    files = run_folder / "files" if (run_folder / "files").exists() else run_folder
    cfg = read_run_config(run_folder)
    model_cfg = dict(cfg["encoder"]["model"])
    use_boundary_edges, use_task_edges, use_node_type_features = ablation_flags_from_config(cfg)
    from path_planning.gnn.train_cluster import (  # deferred: pulls in wandb
        HeteroClusterEncoder, _get_graph_unet_params)
    graph_unet_params = _get_graph_unet_params(cfg["encoder"], model_cfg)

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
    metadata = get_dummy_sample_data(dim=dim, use_boundary_edges=use_boundary_edges,
                                     use_task_edges=use_task_edges).metadata()
    # The caller may pass a full sample; drop the ablated relations so the lazy
    # init below sees exactly the checkpoint's relation set.
    sample_data = apply_edge_ablation(sample_data, use_boundary_edges, use_task_edges,
                                      use_node_type_features)
    model = to_hetero(homogeneous, metadata, aggr=model_cfg['to_hetero_aggr']).to(device)
    if graph_unet_params is not None:
        # stage-2 checkpoints wrap the traced encoder with a GraphUNet post-stage
        model = HeteroClusterEncoder(model, graph_unet_params,
                                     default_hidden=model_cfg.get('gnn_hidden_channels', 64)).to(device)
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


def _kmeans_medoids(z: np.ndarray, members: np.ndarray, k: int
                    ) -> Tuple[List[int], np.ndarray]:
    """K-means over z[members] followed by latent-space medoid selection
    (cluster.md §17). Returns (medoid node indices in global indexing,
    centroids with shape (len(seeds), dim)); empty clusters are dropped from
    both so row j of the centroids belongs to seeds[j]."""
    from sklearn.cluster import KMeans

    dim = z.shape[1] if z.ndim == 2 else 0
    k = int(min(max(k, 1), len(members) - 1)) if len(members) > 1 else 0
    if k <= 0:
        return [], np.empty((0, dim), dtype=z.dtype)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(z[members])
    seeds: List[int] = []
    kept: List[int] = []
    for c in range(k):
        cluster_members = members[km.labels_ == c]
        if cluster_members.size == 0:
            continue
        d = np.linalg.norm(z[cluster_members] - km.cluster_centers_[c], axis=1)
        seeds.append(int(cluster_members[np.argmin(d)]))
        kept.append(c)
    return seeds, np.asarray(km.cluster_centers_[kept], dtype=z.dtype)


def _boundary_distances(map_: GraphSampler) -> np.ndarray:
    """Per-node Euclidean distance to the obstacle boundary (d_i^B), from the
    CDT boundary extraction — works on maps without registered boundary nodes
    (the evaluation roots). Includes the outer map walls, matching §6."""
    from scipy.spatial import cKDTree

    pos = np.array([n.current for n in map_.nodes], dtype=float)
    bnd_pts, _, _ = map_.get_obstacle_boundary()
    if len(bnd_pts) == 0:
        return np.full(len(pos), np.inf)
    return cKDTree(np.asarray(bnd_pts, dtype=float)).query(pos)[0]


@torch.no_grad()
def gnn_seed_indices(model, data: HeteroData, map_: GraphSampler, k: int,
                     device: torch.device,
                     obstacle_aware: bool = False,
                     boundary_distance_threshold: float = 1.5,
                     obstacle_cluster_budget: Optional[float] = None
                     ) -> Tuple[List[int], Dict[str, float], Dict[str, np.ndarray]]:
    """Embedding K-means anchor nodes: cluster.md §16A + §17 medoids in latent
    space. Start/goal nodes are excluded (protected singletons, §5) — CTopPRM
    adds them back as the leading seeds.

    obstacle_aware=True enables §15 explicit cluster resolution: nodes are
    partitioned by d_i^B <= boundary_distance_threshold into V_B (obstacle
    region) and V_F (open space), and the K budget is split so V_B gets
    obstacle_cluster_budget of it (default: twice its population share,
    capped at 0.9) — more retained resolution near obstacles, coarser open
    space. Returns (medoid node indices, timing dict, latent arrays:
    'embeddings' (num_source_nodes, dim) row-aligned with map_.nodes, and
    'centroids' (len(seeds), dim) with row j the K-means centre of seed j)."""
    t0 = time.perf_counter()
    ds = data.to(device)
    z = model(dict(ds.x_dict), ds.edge_index_dict, ds.edge_attr_dict)['node'].cpu().numpy()
    t_inf = time.perf_counter() - t0

    sg = set(map_.start_nodes_index.values()) | set(map_.goal_nodes_index.values())
    free = np.array([i for i in range(len(z)) if i not in sg], dtype=np.int64)
    k_free = min(max(k - len(sg), 1), len(free) - 1)
    t1 = time.perf_counter()
    budget_info: Dict[str, float] = {}
    if not obstacle_aware:
        seeds, centroids = _kmeans_medoids(z, free, k_free)
    else:
        d_b = _boundary_distances(map_)[free]
        near = free[d_b <= boundary_distance_threshold]
        far = free[d_b > boundary_distance_threshold]
        if len(near) < 2 or len(far) < 2:
            seeds, centroids = _kmeans_medoids(z, free, k_free)  # degenerate split
        else:
            # Default: boost the obstacle region to 1.5x its population share,
            # capped so open space keeps at least a quarter of the budget
            # (walls make the "near boundary" population large on these maps).
            share = (obstacle_cluster_budget if obstacle_cluster_budget is not None
                     else min(0.75, 1.5 * len(near) / len(free)))
            k_b = int(np.clip(round(k_free * share), 1, len(near) - 1))
            k_f = int(np.clip(k_free - k_b, 1, len(far) - 1))
            seeds_b, cent_b = _kmeans_medoids(z, near, k_b)
            seeds_f, cent_f = _kmeans_medoids(z, far, k_f)
            seeds = seeds_b + seeds_f
            centroids = np.concatenate([cent_b, cent_f], axis=0)
            budget_info = {"k_boundary": k_b, "k_free_space": k_f,
                           "num_boundary_nodes": int(len(near)),
                           "num_free_nodes": int(len(far)),
                           "budget_share": round(float(share), 4)}
    t_km = time.perf_counter() - t1
    times = {"inference": t_inf, "kmeans": t_km}
    times.update(budget_info)
    latent = {"embeddings": z, "centroids": centroids,
              "seeds": np.asarray(seeds, dtype=np.int64)}
    return seeds, times, latent


def build_gnn_cluster_map(source_map: GraphSampler, agents: List[dict], model,
                          device: torch.device, cluster_fraction: float = 0.05,
                          obstacle_aware: bool = False,
                          boundary_distance_threshold: float = 1.5,
                          obstacle_cluster_budget: Optional[float] = None,
                          use_boundary_edges: bool = True,
                          use_task_edges: bool = True,
                          use_node_type_features: bool = True,
                          ) -> Tuple[GraphSampler, dict, Dict[str, np.ndarray]]:
    """GNN analogue of cluster_map.build_cluster_map: same K formula, same
    distilled-sampler assembly and alignment assertion; the cluster anchors
    come from the encoder instead of a graph clusterer. obstacle_aware routes
    the K budget per cluster.md §15 (see gnn_seed_indices). Returns
    (distilled sampler, stats, latent arrays from gnn_seed_indices)."""
    agents_rt = agents_yaml_to_roadmap_frame(source_map, agents)
    starts = [tuple(a["start"]) for a in agents_rt]
    goals = [tuple(a["goal"]) for a in agents_rt]
    num_endpoints = len(dict.fromkeys(starts + goals))
    k = max(num_endpoints + 2, math.ceil(cluster_fraction * len(source_map.nodes)))

    t0 = time.perf_counter()
    data = sampler_to_cluster_heterodata(source_map, use_boundary_edges, use_task_edges,
                                         use_node_type_features)
    t_hetero = time.perf_counter()
    seeds, seed_times, latent = gnn_seed_indices(
        model, data, source_map, k, device,
        obstacle_aware=obstacle_aware,
        boundary_distance_threshold=boundary_distance_threshold,
        obstacle_cluster_budget=obstacle_cluster_budget)
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
        "clustering_mode": ("custom(gnn-embedding-kmeans, obstacle-aware)"
                            if obstacle_aware else "custom(gnn-embedding-kmeans)"),
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
    if obstacle_aware:
        stats["obstacle_aware"] = {
            "boundary_distance_threshold": boundary_distance_threshold,
            **{key: seed_times[key] for key in
               ("k_boundary", "k_free_space", "num_boundary_nodes",
                "num_free_nodes", "budget_share") if key in seed_times},
        }
    return fresh, stats, latent


def create_gnn_cluster_maps(path: Path, num_cases: int, config: Dict,
                            run_folder, epoch: Optional[int],
                            cluster_fraction: float,
                            overwrite: bool = False, verbose: bool = True,
                            obstacle_aware: bool = False,
                            boundary_distance_threshold: float = 1.5,
                            obstacle_cluster_budget: Optional[float] = None,
                            method_name: str = GNN_METHOD,
                            save_embeddings: bool = True):
    """Sequential per-case GNN distillation (the model lives in-process).
    Skips existing pkls unless overwrite; raises listing per-case failures so
    downstream graph_files stay complete. With save_embeddings three numpy
    arrays are written next to the pickle: the encoder's node embeddings
    (gnn_embedding_name, float32, row-aligned with the *source* roadmap
    nodes), the K-means centroids (gnn_centroid_name) and the medoid seed
    indices they belong to (gnn_seed_name). Returns the resolved epoch."""
    path = Path(path)
    road_map_type = config["road_map_type"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Edge-ablation flags travel with the run: the test graph must carry the
    # same relations the checkpoint was trained on.
    use_boundary_edges, use_task_edges, use_node_type_features = \
        ablation_flags_from_config(read_run_config(run_folder))
    model = None
    resolved_epoch = None
    failures = []
    for case_id in range(num_cases):
        set_global_seed(config.get("seed", 42) + case_id)
        case_path, map_path = generate_base_case_path(path, case_id, road_map_type)
        cluster_dir = generate_cluster_path(map_path)
        graph_file = get_graph_file_path(cluster_dir, gnn_graph_name(method_name))
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
                sample = sampler_to_cluster_heterodata(source_map, use_boundary_edges, use_task_edges,
                                                       use_node_type_features)
                model, resolved_epoch = load_cluster_encoder(run_folder, sample,
                                                             epoch=epoch, device=device)
                if verbose:
                    print(f"loaded encoder from {run_folder} (epoch {resolved_epoch}, {device}; "
                          f"boundary edges {'on' if use_boundary_edges else 'OFF'}, "
                          f"task edges {'on' if use_task_edges else 'OFF'}, "
                          f"node type features {'on' if use_node_type_features else 'FLAT'})")
            cluster_map, stats, latent = build_gnn_cluster_map(
                source_map, inpt["agents"], model, device, cluster_fraction,
                obstacle_aware=obstacle_aware,
                boundary_distance_threshold=boundary_distance_threshold,
                obstacle_cluster_budget=obstacle_cluster_budget,
                use_boundary_edges=use_boundary_edges, use_task_edges=use_task_edges,
                use_node_type_features=use_node_type_features)
            t_save0 = time.perf_counter()
            save_cluster_map(cluster_map, graph_file)
            if save_embeddings:
                embedding_file = cluster_dir / gnn_embedding_name(method_name)
                centroid_file = cluster_dir / gnn_centroid_name(method_name)
                seed_file = cluster_dir / gnn_seed_name(method_name)
                np.save(embedding_file, np.asarray(latent["embeddings"], dtype=np.float32))
                np.save(centroid_file, np.asarray(latent["centroids"], dtype=np.float32))
                np.save(seed_file, np.asarray(latent["seeds"], dtype=np.int64))
                stats["embedding_file"] = str(embedding_file)
                stats["embedding_shape"] = [int(d) for d in latent["embeddings"].shape]
                stats["centroid_file"] = str(centroid_file)
                stats["centroid_shape"] = [int(d) for d in latent["centroids"].shape]
                stats["seed_file"] = str(seed_file)
            stats["runtime_breakdown"]["load"] = t_load
            stats["runtime_breakdown"]["save"] = time.perf_counter() - t_save0
            stats["runtime"] = sum(stats["runtime_breakdown"].values())
            stats["source_graph"] = str(source_graph_file)
            stats["cluster_fraction"] = cluster_fraction
            stats["run_folder"] = str(run_folder)
            stats["epoch"] = resolved_epoch
            stats["method"] = method_name
            write_runtime_yaml(get_graph_runtime_file_path(cluster_dir, gnn_graph_name(method_name)), stats)
            if verbose:
                print(f"case_{case_id} {road_map_type}/{method_name}: "
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
                                   output_file=None, method_name: str = GNN_METHOD):
    """gnn analogue of summarize_cluster_runtimes; writes
    <path>/gnn_cluster_runtime.yaml keyed road_map_type -> gnn -> stats."""
    path = Path(path)
    road_map_type = config["road_map_type"]
    runtimes, breakdowns, cases = [], [], {}
    for case_id in range(num_cases):
        _, map_path = generate_base_case_path(path, case_id, road_map_type)
        runtime_file = get_graph_runtime_file_path(
            generate_cluster_path(map_path), gnn_graph_name(method_name))
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
        existing[road_map_type] = {method_name: {
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
