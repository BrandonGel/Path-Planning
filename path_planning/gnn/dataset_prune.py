"""GNN pruning of roadmaps: load a trained score model (and its threshold model), score every node of a
saved ``GraphSampler``, keep the nodes the prune mechanism selects and save the pruned roadmap."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import to_hetero
from tqdm import tqdm

from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.data_generation.dataset_util import (
    generate_gnn_sampler_path, get_graph_file_path, get_graph_runtime_file_path, get_prediction_file_path,
    write_runtime_yaml,
)
from path_planning.gnn.dataloader import build_graph_data
from path_planning.gnn.model import get_model
from path_planning.gnn.train import get_threshold_x_dict, normalize_threshold

# A prune mechanism: {mode: 'mean' | 'median' | None, std_scale, value, prune_gnn, threshold_scale, k_hop},
# e.g. mean - 0.5 std, capped at value 0.5: {mode: 'mean', std_scale: 0.5, value: 0.5}.
PRUNE_MECHANISMS_MODES = ["mean", "median"]

def read_prune_mechanism_from_yaml(prune_mechanism_yaml: Path):
    with open(prune_mechanism_yaml, 'r') as f:
        prune_mechanism = yaml.load(f, Loader=yaml.FullLoader)
    prune_gnn = prune_mechanism.get('prune_gnn', False)
    threshold_scale = prune_mechanism.get('threshold_scale', 1.0)
    prune_mode = prune_mechanism.get('prune_mode', None)
    prune_std_scale = prune_mechanism.get('prune_std_scale', None)
    prune_value = prune_mechanism.get('prune_value', None)
    k_hop = prune_mechanism.get('k_hop', 0)

    assert k_hop >= 0, "k_hop must be >= 0"
    prune_mechanism = create_prune_mechanism(prune_mode, prune_std_scale, prune_value, prune_gnn, threshold_scale, k_hop)
    return prune_mechanism

def create_prune_mechanism(prune_mode: str, prune_std_scale: float, prune_value: float, prune_gnn: bool = False,threshold_scale: float = 1.0,k_hop: int = 0):
    mode = prune_mode.lower() if prune_mode and prune_mode.lower() in PRUNE_MECHANISMS_MODES else None
    std_scale = float(prune_std_scale) if prune_std_scale and prune_std_scale is not None else None
    value = float(prune_value) if prune_value and prune_value is not None else None
    return {
        "mode": mode,
        "std_scale": std_scale,
        "value": value,
        "prune_gnn": prune_gnn,
        "threshold_scale": threshold_scale,
        "k_hop": k_hop}

def get_prune_function(prune_mechanism: dict = None):
    if prune_mechanism is not None:
        mode = prune_mechanism.get('mode', None)
        std_scale = prune_mechanism.get('std_scale', None)
        value = prune_mechanism.get('value', None)
        threshold_scale = prune_mechanism.get('threshold_scale', 1.0)
        mode_func = None
        if mode in PRUNE_MECHANISMS_MODES:
            if mode == 'mean':
                mode_func = np.mean
            elif mode == 'median':
                mode_func = np.median

        def get_prune_value(out:np.ndarray,threshold_value:np.ndarray = 0) -> np.ndarray:
            assert isinstance(out, np.ndarray), "out must be a numpy array"
            threshold = float('inf')
            # Node keep-value as a probability in [0,1] (sigmoid), matching how the model is
            # trained (get_probability = sigmoid(alpha*(logit - threshold))). So `value` /
            # the mean/median threshold are probability cutoffs in [0,1] (e.g. 0.5).
            out = 1.0 / (1.0 + np.exp(-threshold_scale * (out - threshold_value)))
            if mode_func is not None:
                threshold = mode_func(out)
                if std_scale is not None:
                    threshold = threshold - out.std()*std_scale
            if value is not None:
                threshold = min(threshold, value)
            if threshold == float('inf'):
                return np.zeros_like(out)
            # Order-independent mask. (Mutating `out` in place — set >thr to 1 then <=thr to 0 —
            # was buggy for threshold >= 1: the just-set 1.0 entries satisfy 1.0 <= threshold and
            # got reset to 0, so any value >= 1 pruned everything.)
            return (out > threshold).astype(np.float64)
        return get_prune_value
    return None

def get_prune_mechanism_folder(prune_mechanism: dict = None) -> Path:
    if prune_mechanism is not None:
        mode = prune_mechanism.get('mode', None)
        std_scale = prune_mechanism.get('std_scale', None)
        value = prune_mechanism.get('value', None)
        prune_gnn = prune_mechanism.get('prune_gnn', False)
        k_hop = prune_mechanism.get('k_hop', 0)
        threshold_scale = prune_mechanism.get('threshold_scale', 1.0)
        name = ""
        if mode is not None:
            name += f"{mode}"
        if std_scale is not None:
            name += f"std{std_scale}"
        if value is not None:
            name += f"value{value}"
        if prune_gnn:
            name += f"gnn"
        if threshold_scale != 1.0:
            name += f"thresholdscale{threshold_scale}"
        if name == "":
            name += ""
        else:
            name += f"k{k_hop}"
        return name

def _flatten_wandb_config(config):
    """Recursively flatten wandb-style config where dicts may be {'value': ...}."""
    if isinstance(config, dict):
        if set(config.keys()) == {"value"}:
            return _flatten_wandb_config(config["value"])
        return {k: _flatten_wandb_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_flatten_wandb_config(v) for v in config]
    return config

def get_gnn_paths(
    run_folder: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    model_name_suffix: str = "",
):
    if run_folder is not None:
        run_folder = Path(run_folder)
        config_file = run_folder / "files" / "config.yaml"
        model_load_folder = run_folder / "files" / "model"
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        if not model_load_folder.exists():
            raise FileNotFoundError(f"Model folder not found: {model_load_folder}")
        all_model_files = [f for f in os.listdir(model_load_folder) if f.endswith(".pth")]
        if model_name_suffix:
            model_load_files = [f for f in all_model_files if f"_{model_name_suffix}.pth" in f]
        else:
            model_load_files = [f for f in all_model_files if "_threshold.pth" not in f]
        model_load_files = sorted(
            model_load_files,
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if not model_load_files:
            raise FileNotFoundError(f"No .pth files in {model_load_folder}")
        checkpoint_path = model_load_folder / model_load_files[-1]
        config_path = config_file
        if run_id is None:
            run_id = run_folder.name.strip()
            if run_id.startswith("run-"):
                run_id = run_id.split("-")[-1]
    else:
        if checkpoint_path is None or config_path is None:
            raise ValueError("Must provide run_folder or both checkpoint_path and config_path")
        checkpoint_path = Path(checkpoint_path)
        config_path = Path(config_path)
        if run_id is None:
            run_id = "custom"

    with open(config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_config = _flatten_wandb_config(train_config)
    # (Existing runs: the model block sits under "evaluator", in the oldest ones at the top level.)
    model_config = dict(train_config.get("model", {}))
    if not model_config:
        model_config = dict(train_config.get("evaluator", {}).get("model", {}))
    model_type = model_config.get("type", "gat")
    gnn_folder_name = f"{model_type}_{run_id}"
    return checkpoint_path, config_path, run_id, gnn_folder_name

def load_gnn_model(
    run_folder: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    sample_data: Optional[HeteroData] = None,
    model_name_suffix: str = "",
    ):
    """
    Load GNN model from either a wandb run folder or explicit checkpoint + config paths.

    Args:
        run_folder: Path to wandb run folder (e.g. logs/.../wandb/run-20260217_062327-z447mk1j).
        checkpoint_path: Path to .pth checkpoint (used when run_folder is None).
        config_path: Path to config.yaml (used when run_folder is None).
        run_id: Optional run ID for gnn_folder_name; inferred from run_folder name if not provided.
        sample_data: HeteroData from normalize_data(map_) to get metadata and node feature dim.
        model_name_suffix: "threshold" loads the run's threshold model (``epoch_<n>_threshold.pth``).

    Returns:
        Tuple of (model, config, device, gnn_folder_name).
    """
    checkpoint_path, config_path, run_id,gnn_folder_name = get_gnn_paths(
        run_folder,
        checkpoint_path,
        config_path,
        run_id,
        model_name_suffix=model_name_suffix,
    )

    with open(config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_config = _flatten_wandb_config(train_config)

    if sample_data is None:
        raise ValueError("sample_data (HeteroData) is required to build model metadata and in_channels")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'threshold' in model_name_suffix:
        node_in_channels = sample_data["node"].x.shape[1] + 1
        model_config = dict(train_config.get("threshold", {}).get("model", {}))
    else:
        node_in_channels = sample_data["node"].x.shape[1]
        model_config = dict(train_config.get("evaluator", {}).get("model", {}))
    model_config["node_in_channels"] = node_in_channels
    if "in_channels" in model_config:
        model_config["in_channels"] = node_in_channels

    model_type = model_config.get("type", "gat")
    model = get_model(model_type=model_type, **model_config)
    model = to_hetero(
        model,
        sample_data.metadata(),
        aggr=model_config.get("to_hetero_aggr", "sum"),
    ).to(device)
    device_config = train_config.get("device", {})
    if device_config.get("compile", False):
        model = torch.compile(model, dynamic=device_config.get("compile_dynamic", False))

    state_dict = torch.load(checkpoint_path, map_location=device)

    def _strip_orig_mod_prefix(sd: dict) -> dict:
        return {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()
        }

    def _add_orig_mod_prefix(sd: dict) -> dict:
        return {
            (k if k.startswith("_orig_mod.") else f"_orig_mod.{k}"): v
            for k, v in sd.items()
        }

    # Support checkpoints saved from both compiled and uncompiled modules.
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        try:
            model.load_state_dict(_strip_orig_mod_prefix(state_dict))
        except RuntimeError:
            model.load_state_dict(_add_orig_mod_prefix(state_dict))
    model.eval()

    return model, train_config, device, gnn_folder_name

def normalize_data(map_: GraphSampler) -> HeteroData:
    """``map_`` as the model's input graph, built by the same ``build_graph_data`` the training loader
    uses: node features ``[position | start/goal | other]``, the roadmap edges and the start / goal ->
    node edges with their lengths."""
    dims = map_.dim
    pos = np.array([node.current for node in map_.get_nodes()], dtype=float).reshape(-1, dims)
    ndata = np.concatenate((pos, np.zeros((len(pos), 2))), axis=1)
    start_goal_mask = np.zeros(len(ndata), dtype=bool)
    start_goal_mask[list(map_.start_nodes_index.values()) + list(map_.goal_nodes_index.values())] = True
    ndata[start_goal_mask, dims] = 1
    ndata[~start_goal_mask, dims + 1] = 1

    start_goal_edges, start_goal_weights = [], []
    for u_idx, u_to_v_edge in {**map_.get_start_nodes_with_all_edges(), **map_.get_goal_nodes_with_all_edges()}.items():
        for v_idx, edge_weight in u_to_v_edge:
            start_goal_edges.append((u_idx, v_idx))
            start_goal_weights.append(edge_weight)

    return build_graph_data(ndata, map_.bounds, np.asarray(map_.edges, dtype=np.int64),
                            np.asarray(map_.edge_weights, dtype=np.float32),
                            np.asarray(start_goal_edges, dtype=np.int64),
                            np.asarray(start_goal_weights, dtype=np.float32))


def prune_map(map_: GraphSampler, prune_value: np.ndarray, k_hop: int = 0) -> GraphSampler:
    # Get indices with prune_value == 1
    nodes_idx = np.where(prune_value == 1)[0]
    # Expand to include neighboring nodes
    if k_hop > 0:
        for _ in range(k_hop):
            nodes_idx_set = set(nodes_idx.tolist())
            for node_i in nodes_idx.tolist():
                nodes_idx_set.update(map_.neighbors(node_i).tolist())
            nodes_idx = np.array(sorted(nodes_idx_set), dtype=int)

    # Always keep start and goal nodes
    start_indices = set(map_.start_nodes_index.values())
    goal_indices = set(map_.goal_nodes_index.values())
    kept_indices = np.array(
        list(set(int(i) for i in nodes_idx.tolist()) | start_indices | goal_indices),
        dtype=int,
    )

    # Create and save pruned graph sampler
    pruned_map = map_.pruned(kept_indices)      # in place: map_ itself is pruned (kept_indices are its old numbering)
    return pruned_map, kept_indices

def process_single_case_gnn_task(task: Tuple[Path, str, str], config: Dict):
    """Load one task's roadmap and turn it into the model's input graph.

    ``task``: ``(input_file, graph_dir, graph_file_name)``; ``config``: the map config (``bounds``,
    ``resolution``). Returns ``(graph_file, data, map_, (load_seconds, normalize_seconds))``, or None when
    the graph file does not exist."""
    graph_file = get_graph_file_path(task[1], task[2])
    if not graph_file.exists():
        return None

    t0 = time.perf_counter()
    map_ = GraphSampler(bounds=config["bounds"], resolution=config["resolution"], start=[], goal=[])
    map_.load_graph_sampler(str(graph_file), args={"use_collision_sweep": False})
    t_load = time.perf_counter() - t0  # unpickle the source roadmap
    t0 = time.perf_counter()
    data = normalize_data(map_)
    t_normalize = time.perf_counter() - t0  # GraphSampler -> HeteroData features
    return graph_file, data, map_, (t_load, t_normalize)

def process_gnn_task_batch(
    graph_file_list: List[Path],
    data_list: List[HeteroData],
    map_list: List[GraphSampler],
    model: torch.nn.Module,
    device: torch.device,
    gnn_folder_name: str,
    prune_mechanism: Optional[dict] = None,
    load_time_list: Optional[List[Tuple[float, float]]] = None,
    threshold_model: Optional[torch.nn.Module] = None,
    ):
    """Run GNN inference on a batch of graphs and save predictions / pruned roadmaps.

    ``threshold_model`` (the run's threshold GNN) gives each graph its own cut-off, used when the prune
    mechanism has ``prune_gnn``; it is applied as in training (``normalize_threshold``).
    ``load_time_list`` (per graph, ``(load_seconds, normalize_seconds)`` measured in
    process_single_case_gnn_task) feeds the ``<pruned pickle stem>_runtime.yaml`` sidecar
    written next to every pruned roadmap. Batched forward passes are attributed to the
    graphs as an even share of the batch time (``batch_size`` is recorded so it can be undone).
    """
    n_graphs = len(data_list)
    data = Batch.from_data_list(data_list).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict)
    t_infer = time.perf_counter() - t0
    pred = out["node"].cpu().numpy()

    prune_name = get_prune_mechanism_folder(prune_mechanism) if prune_mechanism is not None else ""
    threshold_values, t_thr = None, 0.0
    if prune_name and prune_mechanism.get('prune_gnn', False):
        if threshold_model is None:
            raise ValueError("prune_gnn needs the run's threshold model")
        t0 = time.perf_counter()
        with torch.no_grad():
            threshold = threshold_model(get_threshold_x_dict(data.x_dict, out), data.edge_index_dict,
                                        data.edge_attr_dict, data.batch_dict)
            threshold = normalize_threshold(threshold, data.batch_dict, out, 'node')
        threshold_values = threshold.cpu().numpy()
        t_thr = time.perf_counter() - t0

    offsets = np.cumsum([0] + [len(d["node"]["x"]) for d in data_list])
    for ii, (graph_file, map_) in enumerate(zip(graph_file_list, map_list)):
        unbatched_pred = pred[offsets[ii]:offsets[ii + 1]]
        gnn_sampler_path = generate_gnn_sampler_path(graph_file.parent, gnn_folder_name)
        np.save(get_prediction_file_path(gnn_sampler_path), unbatched_pred)
        if not prune_name:                     # no prune mechanism (or an empty one): predictions only
            continue

        prune_fn = get_prune_function(prune_mechanism)
        threshold_value = threshold_values[ii] if threshold_values is not None else 0
        t0 = time.perf_counter()
        prune_value = prune_fn(unbatched_pred, threshold_value)
        num_nodes = len(map_.nodes)
        map_pruned, _ = prune_map(map_, prune_value, prune_mechanism.get('k_hop', 0))
        t_prune = time.perf_counter() - t0

        t0 = time.perf_counter()
        np.save(get_prediction_file_path(gnn_sampler_path, prune_name), prune_value)
        pruned_graph_name = f"graph_sampler_{prune_name}.pkl"
        map_pruned.save_graph_sampler(get_graph_file_path(gnn_sampler_path, pruned_graph_name))
        t_save = time.perf_counter() - t0

        t_load, t_normalize = (
            (float(load_time_list[ii][0]), float(load_time_list[ii][1]))
            if load_time_list is not None else (0.0, 0.0)
        )
        breakdown = {
            "load": t_load,
            "normalize": t_normalize,
            "inference": t_infer / n_graphs,
            "threshold": t_thr / n_graphs,
            "prune": t_prune,
            "save": t_save,
        }
        write_runtime_yaml(
            get_graph_runtime_file_path(gnn_sampler_path, pruned_graph_name),
            {
                "prune_name": prune_name,
                "source_graph": graph_file.name,
                "runtime": sum(breakdown.values()),
                "runtime_breakdown": breakdown,
                "batch_size": n_graphs,
                "device": str(device),
                "num_nodes": len(map_pruned.nodes),
                "num_edges": len(map_pruned.edges),
                "source_num_nodes": num_nodes,
            },
        )

def create_gnn_map_tasks(
        tasks,
        map_config: Dict,
        run_folder: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        run_id: Optional[str] = None,
        prune_mechanism: dict = None,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
    """
    For every task's roadmap: run GNN inference, save the predictions and (with a prune mechanism) the
    pruned roadmap, in a folder named {model_type}_{run_id} next to the source graph.

    Args:
        tasks: ``(input_file, graph_dir, graph_file_name)`` per roadmap (``collect_graph_tasks``).
        map_config: The map's ``bounds`` and ``resolution``.
        run_folder: Wandb run folder (e.g. logs/.../wandb/run-20260217_062327-z447mk1j).
        checkpoint_path: Path to model .pth (use with config_path when run_folder is None).
        config_path: Path to config.yaml (use with checkpoint_path when run_folder is None).
        run_id: Optional run ID for folder name; inferred from run_folder if not set.
        prune_mechanism: See ``create_prune_mechanism``; None saves the predictions only.
        batch_size: Graphs per forward pass.
        verbose: Whether to print progress.
    """
    if len(tasks) == 0:
        if verbose:
            print("No tasks to process.")
        return

    if verbose:
        print("GNN evaluation: preparing model and task execution...")

    first = process_single_case_gnn_task(tasks[0], map_config)
    if first is None:
        raise FileNotFoundError(f"Graph file not found: {get_graph_file_path(tasks[0][1], tasks[0][2])}")
    sample_data = first[1]

    load_args = dict(run_folder=run_folder, checkpoint_path=checkpoint_path, config_path=config_path,
                     run_id=run_id, sample_data=sample_data)
    model, train_config, device, gnn_folder_name = load_gnn_model(**load_args)
    threshold_model = None
    if prune_mechanism is not None and prune_mechanism.get('prune_gnn', False):
        if not train_config.get('threshold', {}).get('use', False):
            raise ValueError("prune_gnn is set, but this run was trained without a threshold model "
                             "(threshold.use is off in its config)")
        threshold_model, _, _, _ = load_gnn_model(**load_args, model_name_suffix="threshold")

    if verbose:
        print(f"GNN folder name: {gnn_folder_name}, processing {len(tasks)} tasks...")

    batch = []                                   # (graph_file, data, map_, load times) of the graphs waiting

    def flush():
        if batch:
            graph_files, data_list, map_list, load_times = map(list, zip(*batch))
            process_gnn_task_batch(graph_files, data_list, map_list, model, device, gnn_folder_name,
                                   prune_mechanism, load_time_list=load_times, threshold_model=threshold_model)
            batch.clear()

    for k, task in enumerate(tqdm(tasks, desc="GNN tasks", disable=not verbose)):
        loaded = first if k == 0 else process_single_case_gnn_task(task, map_config)
        if loaded is None:
            print(f"Graph file not found, skipped: {get_graph_file_path(task[1], task[2])}")
            continue
        batch.append(loaded)
        if len(batch) >= batch_size:
            flush()
    flush()

    if verbose:
        print(f"GNN evaluation complete: {len(tasks)} tasks saved under */{gnn_folder_name}/")
