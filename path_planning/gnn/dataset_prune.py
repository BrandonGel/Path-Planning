"""
"""

import os
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.common.environment.map.graph_sampler import GraphSampler
from path_planning.common.environment.node import Node
from path_planning.utils.util import set_global_seed
import math
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
import copy


from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.utils.util import write_to_yaml, set_global_seed
from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from path_planning.multi_agent_planner.centralized.icbs.icbs import Environment as ICBS_Environment, CBS as ICBS
from path_planning.multi_agent_planner.centralized.lacam.lacam import LaCAM
from path_planning.multi_agent_planner.centralized.lacam.lacam_random import LaCAM as LaCAM_Random
from path_planning.multi_agent_planner.centralized.lacam.utility import set_starts_goals_config, is_valid_mapf_solution
from path_planning.multi_agent_planner.centralized.sipp.sipp import SippPlanner
from path_planning.gnn.dataloader import get_graph_dataset_file_paths, GraphDataset
from path_planning.gnn.train import get_threshold_x_dict

import time
import numpy as np
import os
from copy import deepcopy
from path_planning.utils.checker import check_time_anomaly, check_velocity_anomaly, check_collision
import yaml
from pathlib import Path
import torch
from torch_geometric.nn import to_hetero
import yaml
import os
from path_planning.gnn.model import get_model
from torch_geometric.data import HeteroData, Batch
from path_planning.data_generation.dataset_util import generate_base_case_path, generate_roadmap_path, generate_ground_truth_path, get_graph_file_path, generate_gnn_sampler_path,get_prediction_file_path
from torch_geometric.utils import unbatch

# example: 
# prune_mode = "mean_std0.5_0.5" 
# prune_mode = "mean_0.5" 
# prune_mode = "median_std0.5_value0.5" 
# prune_mode = "median_0.5" 
# prune_setting = {mode: 'mean', std_scale: 0.5, value: 0.5}
prune_mechanism = {'mode': None, 'std_scale': None, 'value': None}
PRUNE_MECHANISMS_MODES = ["mean", "median"]
PRUNE_MECHANISMS_SETTINGS = ["std", "std_scale"]
PRUNE_MECHANISMS_VALUES = ["value"]

def read_prune_mechanism_from_yaml(prune_mechanism_yaml: Path):
    with open(prune_mechanism_yaml, 'r') as f:
        prune_mechanism = yaml.load(f, Loader=yaml.FullLoader)
    prune_gnn = prune_mechanism.get('prune_gnn', False)
    threshold_scale = prune_mechanism.get('threshold_scale', 1.0)
    prune_mode = prune_mechanism.get('prune_mode', None)
    prune_std_scale = prune_mechanism.get('prune_std_scale', None)
    prune_value = prune_mechanism.get('prune_value', None)
    k_hop = prune_mechanism.get('k_hop', -1)
    prune_mechanism = create_prune_mechanism(prune_mode, prune_std_scale, prune_value, prune_gnn, threshold_scale, k_hop)
    return prune_mechanism

def create_prune_mechanism(prune_mode: str, prune_std_scale: float, prune_value: float, prune_gnn: bool = False,threshold_scale: float = 1.0,k_hop: int = -1):
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
            out = np.exp(threshold_scale*(out-threshold_value))
            if mode_func is not None:
                threshold = mode_func(out)
                if std_scale is not None:
                    threshold = threshold - out.std()*std_scale
            if value is not None:
                threshold = min(threshold, value)
            if threshold == float('inf'):
                return np.zeros_like(out)
            out[out > threshold] = 1
            out[out <= threshold] = 0
            return out 
        return get_prune_value
    return None

def get_prune_mechanism_folder(prune_mechanism: dict = None) -> Path:
    if prune_mechanism is not None:
        mode = prune_mechanism.get('mode', None)
        std_scale = prune_mechanism.get('std_scale', None)
        value = prune_mechanism.get('value', None)
        prune_gnn = prune_mechanism.get('prune_gnn', False)
        k_hop = prune_mechanism.get('k_hop', -1)
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
        if k_hop != -1:
            name += f"k{k_hop}"
        if threshold_scale != 1.0:
            name += f"thresholdscale{threshold_scale}"
        if name == "":
            name += ""
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

def _to_native_yaml(obj):
    """Convert numpy scalars/arrays in nested structures to native Python for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _to_native_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native_yaml(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def normalize_data(map_: GraphSampler):
    dims = map_.dim

    # Get Edges & Edge Weights
    edge_weights = map_.edge_weights
    start_goal_edges_dict = {**map_.get_start_nodes_with_all_edges(), **map_.get_goal_nodes_with_all_edges()}
    start_goal_nodes = map_.get_start_nodes() + map_.get_goal_nodes()


    # Generate Node Data ('node')
    pos = np.array([node.current for node in map_.get_nodes()])
    # Concatenate Position & zero class vector
    ndata = np.concatenate((pos, np.zeros((pos.shape[0], 2))), axis=1)
    start_goal_mask = np.full(len(ndata),fill_value=0).astype(bool)
    start_goal_idx = [map_.get_node_index(node) for node in start_goal_nodes]
    start_goal_mask[start_goal_idx] = True
    # Specify Start & Goal Nodes to have the one class value
    ndata[start_goal_mask, dims] = 1
    ndata[~start_goal_mask, dims+1] = 1

    # Generate Edges ('node', 'to', 'node')
    edges = np.array(map_.edges)
    edata = np.array(edge_weights)

    # Generate Start & Goal Edges ('node', 'approx', 'node')
    start_goal_edges_list = []
    start_goal_weights_list = []
    for u_idx, u_to_v_edge in start_goal_edges_dict.items():
        for v_idx, edge_weight in u_to_v_edge:
            start_goal_edges_list.append((u_idx, v_idx))
            start_goal_weights_list.append(edge_weight)
    
    # Convert to numpy arrays and save as compressed npz (much smaller than pickle)
    ndata = ndata.astype(np.float32)
    node_to_node_edges_arr = edges.astype(np.int32)
    node_to_node_weights_arr = edata.astype(np.float32)
    start_goal_edges_arr = np.array(start_goal_edges_list, dtype=np.int32)
    start_goal_weights_arr = np.array(start_goal_weights_list, dtype=np.float32)

    dims = len(map_.bounds)
    bounds = np.max(map_.bounds)
    ndata[:,:dims] = ndata[:,:dims] / bounds
    # Convert to PyG expected shapes: edge_index (2, E), edge_attr (E, F)
    node_to_node_edges_arr = node_to_node_edges_arr.T
    node_to_node_weights_arr = node_to_node_weights_arr.reshape(len(node_to_node_weights_arr), -1)
    if node_to_node_weights_arr.size > 0:
        nn_min = node_to_node_weights_arr.min(axis=0, keepdims=True)
        nn_max = node_to_node_weights_arr.max(axis=0, keepdims=True)
        nn_denom = np.where(nn_max > 0, nn_max, 1.0)
        node_to_node_weights_arr = (node_to_node_weights_arr - nn_min) / nn_denom

    start_goal_edges_arr = start_goal_edges_arr.T
    start_goal_weights_arr = start_goal_weights_arr.reshape(len(start_goal_weights_arr), -1)
    if start_goal_weights_arr.size > 0:
        sg_min = start_goal_weights_arr.min(axis=0, keepdims=True)
        sg_max = start_goal_weights_arr.max(axis=0, keepdims=True)
        sg_denom = np.where(sg_max > 0, sg_max, 1.0)
        start_goal_weights_arr = (start_goal_weights_arr - sg_min) / sg_denom

    data = HeteroData()
    data['node'].x = torch.tensor(ndata, dtype=torch.float)
    data['node','to','node'].edge_index = torch.tensor(node_to_node_edges_arr, dtype=torch.long)
    data['node','to','node'].edge_attr = torch.tensor(node_to_node_weights_arr, dtype=torch.float)
    data['node','to','node'].edge_weight = None
    data['node','approx','node'].edge_index = torch.tensor(start_goal_edges_arr, dtype=torch.long)
    data['node','approx','node'].edge_attr = torch.tensor(start_goal_weights_arr, dtype=torch.float)
    data['node','approx','node'].edge_weight = None
    
    return data

def heterodata_to_graph_sampler(
        data: HeteroData,
        bounds: np.ndarray,
        resolution: float = 1.0,
    ) -> GraphSampler:
    """
    Reconstruct a GraphSampler from a PyG HeteroData graph.

    Assumptions:
    - `data` was produced by `normalize_data(map_)` (or follows the same schema).
    - `data['node'].x[:, :dims]` are normalized coordinates where normalization used
      `np.max(bounds)` as in `normalize_data`.
    - `data['node','to','node'].edge_index` provides node indices (0..N-1).

    This function preserves node ordering so that `edge_index` indices remain valid.
    """
    if "node" not in data.node_types:
        raise ValueError("HeteroData must contain node type 'node'.")

    bounds_arr = np.asarray(bounds)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must have shape (dims, 2) like [[min,max], ...].")
    dims = int(bounds_arr.shape[0])

    x = data["node"].x.detach().cpu().numpy()
    if x.shape[1] < dims:
        raise ValueError(f"data['node'].x must have at least {dims} columns for positions.")

    scale = float(np.max(bounds_arr))
    coords = x[:, :dims] * scale
    num_nodes = int(coords.shape[0])

    # Identify start/goal nodes via the one-hot columns (if present)
    start_indices: List[int] = []
    goal_indices: List[int] = []
    if x.shape[1] >= dims + 2:
        start_indices = np.where(x[:, dims] > 0.5)[0].astype(int).tolist()
        goal_indices = np.where(x[:, dims + 1] > 0.5)[0].astype(int).tolist()

    start = [tuple(coords[i]) for i in start_indices]
    goal = [tuple(coords[i]) for i in goal_indices]

    # Build Node list preserving order (critical for edge_index alignment)
    nodes = [Node(tuple(coords[i]), None, 0, 0) for i in range(num_nodes)]

    # Build undirected adjacency and weights (mirror generate_custom_roadmap)
    road_map: List[List[int]] = [[] for _ in range(num_nodes)]
    road_map_edge_weights: List[List[float]] = [[] for _ in range(num_nodes)]
    edge_set = set()

    if ("node", "to", "node") not in data.edge_types:
        raise ValueError("HeteroData must contain edge type ('node','to','node').")
    edge_index = data["node", "to", "node"].edge_index.detach().cpu().numpy()
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E).")

    for u, v in edge_index.T.tolist():
        u = int(u)
        v = int(v)
        if u == v:
            continue
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        e = (u, v)
        if e in edge_set or (v, u) in edge_set:
            continue
        edge_set.add(e)
        edge_set.add((v, u))

        w = float(np.linalg.norm(coords[u] - coords[v]))
        road_map[u].append(v)
        road_map[v].append(u)
        road_map_edge_weights[u].append(w)
        road_map_edge_weights[v].append(w)

    # Treat non-start/goal nodes as sampled/grid points for bookkeeping
    start_goal_set = set(start_indices) | set(goal_indices)
    grid_points = [tuple(coords[i]) for i in range(num_nodes) if i not in start_goal_set]

    sampler_data = {
        "start": start,
        "goal": goal,
        "sample_num": len(grid_points),
        "num_neighbors": 0.0,
        "min_edge_length": 0.0,
        "max_edge_length": 0.0,
        "use_discrete_space": True,
        "grid_points": grid_points,
        "nodes": nodes,
        "obstacles": [],
        "inflation_radius": 0.0,
        "track_with_link": False,
        "road_map": road_map,
        "road_map_edge_weights": road_map_edge_weights,
        "use_constraint_sweep": False,
        "record_sweep": False,
        "use_exact_collision_check": True,
    }

    sampler = GraphSampler(bounds=bounds_arr, resolution=resolution, start=[], goal=[])
    sampler._load_from_dict(sampler_data)
    return sampler

def unnormalize_data(data: HeteroData, map_: GraphSampler) -> GraphSampler:
    """
    Project node features in a HeteroData object back into the GraphSampler.

    This is the inverse of the coordinate scaling done in normalize_data:
    - data['node'].x[:, :dims] stores normalized positions in [0, 1]
      (divided by max(map_.bounds)).
    - Here we rescale them back to world coordinates and write them into
      the corresponding GraphSampler nodes (in-place).

    Args:
        data: HeteroData produced by normalize_data(map_) (possibly modified).
        map_: Original GraphSampler to update.

    Returns:
        The same GraphSampler instance with updated node coordinates.
    """
    dims = map_.dim
    bounds = np.max(map_.bounds)

    # Extract (possibly modified) normalized positions and unscale
    x = data["node"].x.detach().cpu().numpy()
    coords = x[:, :dims] * bounds

    # Update GraphSampler node coordinates to match unnormalized positions
    for idx, node in enumerate(map_.get_nodes()):
        node.current = tuple(coords[idx])

    return map_

def prune_map(map_: GraphSampler, prune_value: np.ndarray, k_hop: int = 0) -> GraphSampler:
    # Get indices with prune_value == 1
    nodes_idx = np.where(prune_value == 1)[0]
    # Expand to include neighboring nodes
    if k_hop > 0:
        for _ in range(k_hop):
            nodes_idx_set = set(nodes_idx.tolist())
            for node_i in nodes_idx:
                if node_i < len(map_.road_map):
                    nodes_idx_set.update(map_.road_map[node_i])
            nodes_idx = np.array(list(nodes_idx_set))

    # Always keep start and goal nodes
    start_indices = set(map_.start_nodes_index.values())
    goal_indices = set(map_.goal_nodes_index.values())
    kept_indices = np.array(
        list(set(int(i) for i in nodes_idx.tolist()) | start_indices | goal_indices),
        dtype=int,
    )

    # Create and save pruned graph sampler
    pruned_map = map_.create_pruned_copy(kept_indices)
    return pruned_map, kept_indices

def process_single_case_gnn(
        case_id: int,
        base_path: Path,
        road_map_type: str,
        gnn_folder_name: str,
        model: torch.nn.Module,
        config: Dict,
        device: torch.device,
        prune_mechanism: dict = None,
        k_hop: int = 0,
    ) -> int:
    """
    Load graph_sampler.pkl for one case, run GNN inference for each permutation, save predictions.

    Args:
        case_id: Case index (case_{case_id}).
        base_path: Dataset path (e.g. .../agents4_obst0.1).
        road_map_type: One of grid, prm, planar (folder name under case_{id}).
        gnn_folder_name: Output folder name (e.g. gatv2_z447mk1j).
        model: Loaded GNN model (eval mode).
        config: Config with bounds, resolution, etc.
        device: Device to run inference on.

    Returns:
        Number of permutations processed.
    """
    base_path = Path(base_path)
    case_path,_ = generate_base_case_path(base_path, case_id, road_map_type)
    ground_truth_path = generate_roadmap_path(generate_ground_truth_path(case_path), road_map_type)
    graph_file = get_graph_file_path(ground_truth_path)
    if not graph_file.exists():
        return 0

    bounds = config.get("bounds", [[0, 32.0], [0, 32.0]])
    resolution = config.get("resolution", 1.0)
    map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
    map_.load_graph_sampler(str(graph_file))
    data = normalize_data(map_)
    data = data.to(device)
    edge_attr_dict = None
    if hasattr(data, "edge_attr_dict"):
        edge_attr_dict = data.edge_attr_dict
    else:
        edge_attr_dict = {
            ("node", "to", "node"): data["node", "to", "node"].edge_attr,
            ("node", "approx", "node"): data["node", "approx", "node"].edge_attr,
        }

    with torch.no_grad():
        try:
            out = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
        except TypeError:
            out = model(data.x_dict, data.edge_index_dict)

    if isinstance(out, dict):
        pred = out["node"].cpu().numpy()
    else:
        pred = out.cpu().numpy()    

    gnn_sampler_path = generate_gnn_sampler_path(ground_truth_path, gnn_folder_name)
    prediction_file = get_prediction_file_path(gnn_sampler_path)
    np.save(prediction_file, pred)

    if prune_mechanism is not None:
        # Get the prune function & value -> save as npy file
        prune_name = get_prune_mechanism_folder(prune_mechanism)
        prune_value = get_prune_mechanism(prune_mechanism)(pred)
        prediction_file = get_prediction_file_path(gnn_sampler_path, prune_name)
        np.save(prediction_file, prune_value)

        map_pruned, kept_indices = prune_map(map_, prune_value, k_hop)
        graph_file = get_graph_file_path(ground_truth_path, f"graph_sampler_{prune_name}_k{k_hop}.pkl")
        map_pruned.save_graph_sampler(graph_file)

    return 1

def create_gnn_map(
        path: Path,
        num_cases: int,
        map_config: Dict,
        run_folder: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        run_id: Optional[str] = None,
        road_map_type: Optional[str] = None,
        verbose: bool = True,
        prune_mechanism: dict = None,
        k_hop: int = 0,
    ) -> None:
    """
    Load all graph_sampler.pkl under path, normalize to HeteroData, run GNN inference,
    and save predictions into a folder named {model_type}_{run_id} alongside grid/prm/planar.

    Args:
        path: Base path for dataset (e.g. .../map32.0x32.0_resolution1.0/agents4_obst0.1).
        num_cases: Number of cases to process (case_0 .. case_{num_cases-1}).
        map_config: Configuration with bounds, resolution, etc. (used for GraphSampler and sample_data).
        run_folder: Wandb run folder (e.g. logs/.../wandb/run-20260217_062327-z447mk1j).
        checkpoint_path: Path to model .pth (use with config_path when run_folder is None).
        config_path: Path to config.yaml (use with checkpoint_path when run_folder is None).
        run_id: Optional run ID for folder name; inferred from run_folder if not set.
        road_map_type: Road map type to load graphs from (grid, prm, planar). Default from config.
        verbose: Whether to print progress.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    road_map_type = road_map_type or map_config.get("road_map_type", "grid")

    if verbose:
        print(f"GNN evaluation: loading model and building sample_data from first graph...")

    sample_data = None
    for case_id in range(num_cases):
        case_path = path / f"case_{case_id}"
        agent_radius = round(map_config.get("agent_radius", 0.0), 3)
        ground_truth_path = case_path / road_map_type / f"radius{agent_radius}" / "ground_truth"
        graph_file = ground_truth_path / "graph_sampler.pkl"
        agents_dir = case_path / "agents"
        if not graph_file.exists() or not agents_dir.exists():
            continue
        bounds = map_config.get("bounds", [[0, 32.0], [0, 32.0]])
        resolution = map_config.get("resolution", 1.0)
        map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
        map_.load_graph_sampler(str(graph_file))
        sample_data = normalize_data(map_)
        break

    if sample_data is None:
        raise FileNotFoundError(
            f"No graph_sampler.pkl and agents found under {path} for road_map_type={road_map_type}"
        )

    model, train_config, device, gnn_folder_name = load_gnn_model(
        run_folder=run_folder,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        run_id=run_id,
        sample_data=sample_data,
    )
    merged_config = {**map_config, **train_config}
    if verbose:
        print(f"GNN folder name: {gnn_folder_name}, processing {num_cases} cases...")

    total_perms = 0
    for case_id in tqdm(range(num_cases), desc="GNN cases", disable=not verbose):
        n = process_single_case_gnn(
            case_id=case_id,
            base_path=path,
            road_map_type=road_map_type,
            gnn_folder_name=gnn_folder_name,
            model=model,
            config=merged_config,
            device=device,
            prune_mechanism=prune_mechanism,
            k_hop=k_hop,
        )
        total_perms += n

    if verbose:
        print(f"GNN evaluation complete: {total_perms} permutations saved under */{gnn_folder_name}/")

def process_single_case_gnn_task(
        task: Tuple[Path, str],
        config: Dict,
    ) -> int:
    """
    Load graph_sampler.pkl for one case, run GNN inference for each permutation, save predictions.

    Args:
        task: Tuple[Path, str].
        model: Loaded GNN model (eval mode).
        config: Config with bounds, resolution, etc.
        device: Device to run inference on.   
        gnn_folder_name: Output folder name (e.g. gatv2_z447mk1j).
        prune_mechanism: Prune mechanism.
        k_hop: Number of hops to prune.

    Returns:
        Number of tasks processed.
    """
    graph_dir, graph_file_name = task[0], task[1]
    graph_file = get_graph_file_path(graph_dir, graph_file_name)
    if not graph_file.exists():
        return 0

    bounds = config.get("bounds", [[0, 32.0], [0, 32.0]])
    resolution = config.get("resolution", 1.0)
    map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
    map_.load_graph_sampler(str(graph_file), args={"use_constraint_sweep": False})
    data = normalize_data(map_)
    return graph_file,data,map_

def process_gnn_task_batch(
    graph_file_list: List[Path],
    data_list: List[HeteroData],
    map_list: List[GraphSampler],
    model: torch.nn.Module,
    device: torch.device,
    gnn_folder_name: str,
    prune_mechanism: Optional[dict] = None,
    ):

    data = Batch.from_data_list(data_list)
    data = data.to(device)
    edge_attr_dict = data.edge_attr_dict
    with torch.no_grad():
        try:
            out = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
        except TypeError:
            out = model(data.x_dict, data.edge_index_dict)

    if isinstance(out, dict):
        pred = out["node"].cpu().numpy()
    else:
        pred = out.cpu().numpy()    

    # TODO: NEED TO REVIST THIS
    prune_gnn = prune_mechanism.get('prune_gnn', False)
    threshold_values = None
    if prune_gnn:
        x_dict = get_threshold_x_dict(data.x_dict,out)  
        with torch.no_grad():
            out = prune_gnn(x_dict, data.edge_index_dict,data.edge_attr_dict,data.batch_dict)
        threshold_values = out.cpu().numpy()

    unbatched_indices = []
    unbatched_max_ind = 0
    for ii in range(len(data_list)):
        unbatched_indices.append(unbatched_max_ind+np.arange(len(data_list[ii]["node"]["x"])))
        unbatched_max_ind += len(data_list[ii]["node"]["x"])
    
    for ii in range(len(unbatched_indices)):
        unbatched_ind = unbatched_indices[ii]
        unbatched_pred = pred[unbatched_ind]
        graph_file = graph_file_list[ii]
        map_ = map_list[ii]
    
        gnn_sampler_path = generate_gnn_sampler_path(graph_file.parent, gnn_folder_name)
        prediction_file = get_prediction_file_path(gnn_sampler_path)
        np.save(prediction_file, unbatched_pred)

        if prune_mechanism is not None:
            # Get the prune function & value -> save as npy file
            prune_name = get_prune_mechanism_folder(prune_mechanism)
            if prune_name == "":
                return
            prune_fn = get_prune_function(prune_mechanism)
            k_hop = prune_mechanism.get('k_hop', -1)
            threshold_value = threshold_values[ii] if threshold_values is not None else 0

            prune_value = prune_fn(unbatched_pred, threshold_value)
            prediction_file = get_prediction_file_path(gnn_sampler_path, prune_name)
            np.save(prediction_file, prune_value)

            map_pruned, kept_indices = prune_map(map_, prune_value, k_hop)
            graph_file = get_graph_file_path(gnn_sampler_path, f"graph_sampler_{prune_name}_k{k_hop}.pkl")
            map_pruned.save_graph_sampler(graph_file)

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
    Load all graph_sampler.pkl under path, normalize to HeteroData, run GNN inference,
    and save predictions into a folder named {model_type}_{run_id} alongside grid/prm/planar.

    Args:
        path: Base path for dataset (e.g. .../map32.0x32.0_resolution1.0/agents4_obst0.1).
        num_cases: Number of cases to process (case_0 .. case_{num_cases-1}).
        map_config: Configuration with bounds, resolution, etc. (used for GraphSampler and sample_data).
        run_folder: Wandb run folder (e.g. logs/.../wandb/run-20260217_062327-z447mk1j).
        checkpoint_path: Path to model .pth (use with config_path when run_folder is None).
        config_path: Path to config.yaml (use with checkpoint_path when run_folder is None).
        run_id: Optional run ID for folder name; inferred from run_folder if not set.
        road_map_type: Road map type to load graphs from (grid, prm, planar). Default from config.
        verbose: Whether to print progress.
    """
    if len(tasks) == 0:
        if verbose:
            print("No tasks to process.")
        return

    graph_dir, graph_file_name = tasks[0][0], tasks[0][1]
    graph_file = get_graph_file_path(graph_dir, graph_file_name)

    if verbose:
        print("GNN evaluation: preparing model and task execution...")

    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    _, _, _, gnn_folder_name = get_gnn_paths(
        run_folder=run_folder,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        run_id=run_id,
    )

    if verbose:
        print(f"GNN folder name: {gnn_folder_name}, processing {len(tasks)} tasks...")

    bounds = map_config.get("bounds", [[0, 32.0], [0, 32.0]])
    resolution = map_config.get("resolution", 1.0)
    map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
    map_.load_graph_sampler(str(graph_file))
    sample_data = normalize_data(map_)

    model, train_config, device, gnn_folder_name = load_gnn_model(
        run_folder=run_folder,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        run_id=run_id,
        sample_data=sample_data,
    )
    threshold_model = None
    if train_config['threshold']['use']:
        threshold_model, _, _, _ = load_gnn_model(
            run_folder=run_folder,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            run_id=run_id,
            sample_data=sample_data,
            model_name_suffix="threshold",
        )
        prune_gnn = prune_mechanism.get('prune_gnn', False)
        if prune_gnn:
            prune_mechanism['prune_gnn'] = threshold_model
    merged_config = {**map_config, **train_config}

    data_list = []
    map_list = []
    graph_file_list = []
    for task in tqdm(tasks, desc="GNN tasks", disable=not verbose):
        graph_file, data, map_ = process_single_case_gnn_task(task, merged_config)
        data_list.append(data)
        graph_file_list.append(graph_file)
        map_list.append(map_)
        if len(data_list) >= batch_size:
            process_gnn_task_batch(graph_file_list, data_list, map_list, model, device, gnn_folder_name, prune_mechanism)
            data_list = []
            map_list = []
            graph_file_list = []

    if verbose:
        print(f"GNN evaluation complete: {len(tasks)} tasks saved under */{gnn_folder_name}/")