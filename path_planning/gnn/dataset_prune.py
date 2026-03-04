"""
"""

import os
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
from path_planning.multi_agent_planner.mapf_solver import solve_mapf
from path_planning.common.environment.map.graph_sampler import GraphSampler
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
from torch_geometric.data import HeteroData

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

def create_prune_mechanism(prune_mode: str, prune_std_scale: float, prune_value: float):
    mode = prune_mode.lower() if prune_mode.lower() in PRUNE_MECHANISMS_MODES else None
    std_scale = float(prune_std_scale) if prune_std_scale is not None else None
    value = float(prune_value) if prune_value is not None else None
    return {
        "mode": mode,
        "std_scale": std_scale,
        "value": value}

def get_prune_mechanism(prune_mechanism: dict = None):
    if prune_mechanism is not None:
        mode = prune_mechanism.get('mode', None)
        std_scale = prune_mechanism.get('std_scale', None)
        value = prune_mechanism.get('value', None)

        mode_func = None
        if mode in PRUNE_MECHANISMS_MODES:
            if mode == 'mean':
                mode_func = np.mean
            elif mode == 'median':
                mode_func = np.median

        def get_prune_value(out:np.ndarray) -> np.ndarray:
            assert isinstance(out, np.ndarray), "out must be a numpy array"
            threshold = 0
            if mode_func is not None:
                threshold = mode_func(out)
                if std_scale is not None:
                    threshold = threshold - out.std()*std_scale
            if value is not None:
                threshold = min(threshold, value)
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
        name = ""
        if mode is not None:
            name += f"{mode}"
        if std_scale is not None:
            name += f"std{std_scale}"
        if value is not None:
            name += f"value{value}"
        if name == "":
            name += "value0"
        return name

def _flatten_wandb_config(config: dict) -> dict:
    """Flatten wandb-style config where each value may be {'value': ...}."""
    out = {}
    for key, value in config.items():
        if isinstance(value, dict) and "value" in value:
            out[key] = value["value"]
        else:
            out[key] = value
    return out

def get_gnn_paths(run_folder: Optional[Path] = None, checkpoint_path: Optional[Path] = None, config_path: Optional[Path] = None, run_id: Optional[str] = None):
    assert sum([run_folder is not None, checkpoint_path is not None, config_path is not None]) == 1, "Exactly one of run_folder, checkpoint_path, and config_path must be provided"
    if run_folder is not None:
        run_folder = Path(run_folder)
        config_file = run_folder / "files" / "config.yaml"
        model_load_folder = run_folder / "files" / "model"
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        if not model_load_folder.exists():
            raise FileNotFoundError(f"Model folder not found: {model_load_folder}")
        model_load_files = sorted(
            [f for f in os.listdir(model_load_folder) if f.endswith(".pth")],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
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
    model_type = model_config.get("type", "gat")
    gnn_folder_name = f"{model_type}_{run_id}"
    return checkpoint_path, config_path, run_id, gnn_folder_name

def load_gnn_model(
    run_folder: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    sample_data: Optional[HeteroData] = None,
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
    checkpoint_path, config_path, run_id,gnn_folder_name = get_gnn_paths(run_folder, checkpoint_path, config_path, run_id)

    with open(config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_config = _flatten_wandb_config(train_config)

    if sample_data is None:
        raise ValueError("sample_data (HeteroData) is required to build model metadata and in_channels")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = dict(train_config.get("model", {}))
    node_in_channels = sample_data["node"].x.shape[1]
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
    if model_config.get("compile", False):
        model = torch.compile(model, dynamic=model_config.get("compile_dynamic", False))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    node_to_node_edges_arr = node_to_node_edges_arr.T
    node_to_node_weights_arr = node_to_node_weights_arr.reshape(len(node_to_node_weights_arr),-1)
    node_to_node_weights_arr = (node_to_node_weights_arr-node_to_node_weights_arr.min(axis=0,keepdims=True))/node_to_node_weights_arr.max(axis=0,keepdims=True)
    start_goal_edges_arr = start_goal_edges_arr.T
    start_goal_weights_arr = start_goal_weights_arr.reshape(len(start_goal_weights_arr),-1)
    start_goal_weights_arr = (start_goal_weights_arr-start_goal_weights_arr.min(axis=0,keepdims=True))/start_goal_weights_arr.max(axis=0,keepdims=True)

    data = HeteroData()
    data['node'].x = torch.tensor(ndata, dtype=torch.float)
    data['node','to','node'].edge_index = torch.tensor(node_to_node_edges_arr, dtype=torch.long)
    data['node','to','node'].edge_attr = torch.tensor(node_to_node_weights_arr, dtype=torch.float)
    data['node','to','node'].edge_weight = None
    data['node','approx','node'].edge_index = torch.tensor(start_goal_edges_arr, dtype=torch.long)
    data['node','approx','node'].edge_attr = torch.tensor(start_goal_weights_arr, dtype=torch.float)
    data['node','approx','node'].edge_weight = None
    
    return data


def process_single_case_gnn(
        case_id: int,
        base_path: Path,
        road_map_type: str,
        gnn_folder_name: str,
        model: torch.nn.Module,
        config: Dict,
        device: torch.device,
        prune_mechanism: dict = None,
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
    case_path = base_path / f"case_{case_id}"
    ground_truth_path = case_path / road_map_type / "ground_truth"
    graph_file = ground_truth_path / "graph_sampler.pkl"
    agents_dir = case_path / "agents"

    if not graph_file.exists():
        return 0
    if not agents_dir.exists():
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

    gnn_sampler_path = case_path / road_map_type / "gnn" /gnn_folder_name
    gnn_sampler_path.mkdir(parents=True, exist_ok=True)
    np.save(gnn_sampler_path / "predictions.npy", pred)

    if prune_mechanism is not None:
        # Get the prune function & value -> save as npy file
        prune_name = get_prune_mechanism_folder(prune_mechanism)
        prune_value = get_prune_mechanism(prune_mechanism)(pred)
        np.save(gnn_sampler_path / f"predictions_{prune_name}.npy" , prune_value)

        # Get indices with prune_value == 1
        nodes_idx = np.where(prune_value == 1)[0]

        # Always keep start and goal nodes
        start_indices = set(map_.start_nodes_index.values())
        goal_indices = set(map_.goal_nodes_index.values())
        kept_indices = np.array(
            list(set(int(i) for i in nodes_idx.tolist()) | start_indices | goal_indices),
            dtype=int,
        )

        # Create and save pruned graph sampler
        pruned_map = map_.create_pruned_copy(kept_indices)
        pruned_map.save_graph_sampler(str(gnn_sampler_path / f"graph_sampler_{prune_name}.pkl"))

    return 1

def create_gnn_maps(
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
        ground_truth_path = case_path / road_map_type / "ground_truth"
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
        )
        total_perms += n

    if verbose:
        print(f"GNN evaluation complete: {total_perms} permutations saved under */{gnn_folder_name}/")
