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
from path_planning.gnn.dataset_prune import normalize_data, load_gnn_model, prune_map, get_prune_mechanism, get_prune_mechanism_folder
from path_planning.data_generation.dataset_util import generate_base_case_path, generate_roadmap_path, generate_ground_truth_path, get_graph_file_path
from path_planning.data_generation.dataset_util import generate_gnn_sampler_path,get_prediction_file_path
import matplotlib.pyplot as plt

def visualize_single_case_gnn(
        case_id: int,
        base_path: Path,
        road_map_type: str,
        gnn_folder_name: str,
        config: Dict,
        prune_mechanism: dict = None,
        k_hop: int = 0,
        show: bool = False,
        verbose: bool = True,
    ) -> None:
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
  
    gnn_sampler_path = generate_gnn_sampler_path(ground_truth_path, gnn_folder_name)
    prediction_file = get_prediction_file_path(gnn_sampler_path)

    if not prediction_file.exists():
        return 0
    pred = np.load(prediction_file)

    plt.close("all")
    visualizer = Visualizer2D(figname=f"{graph_file.parent.name} - Graph", figsize=(8, 8))
    visualizer.plot_grid_map(map_)
    visualizer.plot_road_map(map_, map_.nodes, map_.road_map, map_frame=map_.use_discrete_space, node_value=pred)
    visualizer.plot_density_map(density_map)
    visualizer.ax.set_title(f"{graph_dir.parent.name}/{graph_dir.name}")
    output_file = graph_dir / "graph_map_visualization.png"
    visualizer.savefig(output_file)
    if show:
        visualizer.show()
    visualizer.close()

    if prune_mechanism is not None:
        # Get the prune function & value -> save as npy file
        prune_name = get_prune_mechanism_folder(prune_mechanism)
        prune_value = get_prune_mechanism(prune_mechanism)(pred)
        prediction_file = get_prediction_file_path(gnn_sampler_path, prune_name)

        map_pruned, kept_indices = prune_map(map_, prune_value, k_hop)
        graph_file = get_graph_file_path(ground_truth_path, f"graph_sampler_{prune_name}_k{k_hop}.pkl")

        if not prediction_file.exists() or not graph_file.exists():
            return 0

    return 1

def visualize_gnn_maps(
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
        n = visualize_single_case_gnn(
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
