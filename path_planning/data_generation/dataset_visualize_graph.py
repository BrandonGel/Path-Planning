"""
Visualize all training cases from the benchmark dataset.
Displays both static path visualizations and animations for each case.
"""
import yaml
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.common.environment.map.graph_sampler import GraphSampler
from python_motion_planning.common import TYPES
from path_planning.data_generation.dataset_label import get_trajectory_map
from path_planning.data_generation.dataset_util import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from path_planning.data_generation.dataset_generate import TARGET_SPACE_TYPE_TO_NAME
from path_planning.utils.util import read_graph_sampler_from_yaml


def visualize_graph_sample(graph_dir: Path, graph_file: Path, target_file: Path,density_map_file: Path,save_path: Path, show: bool = False, verbose: bool = True) -> Path:
    """Load graph_map.pkl from graph_dir"""
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file does not exist: {graph_file}")

    case_dir = graph_dir.parents[2]
    input_file = get_input_file_path(case_dir)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    augmentation_id = int(graph_dir.stem.split("_")[-1])
    y = np.load(target_file)

    map_ =read_graph_sampler_from_yaml(input_file,graph_file=graph_file, args={"use_constraint_sweep": False})
    use_discrete_space = map_.use_discrete_space
    density_map = np.rot90(np.load(density_map_file), k=augmentation_id, axes=(0, 1))

    if len(y) > len(map_.nodes):
        y = y[y > 0]

    plt.close("all")
    visualizer = Visualizer2D(figname=f"{graph_dir.name} - Graph", figsize=(8, 8))
    visualizer.plot_grid_map(map_)
    visualizer.plot_road_map(map_, map_.nodes, map_.road_map, map_frame=use_discrete_space, node_value=y)
    visualizer.plot_density_map(density_map)
    visualizer.ax.set_title(f"{graph_dir.parent.name}/{graph_dir.name}")
    output_file = save_path / f"graph_map_{target_file.stem}.png"
    visualizer.savefig(output_file)
    if show:
        visualizer.show()
    visualizer.close()

    if verbose:
        print(f"Saved graph visualization: {output_file}")
    return output_file


def process_graph_visualization(
    args: Tuple[Path, Path, bool, bool],
) -> Tuple[bool, str, str]:
    """Worker for parallel graph sample visualization (must be top-level for Pool pickling)."""
    graph_dir,  graph_file, target_file,density_map_file,save_path, show, verbose = args
    try:
        output_file = visualize_graph_sample(
            graph_dir=graph_dir,
            graph_file=graph_file,
            target_file=target_file,
            density_map_file=density_map_file,
            save_path=save_path,
            show=show,
            verbose=verbose,
        )
        return True, graph_dir.name, str(output_file)
    except Exception as exc:
        return False, graph_dir.name, str(exc)


def visualize_graphs(
        tasks: List[Tuple[Path, Path, bool, bool]], num_workers: int = cpu_count()
    ) -> None:
    """Visualize all collected graph sample folders."""
    successful = 0
    failed = 0
    num_tasks = len(tasks)
    print(f"\nVisualizing {num_tasks} graph samples with {num_workers} workers...")

    if num_workers > 1 and num_tasks > 1:
        with Pool(processes=num_workers) as pool:
            for success, sample_name, result in tqdm(
                pool.imap_unordered(process_graph_visualization, tasks),
                total=num_tasks,
                desc="Visualizing graph samples",
            ):
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"Error in {sample_name}: {result}")
    else:
        for task in tqdm(tasks, total=num_tasks, desc="Visualizing graph samples"):
            success, sample_name, result = process_graph_visualization(task)
            if success:
                successful += 1
            else:
                failed += 1
                print(f"Error in {sample_name}: {result}")

    print("\n" + "=" * 60)
    print(f"Graph visualization complete: {successful} tasks succeeded, {failed} tasks failed")
    print("=" * 60)

def collect_graph_tasks(
        base_path: Path,
        road_map_types: List[str],
        graph_file_name: str = "graph_map.pkl",
        ground_truth_graph_file_name: str = "graph_map.pkl",
        ground_truth_road_map_type: str = "grid",
        target_space: str = "binary",
        case_mode: str = "first_n",
        num_cases: int = 3,
        case_range: List[int] = [0, 16],
        specific_cases: List[int] = [0, 5, 10],
        agent_velocity: float = 0.0,
        gnn_folder_name: str = None,
        show: bool = False,
        verbose: bool = True,
    ) -> List[Tuple[Path, bool, bool]]:
    """Collect graph sample visualization tasks under case_*/sample/{road_type}/graph_*_*."""
    case_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("case_") and len(os.listdir(d)) > 0],
        key=lambda value: int(value.name.split("_")[1]),
    )
    if len(case_dirs) == 0:
        raise ValueError(f"No case directories found in: {base_path}")

    case_range_arr = np.array(case_range).astype(int).flatten()
    specific_cases_arr = np.array(specific_cases).astype(int).flatten()
    if case_mode == "all":
        selected_cases = case_dirs
    elif case_mode == "first_n":
        selected_cases = case_dirs[:num_cases]
    elif case_mode == "range":
        selected_cases = case_dirs[case_range_arr[0]:case_range_arr[1]]
    elif case_mode == "specific":
        selected_cases = [case_dirs[i] for i in specific_cases_arr if i < len(case_dirs)]
    else:
        raise ValueError(f"Unsupported case_mode: {case_mode}")

    tasks: List[Tuple[Path,Path, str, Path, bool, bool]] = []
    target_space_type = target_space.lower() 
    
    for case_path in selected_cases:
        sample_base_path = generate_sample_base_path(case_path)
        ground_truth_path = generate_roadmap_path(generate_ground_truth_path(case_path), ground_truth_road_map_type)
        solution_name_suffix = get_solution_name_suffix(graph_file=ground_truth_graph_file_name)
        density_map_file = get_density_map_file(ground_truth_path, solution_name_suffix, agent_velocity)
        for road_map_type in road_map_types:
            sample_road_map_path = generate_roadmap_path(sample_base_path, road_map_type)
            graph_dirs = sorted(
                [d for d in sample_road_map_path.iterdir() if d.is_dir() and d.name.startswith("graph_")],
                key=lambda value: value.name,
            )
            for graph_dir in graph_dirs:
                if gnn_folder_name is not None:
                    gnn_sampler_path = generate_gnn_sampler_path(graph_dir, gnn_folder_name)
                    if target_space_type == 'predictions':
                        target_file = get_prediction_file_path(gnn_sampler_path)
                        graph_file = get_graph_file_path(graph_dir, graph_file_name)
                    else:
                        target_file = get_prediction_file_path(gnn_sampler_path, target_space_type)
                        graph_file = get_graph_file_path(gnn_sampler_path, graph_file_name)
                    save_path = gnn_sampler_path
                else:
                    graph_file = get_graph_file_path(graph_dir, graph_file_name)
                    target_file = get_target_file_path(graph_dir, target_space_type)
                    save_path = graph_dir
                if graph_file.exists() and (target_file.exists() or target_space == 'ignore'):
                    tasks.append((graph_dir, graph_file, target_file,density_map_file,save_path, show, verbose))
                else:
                    if not graph_file.exists():
                        print(f"Graph file does not exist: {graph_file}")
                    if not target_file.exists():
                        print(f"Target file does not exist: {target_file}")

    if verbose:
        print(f"Collected {len(tasks)} graph visualization tasks from {len(selected_cases)} cases")
    return tasks
