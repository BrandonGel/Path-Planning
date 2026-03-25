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

def load_and_visualize_case(perm_path: Path,graph_file: Path = None,mapf_solver_name: str = "cbs",road_map_type: str = "grid", agent_velocity: float = 0.0, show_static=True, show_animation=True,verbose=True):
    """
    Load and visualize a single training case.
    
    Args:
        perm_path: Path to the permutation directory
        mapf_solver_name: Name of the MAPF solver
        agent_velocity: Velocity of the agent
        show_static: Whether to show static path visualization
        show_animation: Whether to show animation
    """
    # Load input data
    input_file = get_input_file_path(perm_path)
    with open(input_file, "r") as f:
        input_data = yaml.safe_load(f)
    
    # Load solution data
    mapf_path = generate_mapf_path(perm_path, mapf_solver_name)
    roadmap_path = generate_roadmap_path(mapf_path, road_map_type)
    solution_name_suffix = get_solution_name_suffix(graph_file=graph_file)
    solution_file = get_solution_file_path(roadmap_path,solution_name_suffix,agent_velocity)
    with open(solution_file, "r") as f:
        solution_data = yaml.safe_load(f)

    if not solution_data["success"]:
        raise Exception(f"Solution not successful: {solution_file}")
    
    # Extract map parameters
    bounds = input_data["map"]["bounds"]
    resolution = input_data["map"]["resolution"]
    obstacles = np.array(input_data["map"]["obstacles"])
    agents = input_data["agents"]
    
    
    # Create map
    map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
    
    # Place obstacles
    if len(obstacles) > 0:
        map_.type_map[obstacles[:, 0], obstacles[:, 1]] = TYPES.OBSTACLE
    
    # Configure map
    map_.inflate_obstacles(radius=0)
    map_.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)
    
    # Set agent positions
    start = [agent['start'] for agent in agents]
    goal = [agent['goal'] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    
    # Generate nodes and roadmap
    nodes = map_.generateRandomNodes(generate_grid_nodes=True)
    road_map = map_.generate_roadmap(nodes)
    
    # Print case info
    if verbose:
        print(f"{solution_file.parent.parent.name} + {solution_file.name} | Agents: {len(agents)} | SoC: {solution_data['flowtime']} | Obstacles: {len(obstacles)}")
    
    # Static visualization
    if show_static:
        plt.close('all')
        vis = Visualizer2D(figname = f"{solution_file.name} - Static Paths", figsize=(8, 8))
        vis.plot_grid_map(map_)
        
        # Plot each agent's path
        schedule = solution_data["schedule"]
        for agent_name, trajectory in schedule.items():
            path = np.array([[point['x'], point['y']] for point in trajectory])
            vis.plot_path(path)
        
        plt.title(f"{perm_path.name} - Static Paths")
        vis.savefig(get_path_visualization_file(roadmap_path,solution_name_suffix,agent_velocity))
        if verbose:
            vis.show()
        vis.close()

        perm_trajectory_map = get_trajectory_map(solution_data["schedule"], map_)
        density_map = perm_trajectory_map.sum(axis=(0,1))
        visualizer = Visualizer2D(figname = f"{solution_file.name} - Density Map", figsize=(8, 8))
        masked_map = ~map_.get_obstacle_map()
        visualizer.plot_grid_map(map_, masked_map=masked_map)
        visualizer.plot_density_map(density_map)
        visualizer.savefig(get_heatmap_visualization_file(roadmap_path,solution_name_suffix,agent_velocity))
        if verbose:
            visualizer.show()
        visualizer.close()
    
    # Animation
    if show_animation:
        plt.close('all')
        vis = Visualizer2D(figsize=(8, 8))
        
        # Format schedule for animate method
        schedule = {"schedule": solution_data["schedule"]}
        
        # Create animation
        path_animation_file = get_path_animation_file(roadmap_path,solution_name_suffix,agent_velocity)
        vis.animate(path_animation_file, map_, schedule, road_map=road_map)
        if verbose:
            print(f"Animation saved to: {path_animation_file}")

def process_visualization(args):
    """Worker function for parallel visualization."""
    case_path,perm_dir, graph_file, mapf_solver_name, road_map_type, agent_velocity, show_static, show_animation, verbose = args
    try:
        load_and_visualize_case(perm_dir,graph_file=graph_file,mapf_solver_name=mapf_solver_name,road_map_type=road_map_type, agent_velocity=agent_velocity, show_static=show_static, show_animation=show_animation,verbose=verbose)
        return True, case_path.name, perm_dir.name
    except Exception as e:
        return False, case_path.name, f"{perm_dir.name}: {str(e)}"

def visualize_graph(tasks, num_workers: int = cpu_count()):
    successful = 0
    failed = 0
    num_tasks = len(tasks)
    print(f"\nVisualizing {num_tasks} {num_tasks} graphs with {num_workers} workers...")

    if num_workers > 1 and len(tasks) > 1:
        with Pool(processes=num_workers) as pool:
            for success, case_name, result in tqdm(
                pool.imap_unordered(process_visualization, tasks),
                total=len(tasks),
                desc="Visualizing tasks",
            ):
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"Error in {case_name}: {result}")
    else:
        for task in tqdm(tasks, desc="Visualizing tasks"):
            success, case_name, result = process_visualization(task)
            if success:
                successful += 1
            else:
                failed += 1
                print(f"Error in {case_name}: {result}")

    print("\n" + "="*60)
    print(f"Visualization complete: {successful} tasks succeeded, {failed} tasks failed")
    print("="*60)

def build_nx_graph(sampler: GraphSampler) -> nx.Graph:
    """Build a NetworkX graph from a GraphSampler roadmap."""
    graph = nx.Graph()
    for idx, node in enumerate(sampler.nodes):
        x_coord, y_coord = sampler.map_to_world(node.current)
        graph.add_node(idx, pos=(x_coord, y_coord))

    for node_idx, neighbors in enumerate(sampler.road_map):
        for neighbor_idx in neighbors:
            if node_idx < neighbor_idx:
                graph.add_edge(node_idx, neighbor_idx)
    return graph

def visualize_graph_sample(graph_dir: Path, sample_road_map_path: Path, show: bool = False, verbose: bool = True) -> Path:
    """Load graph_map.pkl from graph_dir and save a NetworkX visualization image."""
    graph_file = get_graph_file_path(graph_dir)
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file does not exist: {graph_file}")

    # Infer map bounds/resolution from case input.yaml so sampler.map_to_world() is correct.
    # Expected layout:
    #   case_N/sample/<road_type>/graph_<i>_<j>/graph_map.pkl
    case_dir = graph_dir.parents[2]
    input_file = get_input_file_path(case_dir)
    if input_file.exists():
        with open(input_file, "r") as f:
            input_data = yaml.safe_load(f)
        bounds = input_data["map"]["bounds"]
        resolution = float(input_data["map"]["resolution"])
    else:
        raise ValueError(f"Input file does not exist: {input_file}")

    map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
    map_.load_graph_sampler(graph_file)
    use_discrete_space = map_.use_discrete_space

    # graph = build_nx_graph(map_)
    # pos = nx.get_node_attributes(graph, "pos")

    plt.close("all")
    visualizer = Visualizer2D(figname=f"{graph_dir.name} - Graph", figsize=(8, 8))
    visualizer.plot_grid_map(map_)
    visualizer.plot_road_map(map_, map_.nodes, map_.road_map, map_frame=use_discrete_space)
    # nx.draw(
    #     graph,
    #     pos=pos,
    #     ax=visualizer.ax,
    #     node_size=10,
    #     node_color="#8c564b",
    #     edge_color="#e377c2",
    #     width=0.7,
    #     alpha=0.7,
    #     with_labels=False,
    # )
    visualizer.ax.set_title(f"{graph_dir.parent.name}/{graph_dir.name}")
    output_file = graph_dir / "graph_map_visualization.png"
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
    graph_dir, sample_road_map_path, show, verbose = args
    try:
        output_file = visualize_graph_sample(
            graph_dir=graph_dir,
            sample_road_map_path=sample_road_map_path,
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
        target_space: str = "binary",
        case_mode: str = "first_n",
        num_cases: int = 3,
        case_range: List[int] = [0, 16],
        specific_cases: List[int] = [0, 5, 10],
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

    tasks: List[Tuple[Path,Path, bool, bool]] = []
    target_space_type = target_space.lower() 
    if target_space_type not in TARGET_SPACE_TYPE_TO_NAME:
        raise ValueError(f"Target space type {target_space_type} not supported")
    for case_path in selected_cases:
        sample_base_path = generate_sample_base_path(case_path)
        for road_map_type in road_map_types:
            sample_road_map_path = generate_roadmap_path(sample_base_path, road_map_type)
            graph_dirs = sorted(
                [d for d in sample_road_map_path.iterdir() if d.is_dir() and d.name.startswith("graph_")],
                key=lambda value: value.name,
            )
            for graph_dir in graph_dirs:
                if get_graph_file_path(graph_dir).exists() and get_target_file_path(graph_dir, target_space_type).exists():
                    tasks.append((graph_dir, sample_road_map_path, show, verbose))
                else:
                    if not get_graph_file_path(graph_dir).exists():
                        print(f"Graph file does not exist: {get_graph_file_path(graph_dir)}")
                    if not get_target_file_path(graph_dir, target_space_type).exists():
                        print(f"Target file does not exist: {get_target_file_path(graph_dir, target_space_type)}")

    if verbose:
        print(f"Collected {len(tasks)} graph visualization tasks from {len(selected_cases)} cases")
    return tasks

