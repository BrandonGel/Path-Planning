"""
Visualize all training cases from the benchmark dataset.
Displays both static path visualizations and animations for each case.
"""
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.common.environment.map.graph_sampler import GraphSampler
from python_motion_planning.common import TYPES
from path_planning.data_generation.dataset_label import get_trajectory_map
from path_planning.data_generation.dataset_util import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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

def visualize_gt(tasks, num_workers: int = cpu_count()):
    successful = 0
    failed = 0
    num_tasks = len(tasks)
    print(f"\nVisualizing {num_tasks} permutations across {num_tasks} tasks with {num_workers} workers...")

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

def load_and_visualize_solver_case(
        solver_path: Path,
        map_,
        show_static: bool = True,
        show_animation: bool = False,
        show: bool = False,
        map_frame: bool = False,
    ):
    """
    Load and visualize a single solver result (valid solution only).
    Used for run_visualize_solvers.py: solution lives in perm_{id}/{solver}/solution_radius*_velocity*.yaml.

    Args:
        solver_path: Path to the solver directory (contains input.yaml and solution_radius*_velocity*.yaml)
        map_: GraphSampler instance (e.g. from create_map(..., load_graph_sampler=True))
        show_static: Whether to plot static paths and save paths.png
        show_animation: Whether to save an animation GIF
        show: Whether to show the matplotlib figure
        map_frame: Whether to use the map frame
    """
    sol_files = list(solver_path.glob("solution*_velocity*.yaml"))
    if not sol_files:
        raise FileNotFoundError(f"No solution_radius*_velocity*.yaml in {solver_path}")
    with open(sol_files[0], "r") as f:
        solution_data = yaml.safe_load(f)
    if not solution_data.get("success", False):
        raise ValueError(f"Solution not successful: {solver_path}")

    schedule = solution_data.get("schedule", {})
    if not schedule:
        raise ValueError(f"Empty schedule: {solver_path}")

    label = f"{solver_path.parent.parent.name}/{solver_path.name}"

    if show_static:
        plt.close("all")
        vis = Visualizer2D(figname=f"{label} - Static Paths", figsize=(8, 8))
        vis.plot_grid_map(map_)
        if hasattr(map_, "nodes") and hasattr(map_, "road_map") and map_.nodes and map_.road_map is not None:
            vis.plot_road_map(map_, map_.nodes, map_.road_map, map_frame=map_frame)
        for agent_name, trajectory in schedule.items():
            path = np.array([[p["x"], p["y"]] for p in trajectory])
            vis.plot_path(path, map_frame=map_frame)
        plt.title(f"{label} - Static Paths")
        path_img_file = get_path_visualization_file(solver_path)
        vis.savefig(path_img_file)
        if show:
            vis.show()
        vis.close()

    if show_animation and hasattr(map_, "road_map") and map_.road_map is not None:
        plt.close("all")
        agent_radius = solution_data["agent_radius"]
        agent_velocity = solution_data["agent_velocity"]
        vis = Visualizer2D(figsize=(8, 8))
        combined_schedule = {"schedule": schedule}
        path_animation_file = get_path_animation_file(solver_path)
        vis.animate(path_animation_file, map_, combined_schedule, road_map=map_.road_map, map_frame=map_frame,radius=agent_radius,velocity=agent_velocity)
        vis.close()
