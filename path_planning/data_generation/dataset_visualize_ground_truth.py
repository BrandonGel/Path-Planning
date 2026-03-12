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

def load_and_visualize_case(case_path: Path, show_static=True, show_animation=True):
    """
    Load and visualize a single training case.
    
    Args:
        case_path: Path to the case directory
        show_static: Whether to show static path visualization
        show_animation: Whether to show animation
    """
    # Load input data
    with open(case_path / "input.yaml", "r") as f:
        input_data = yaml.safe_load(f)
    
    # Load solution data
    with open(case_path / "solution.yaml", "r") as f:
        solution_data = yaml.safe_load(f)
    
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
    print(f"{case_path.parent.parent.name} + {case_path.name} | Agents: {len(agents)} | Cost: {solution_data['cost']} | Obstacles: {len(obstacles)}")
    
    # Static visualization
    if show_static:
        plt.close('all')
        vis = Visualizer2D(figname = f"{case_path.name} - Static Paths", figsize=(8, 8))
        vis.plot_grid_map(map_)
        
        # Plot each agent's path
        schedule = solution_data["schedule"]
        for agent_name, trajectory in schedule.items():
            path = np.array([[point['x'], point['y']] for point in trajectory])
            vis.plot_path(path)
        
        plt.title(f"{case_path.name} - Static Paths")
        vis.show()
        vis.close()

        perm_trajectory_map = get_trajectory_map(solution_data["schedule"], map_)
        density_map = perm_trajectory_map.sum(axis=(0,1))
        visualizer = Visualizer2D(figname = f"{case_path.name} - Density Map", figsize=(8, 8))
        masked_map = ~map_.get_obstacle_map()
        visualizer.plot_grid_map(map_, masked_map=masked_map)
        visualizer.plot_density_map(density_map)
        visualizer.savefig(case_path / 'density_map.png')
        visualizer.show()
        visualizer.close()
    
    # Animation
    if show_animation:
        plt.close('all')
        vis = Visualizer2D(figsize=(8, 8))
        
        # Format schedule for animate method
        schedule = {"schedule": solution_data["schedule"]}
        
        # Create animation
        temp_filename = f"temp_{case_path.name}_animation.gif"
        vis.animate(temp_filename, map_, schedule, road_map=road_map)
        print(f"Animation saved to: {temp_filename}")


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
    sol_files = list(solver_path.glob("solution_radius*_velocity*.yaml"))
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
        vis.savefig(solver_path / "paths.png")
        if show:
            vis.show()
        vis.close()

    if show_animation and hasattr(map_, "road_map") and map_.road_map is not None:
        plt.close("all")
        vis = Visualizer2D(figsize=(8, 8))
        combined_schedule = {"schedule": schedule}
        gif_path = solver_path / "animation.gif"
        vis.animate(str(gif_path), map_, combined_schedule, road_map=map_.road_map, map_frame=map_frame)
        vis.close()
