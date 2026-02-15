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
from path_planning.data_generation.trajectory_parser import get_trajectory_map

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
