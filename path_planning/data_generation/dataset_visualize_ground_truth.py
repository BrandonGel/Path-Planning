"""
Visualize all training cases from the benchmark dataset.
Displays both static path visualizations and animations for each case.
"""
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from path_planning.agent import Agent
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.data_generation.dataset_label import get_trajectory_map
from path_planning.data_generation.dataset_util import (
    generate_mapf_path,
    generate_roadmap_path,
    get_heatmap_visualization_file,
    get_input_file_path,
    get_path_animation_file,
    get_path_visualization_file,
    get_solution_file_path,
    get_solution_name_suffix,
)
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from path_planning.utils.util import (
    read_graph_sampler_from_yaml,
    read_agents_from_yaml,
    agents_yaml_to_roadmap_frame,
)

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
    map_ = read_graph_sampler_from_yaml(input_file, graph_file=graph_file, args={"use_collision_sweep": False})
    agents = read_agents_from_yaml(input_file)
    road_map = map_.road_map                    # read once: the property rebuilds every list per access
    
    # Load solution data
    mapf_path = generate_mapf_path(perm_path, mapf_solver_name)
    roadmap_path = generate_roadmap_path(mapf_path, road_map_type)
    solution_name_suffix = get_solution_name_suffix(graph_file=graph_file)
    solution_file = get_solution_file_path(roadmap_path,solution_name_suffix,agent_velocity)
    with open(solution_file, "r") as f:
        solution_data = yaml.safe_load(f)

    if not solution_data["success"]:
        print(f"Solution not successful: {solution_file}")
    
    # Set agent positions (YAML world coords → roadmap frame for type_map)
    agents_rt = agents_yaml_to_roadmap_frame(map_, agents)
    start = [a["start"] for a in agents_rt]
    goal = [a["goal"] for a in agents_rt]
    map_.set_start(start)
    map_.set_goal(goal)

    n_obstacles = len(map_.obstacles) if getattr(map_, "obstacles", None) else 0

    # Print case info
    if verbose:
        print(f"{solution_file.parent.parent.name} + {solution_file.name} | Agents: {len(agents)} | SoC: {solution_data['flowtime']} | Obstacles: {n_obstacles}")
    
    # Static visualization
    if show_static:
        plt.close('all')
        vis = Visualizer2D(figname = f"{solution_file.name} - Static Paths", figsize=(8, 8))
        vis.plot_grid_map(map_)
        
        # Plot each agent's path
        schedule = solution_data["schedule"]
        for agent_name, trajectory in schedule.items():
            path = np.array([[point['x'], point['y']] for point in trajectory])
            vis.plot_path(path, map_frame=False)      # schedules and nodes are world coordinates

        if map_.nodes:
            vis.plot_road_map(map_, map_.nodes, road_map, map_frame=False)
        
        plt.title(f"{perm_path.name} - Static Paths")
        vis.savefig(get_path_visualization_file(roadmap_path,solution_name_suffix,agent_velocity))
        if verbose:
            vis.show()
        vis.close()

        perm_trajectory_map = get_trajectory_map(solution_data["schedule"], map_)
        density_map = perm_trajectory_map.sum(axis=(0,1))
        visualizer = Visualizer2D(figname = f"{solution_file.name} - Density Map", figsize=(8, 8))
        visualizer.plot_grid_map(map_)
        if map_.nodes:
            visualizer.plot_road_map(map_, map_.nodes, road_map, map_frame=False)
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
        if map_.nodes:
            # Agents are drawn as circles: the bounding radius of the largest recorded body, else agent_radius.
            bodies = solution_data.get("agents") or {}
            radius = max((Agent.from_dict(b).radius for b in bodies.values()), default=solution_data["agent_radius"])
            vis.animate(path_animation_file, map_, schedule, road_map=road_map, map_frame=False, radius=radius)
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
