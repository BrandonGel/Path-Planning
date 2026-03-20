import yaml
import numpy as np
from pathlib import Path
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple
from path_planning.data_generation.dataset_ground_truth_util import *

def get_start_goal_locations(agents):
    start_goal_locations = []
    for agent in agents:
        start_goal_locations.append(agent['start'])
        start_goal_locations.append( agent['goal'])
    return np.array(start_goal_locations)

def get_longest_path(schedule):
    """Get the longest path length from all agents."""
    longest = 0
    for agent in schedule.keys():
        if len(schedule[agent]) > longest:
            longest = len(schedule[agent])
    return longest

def get_trajectory_map(schedule,map_,discrete: bool = True):    
    longest = get_longest_path(schedule)
    trajectory_map = np.zeros((len(schedule), longest,)+map_.shape,dtype=np.int32)
    for j, agent in enumerate(schedule.keys()):
        agent_path = schedule[agent]
        x = [point['x'] for point in agent_path[:]]
        y = [point['y'] for point in agent_path[:]]
        if 'z' in schedule[next(iter(schedule))][0]:
            z = [point['z'] for point in agent_path[:]]
            point = [(x[i],y[i],z[i]) for i in range(len(x))]
        else:
            z = None
            point = [(x[i],y[i]) for i in range(len(x))]
        
        for i, p in enumerate(point):
            index = (j,i,) + map_.world_to_map(p,True)
            trajectory_map[index] += 1
    
    return trajectory_map

def process_single_case_trajectories(args: Tuple) -> Tuple[bool, Path]:
    """
    Process trajectories for a single case.
    
    Args:
        args: Tuple of (case_dir, solution_name_suffix, mapf_solver_name,road_map_type, agent_velocity, visualize_density_map)
    
    Returns:
        Tuple of (success: bool, case_dir: Path)
    """
    case_dir,solution_name_suffix, mapf_solver_name,roadmap_type, agent_velocity, visualize_density_map = args
    perm_path = generate_perm_base_path(case_dir)
    gt_dir = generate_roadmap_path(generate_ground_truth_path(case_dir), roadmap_type)

    try:
        permutations = sorted([d for d in perm_path.iterdir() if d.is_dir() and d.name.startswith("perm_")])
        
        if not permutations:
            return False, perm_path

        input_file = get_input_file_path(permutations[0])
        agents = read_agents_from_yaml(input_file)
        start_goal_locations = get_start_goal_locations(agents)
        start_goal_file = get_start_goal_file(gt_dir)
        np.save(start_goal_file, start_goal_locations)
        
        map_ = read_graph_sampler_from_yaml(input_file)
        density_map = np.zeros(map_.shape)
        
        for perm_dir in permutations:
            mapf_path = generate_mapf_path(perm_dir, mapf_solver_name)
            roadmap_path = generate_roadmap_path(mapf_path, roadmap_type)
            solution_file = get_solution_file_path(roadmap_path, solution_name_suffix, agent_velocity)

            if not solution_file.exists():
                continue

            with open(solution_file) as f:
                schedule = yaml.load(f, Loader=yaml.FullLoader)

            perm_trajectory_map = get_trajectory_map(schedule["schedule"], map_)
            trajectory_map_file = get_trajectory_map_file(roadmap_path, solution_name_suffix, agent_velocity)
            np.save(trajectory_map_file, perm_trajectory_map)
            density_map += perm_trajectory_map.sum(axis=(0,1))

        obstacle_map = map_.get_obstacle_map().astype(int)
        obstacle_map_file = get_obstacle_map_file(gt_dir, solution_name_suffix, agent_velocity)
        np.save(obstacle_map_file, obstacle_map)
        density_map_file = get_density_map_file(gt_dir, solution_name_suffix, agent_velocity)
        np.save(density_map_file, density_map)
        
        if visualize_density_map:
            visualizer = Visualizer2D()
            masked_map = ~map_.get_obstacle_map()
            visualizer.plot_grid_map(map_, masked_map=masked_map)
            visualizer.plot_density_map(density_map/len(permutations))
            density_map_visualization_file = get_density_map_visualization_file(gt_dir, solution_name_suffix, agent_velocity)
            visualizer.savefig(density_map_visualization_file)
            visualizer.close()
        
        return True, case_dir
    except Exception as e:
        print(f"Error processing {case_dir.name}: {e}")
        return False, case_dir


def label_dataset(path, graph_file_name: Path = "graph_map.pkl", mapf_solver_name: str = "cbs", roadmap_type: str = "grid", agent_velocity: float = 0.0, visualize_density_map: bool = False, num_workers: int = None):
    """
    Label & parse all trajectories for all cases in a dataset directory.

    Args:
        path: Path to dataset directory containing case folders
        graph_file_name: Name of the graph file
        mapf_solver_name: Name of the MAPF solver
        roadmap_type: Type of the roadmap
        agent_velocity: Velocity of the agent
        visualize_density_map: Whether to visualize and save density maps
        num_workers: Number of parallel workers (default: auto-detect CPU cores)
    """
    path = Path(path)
    cases = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")],key=lambda x: int(x.name.split('_')[-1]))
    
    if not cases:
        print("No cases found to process")
        return
    
    print(f"Parsing trajectories for {len(cases)} cases")
    
    # Get number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    # Prepare case tasks
    solution_name_suffix = get_solution_name_suffix(graph_file_name)
    case_tasks = [(case_dir, solution_name_suffix, mapf_solver_name, roadmap_type, agent_velocity, visualize_density_map) for case_dir in cases]
    
    # Process cases in parallel
    successful = 0
    failed = 0
    
    if num_workers > 1 and len(case_tasks) > 1:
        with Pool(processes=num_workers) as pool:
            results = []
            for success, case_dir in tqdm(
                pool.imap_unordered(process_single_case_trajectories, case_tasks),
                total=len(case_tasks),
                desc="Parsing trajectories"
            ):
                results.append((success, case_dir))
                if success:
                    successful += 1
                else:
                    failed += 1
    else:
        # Sequential fallback
        for task in tqdm(case_tasks, desc="Parsing trajectories"):
            success, case_dir = process_single_case_trajectories(task)
            if success:
                successful += 1
            else:
                failed += 1

    print(f"Trajectory -- [{successful}/{len(cases)}] Complete!")
    if failed > 0:
        print(f"Warning: {failed} cases failed to process")


