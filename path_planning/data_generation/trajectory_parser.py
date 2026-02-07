import yaml
import numpy as np
from pathlib import Path
from path_planning.common.visualizer.visualizer_2d import Visualizer2D
from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple

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
        args: Tuple of (case_dir, visualize_density_map)
    
    Returns:
        Tuple of (success: bool, case_dir: Path)
    """
    case_dir, visualize_density_map = args

    gt_dir = case_dir / "ground_truth" if (case_dir / "ground_truth").exists() else case_dir

    try:
        permutations = sorted([d for d in gt_dir.iterdir() if d.is_dir() and d.name.startswith("perm_")])
        
        if not permutations:
            return False, case_dir

        agents = read_agents_from_yaml(permutations[0] / 'input.yaml')
        start_goal_locations = get_start_goal_locations(agents)
        np.save(gt_dir / 'start_goal_locations.npy', start_goal_locations)
        
        map_ = read_graph_sampler_from_yaml(permutations[0] / 'input.yaml')
        density_map = np.zeros(map_.shape)
        
        for perm_dir in permutations:
            solution_file = perm_dir / "solution.yaml"

            if not solution_file.exists():
                continue

            with open(solution_file) as f:
                schedule = yaml.load(f, Loader=yaml.FullLoader)

            perm_trajectory_map = get_trajectory_map(schedule["schedule"], map_)
            np.save(perm_dir / 'trajectory_map.npy', perm_trajectory_map)
            density_map += perm_trajectory_map.sum(axis=(0,1))

        obstacle_map = map_.get_obstacle_map().astype(int)
        np.save(gt_dir / 'obstacle_map.npy', obstacle_map)
        np.save(gt_dir / 'density_map.npy', density_map)
        
        if visualize_density_map:
            visualizer = Visualizer2D()
            masked_map = ~map_.get_obstacle_map()
            visualizer.plot_grid_map(map_, masked_map=masked_map)
            visualizer.plot_density_map(density_map/len(permutations))
            visualizer.savefig(gt_dir / 'density_map.png')
            visualizer.close()
        
        return True, case_dir
    except Exception as e:
        print(f"Error processing {case_dir.name}: {e}")
        return False, case_dir


def parse_dataset_trajectories(path, visualize_density_map: bool = False, num_workers: int = None):
    """
    Parse trajectories for all cases in a dataset directory.

    Args:
        path: Path to dataset directory containing case folders
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
    case_tasks = [(case_dir, visualize_density_map) for case_dir in cases]
    
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


