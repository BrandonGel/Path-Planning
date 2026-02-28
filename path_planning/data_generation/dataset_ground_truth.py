"""
Ground Truth Dataset generation for MAPF training.
Generates random MAPF instances and solves them using CBS.
Modified from the original implementation to work with the new common environment.

author: Brandon Ho
original author: Victor Retamal
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

def shuffle_agents_goals(inpt: Dict, agent_goal_index: List[int]) -> Dict:
    """
    Shuffle the goals of the agents.
    """
    agents = copy.deepcopy(inpt["agents"])
    start_goals = [agent["start"].copy() for agent in agents] + [
        agent["goal"].copy() for agent in agents
    ]
    for i in range(len(agents)):
        agents[i]["start"] = start_goals[agent_goal_index[i]]
        agents[i]["goal"] = start_goals[agent_goal_index[i + len(agents)]]
    return agents

def gen_input(
    bounds: List[List[float]],
    nb_obstacles: int,
    nb_agents: int,
    road_map_type: str,
    discrete_space: bool,
    num_samples: int,
    num_neighbors: int,
    min_edge_len: float,
    max_edge_len: float,
    resolution: float = 1.0,
    max_iterations: int = 10000,
    time_limit: int = 60,
    ) -> Optional[Dict]:
    """
    Generate random MAPF instance with agents and obstacles.

    Args:
        dimensions: (width, height) of the grid
        nb_obstacles: Number of obstacles to place
        nb_agents: Number of agents

    Returns:
        Dictionary with agent and map configuration, or None if generation fails
    """
    map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
    dimensions = list(map_.shape)
    input_dict = {
        "map": {
            "dimensions": dimensions,
            "bounds": bounds,
            "resolution": resolution,
            "obstacles": [],
        },
        "agents": [],        
        "time_limit": time_limit,
        "max_iterations": max_iterations,
        "road_map_type": road_map_type,
        "discrete_space": discrete_space,
        "num_samples": num_samples,
        "num_neighbors": num_neighbors,
        "min_edge_len": min_edge_len,
        "max_edge_len": max_edge_len,
    }

    total_cells = dimensions[0] * dimensions[1]
    if 0 < nb_obstacles < 1:
        nb_obstacles = int(total_cells * nb_obstacles)
    required_cells = nb_obstacles + 2 * nb_agents  # obstacles + starts + goals

    if required_cells > total_cells * 0.9:
        print(
            f"Warning: Requesting {required_cells} positions in {total_cells} cells (>90% fill)"
        )

    # Use set for O(1) lookup
    occupied_positions = set()

    def get_random_position(
        exclude_set: set, max_attempts: int = 1000
    ) -> Optional[Tuple[int, int]]:
        """Get random position not in exclude set."""
        # For sparse boards, use random sampling
        if len(exclude_set) < total_cells * 0.7:
            for _ in range(max_attempts):
                pos = (
                    np.random.randint(0, dimensions[0]),
                    np.random.randint(0, dimensions[1]),
                )
                if pos not in exclude_set:
                    return pos
        else:
            # For dense boards, sample from available positions
            all_positions = {
                (x, y) for x in range(dimensions[0]) for y in range(dimensions[1])
            }
            available = list(all_positions - exclude_set)
            if available:
                return available[np.random.randint(0, len(available))]

        return None

    # Place obstacles
    obstacles = []
    for _ in range(nb_obstacles):
        obs_pos = get_random_position(occupied_positions)
        if obs_pos is None:
            print(
                f"Failed to place all obstacles (placed {len(obstacles)}/{nb_obstacles})"
            )
            return None

        obstacles.append(obs_pos)
        occupied_positions.add(obs_pos)
        input_dict["map"]["obstacles"].append(obs_pos)

    # Place agents
    for agent_id in range(nb_agents):
        # Get start position
        start_pos = get_random_position(occupied_positions)
        if start_pos is None:
            print(f"Failed to place agent {agent_id} start position")
            return None
        occupied_positions.add(start_pos)

        # Get goal position (can overlap with other goals but not starts/obstacles)
        goal_pos = get_random_position(occupied_positions)
        if goal_pos is None:
            print(f"Failed to place agent {agent_id} goal position")
            return None
        occupied_positions.add(goal_pos)

        input_dict["agents"].append(
            {
                "start": list(start_pos),
                "goal": list(goal_pos),
                "name": f"agent{agent_id}",
            }
        )

    return input_dict

def data_gen(
    input_dict: Dict,
    output_path: Path,
    mapf_solver_config: dict = None,
    ) -> bool:
    """
    Generate solution for given MAPF instance.

    Args:
        input_dict: MAPF instance configuration
        output_path: Path to save solution

    Returns:
        True if solution found, False otherwise
    """
    output_path.mkdir(parents=True, exist_ok=True)

    param = input_dict
    obstacles = np.array(param["map"]["obstacles"])
    bounds = param["map"]["bounds"]
    resolution = param["map"]["resolution"]
    agents = param["agents"]
    road_map_type = param["road_map_type"]

    if road_map_type == 'grid':
        map_ = GraphSampler(
            bounds=bounds, resolution=resolution, start=[], goal=[], use_discrete_space=True
        )
        map_.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=1e-10, max_edge_len=1.1)
    elif road_map_type == 'prm' or road_map_type == 'planar' or road_map_type == 'rrg':
        map_ = GraphSampler(
            bounds=bounds,
            resolution=resolution,
            start=[],
            goal=[],
            use_discrete_space=param["discrete_space"],
            sample_num=param["num_samples"],
            min_edge_len=param["min_edge_len"],
            max_edge_len=param["max_edge_len"],
            num_neighbors=param["num_neighbors"]
        )
    else:
        print("Invalid road map name provided")
    
    map_.set_obstacle_map(obstacles)
    map_.inflate_obstacles(radius=0)

    start = [agent["start"] for agent in agents]
    goal = [agent["goal"] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)

    if road_map_type == 'grid':
        nodes = map_.generateRandomNodes(generate_grid_nodes=True)
        road_map = map_.generate_roadmap(nodes)
    elif road_map_type == 'prm':
        nodes = map_.generateRandomNodes()
        road_map = map_.generate_roadmap(nodes)
    elif road_map_type == 'planar':
        nodes = map_.generateRandomNodes()
        road_map = map_.generate_planar_map(nodes)
    elif road_map_type == 'rrg':
        # TODO: Implement RRG generation
        nodes = map_.generateRandomNodes()
        road_map = map_.generate_rrg(nodes)
    else:
        print("Invalid road map name provided")
    

    start = [s.current for s in map_.get_start_nodes()]
    goal = [g.current for g in map_.get_goal_nodes()]
    agents = [
        {"start": start[i], "name": agent["name"], "goal": goal[i]}
        for i, agent in enumerate(agents)
    ]

    solution_summary = solve_mapf(map_, agents, mapf_solver_config)

    agent_radius = mapf_solver_config.get("agent_radius", 0.0)
    agent_velocity = mapf_solver_config.get("agent_velocity", 0.0)

    solution_file = output_path / f"solution_radius{agent_radius}_velocity{agent_velocity}.yaml"
    with open(solution_file, "w") as f:
        yaml.safe_dump(solution_summary, f)

    # Write input parameters file
    parameters_file = output_path / "input.yaml"
    with open(parameters_file, "w") as f:
        yaml.safe_dump(param, f)

    return True

def process_single_permutation(args: Tuple, delete_failed_path: bool = True) -> Tuple[bool, int]:
    """
    Process a single permutation in parallel.

    Args:
        args: Tuple of (inpt_dict, agent_goal_index, case_path, perm_id, seed)

    Returns:
        Tuple of (success: bool, perm_id: int)
    """
    (
        inpt,
        mapf_path,
        mapf_solver_config,
    ) = args

    # Generate solution
    success = data_gen(
        inpt,
        mapf_path,
        mapf_solver_config=mapf_solver_config,
    )

    if not success and mapf_path.exists() and delete_failed_path:
        shutil.rmtree(mapf_path)

    return success

def process_single_case(args: Tuple) -> Tuple[float, float, int]:
    """
    Process a single case in parallel (including all its permutations).

    Args:
        args: Tuple of (case_id, path, config, all)

    Returns:
        Tuple of (successful_ratio, failed_ratio, case_id)
    """
    case_id, path, config, all = args

    # Generate random instance
    inpt = gen_input(
        bounds=config.get("bounds", [[0,32.0],[0,32.0]]),
        nb_obstacles=config.get("nb_obstacles", 0.1),
        nb_agents=config.get("nb_agents", 4),
        resolution=config.get("resolution", 1.0),
        max_iterations=config.get("max_iterations", 10000),
        time_limit=config.get("time_limit", 60),
        road_map_type = config.get("road_map_type", "grid"),
        discrete_space = config.get("use_discrete_space", True),
        num_samples = config.get("num_samples", 0),
        num_neighbors = config.get("num_neighbors", 4.0),
        min_edge_len = config.get("min_edge_len", 1e-10),
        max_edge_len = config.get("max_edge_len", 1 + 1e-10)
    )
    if inpt is None:
        return 0.0, 1.0, case_id
    
    case_path = path / f"case_{case_id}" 
    roadmap_path = case_path / config.get("road_map_type", "grid") 
    roadmap_path.mkdir(parents=True, exist_ok=True)
    if not (roadmap_path / "input.yaml").exists():
        with open(roadmap_path / "input.yaml", "w") as f:
            yaml.safe_dump(inpt, f)
    else:
        if not all:
            with open(roadmap_path / "input.yaml", "r") as f:
                inpt = yaml.load(f, Loader=yaml.FullLoader)

    ground_truth_path = roadmap_path / "ground_truth"
    ground_truth_path.mkdir(parents=True, exist_ok=True)

    # Prepare permutation generation
    start_goal_index = np.arange(0, config["nb_agents"]*2, 1)
    agent_start_goals = [tuple(agent["start"]) for agent in inpt["agents"]] + [tuple(agent["goal"]) for agent in inpt["agents"]]
    agent_start_goals = dict(zip(agent_start_goals, range(len(agent_start_goals))))
    nb_permutations = config["nb_permutations"]
    nb_permutations_tries = config.get("nb_permutations_tries", nb_permutations * 2)

    # Ensure tries >= permutations
    if nb_permutations_tries < nb_permutations:
        print(f"Warning: nb_permutations_tries ({nb_permutations_tries}) < nb_permutations ({nb_permutations})")
        print(f"Setting nb_permutations_tries to {nb_permutations * 2}")
        nb_permutations_tries = nb_permutations * 2

    # Check max permutations
    max_permutations = math.factorial(2 * config["nb_agents"])
    if nb_permutations > max_permutations:
        print(f"Warning: Requested {nb_permutations} permutations, but only {max_permutations} possible")
        nb_permutations = max_permutations
    
    if nb_permutations <= 0:
        assert False, "Number of permutations must be greater than 0"

    # Track unique permutations (as tuples of agent indices)
    unique_permutations = set()
    successful_permutations = 0
    failed_permutations = 0
    total_attempts = 0
    perm_id = 0
    mapf_solver_name = config.get("mapf_solver_name", "cbs")
    agent_radius = config.get("agent_radius", 0.0)
    agent_velocity = config.get("agent_velocity", 0.0)
    delete_failed_path = config.get("delete_failed_path", True)

    perm_ids_unfinished = []
    for perm_id in range(nb_permutations):
        perm_path = ground_truth_path / f"perm_{perm_id}"
        mapf_path = perm_path / mapf_solver_name
        solution_file = mapf_path / f"solution_radius{agent_radius}_velocity{agent_velocity}.yaml"
        if not solution_file.exists():
            perm_ids_unfinished.append(perm_id)
        else:
            with open(mapf_path / "input.yaml", "r") as f:
                perm_input = yaml.load(f, Loader=yaml.FullLoader)
            starts = [tuple(agent["start"]) for agent in perm_input["agents"]]
            goals = [tuple(agent["goal"]) for agent in perm_input["agents"]]
            perm_start_goals = starts + goals
            unique_permutations.add(tuple([agent_start_goals[pos] for pos in perm_start_goals]))
    successful_permutations = len(unique_permutations)
    print(f"Case {case_id}: Started with {nb_permutations-len(perm_ids_unfinished)}/{nb_permutations} permutations (max {nb_permutations_tries} attempts)")

    # Generate permutations until we have enough successes or max attempts reached
    while successful_permutations < nb_permutations and total_attempts < nb_permutations_tries or (not delete_failed_path and total_attempts < nb_permutations_tries + len(perm_ids_unfinished)):
        # Generate unique permutation
        max_unique_attempts = 1000
        unique_attempts = 0
        while tuple(start_goal_index) in unique_permutations:
            start_goal_index = np.random.permutation(start_goal_index)
            unique_attempts += 1
            if unique_attempts >= max_unique_attempts:
                print(f"Case {case_id}: Cannot generate more unique permutations after {unique_attempts} attempts")
                break
        
        # If we couldn't find a unique permutation, stop
        if unique_attempts >= max_unique_attempts:
            break
        
        # Record this permutation
        unique_permutations.add(tuple(start_goal_index))
        
        # Verify start/goal uniqueness
        agents_shuffled = shuffle_agents_goals(inpt, start_goal_index)
        starts = [tuple(agent["start"]) for agent in agents_shuffled]
        goals = [tuple(agent["goal"]) for agent in agents_shuffled]
        

        # Try to generate solution for this permutation
        inpt_copy = copy.deepcopy(inpt)
        inpt_copy["agents"] = agents_shuffled
        
        if len(perm_ids_unfinished) > 0:
            perm_id = perm_ids_unfinished[0]
        else:
            perm_id = total_attempts % nb_permutations
        mapf_solver_config= {
            'mapf_solver_name': config.get("mapf_solver_name", "cbs"),
            'time_limit': config.get("time_limit", 60),
            'agent_radius': config.get("agent_radius", 0.0),
            'agent_velocity': config.get("agent_velocity", 0.0),
            'max_iterations': config.get("max_iterations", 10000),
            'heuristic_type': config.get("heuristic_type", "manhattan"),
        }
        mapf_path =ground_truth_path / f"perm_{perm_id}" / mapf_solver_config['mapf_solver_name']
        task = (inpt_copy, mapf_path, mapf_solver_config)
        success = process_single_permutation(task, delete_failed_path=delete_failed_path)
        
        total_attempts += 1
        
        if success:
            successful_permutations += 1
            if len(perm_ids_unfinished) > 0:
                perm_ids_unfinished.pop(0)
        else:
            failed_permutations += 1
        
        # Shuffle for next iteration
        start_goal_index = np.random.permutation(start_goal_index)
            

    # Summary
    print(f"Case {case_id}: Completed with {successful_permutations}/{nb_permutations} successful permutations")
    print(f"Case {case_id}: Total attempts: {total_attempts}, Failed: {failed_permutations}")

    successful_ratio = successful_permutations / nb_permutations if nb_permutations > 0 else 0.0
    failed_ratio = 1.0 - successful_ratio
    
    return successful_ratio, failed_ratio, case_id

def create_solutions(path: Path, num_cases: int, config: Dict, all:bool, num_workers: int = cpu_count()):
    """
    Create multiple MAPF instances and their solutions with parallel processing.

    Args:
        path: Base path for dataset
        num_cases: Number of cases to generate
        config: Configuration dictionary
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Generating solutions for {num_cases} cases")

    set_global_seed(config.get("seed", 42))
    
    # Prepare case tasks
    case_tasks = []
    for i in range(0, num_cases):
        task = (i, path, config, all)
        case_tasks.append(task)

    # Process cases in parallel
    successful = 0
    failed = 0

    if num_workers > 1 and len(case_tasks) > 1:
        with Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(process_single_case, case_tasks),
                total=len(case_tasks),
                desc="Generating cases",
            ):
                results.append(result)

        # Aggregate results
        for successful_ratio, failed_ratio, case_id in results:
            successful += successful_ratio
            failed += failed_ratio
    else:
        # Sequential fallback
        for task in tqdm(case_tasks, desc="Generating cases"):
            successful_ratio, failed_ratio, case_id = process_single_case(task)
            successful += successful_ratio
            failed += failed_ratio

    # Print summary
    total_permutations = num_cases * config.get("nb_permutations", 1)
    print(f"Generation complete: {successful:.2f} successful cases, {failed:.2f} failed cases")
    print(f"Total Successful permutations: {int(successful * total_permutations)}")
    print(f"Total Failed permutations: {int(failed * total_permutations)}")
    print(f"Cases stored in {path} f")

def create_path_parameter_directory(base_path: Path, config: Dict,dump_config: bool = True):
    """
    Create a path parameters file.
    """
    bounds = config.get("bounds", [[0,32.0],[0,32.0]])
    nb_agents = config.get("nb_agents", 4)
    nb_obstacles = config.get("nb_obstacles", 0.1)
    resolution = config.get("resolution", 1.0)
    str_bounds = ""
    for b in bounds:
        str_bounds += f"{b[1]-b[0]}x"
    str_bounds = str_bounds[:-1]
    path = (
        base_path
        / f"map{str_bounds}_resolution{resolution}"
        / f"agents{nb_agents}_obst{nb_obstacles}"
    )
    os.makedirs(path, exist_ok=True)

    if dump_config:
        with open(path / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)
    return path
