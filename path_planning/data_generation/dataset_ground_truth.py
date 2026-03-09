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
import math
from itertools import product


def _to_native_yaml(obj):
    """Convert numpy scalars/arrays in nested structures to native Python for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _to_native_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native_yaml(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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
    sample_num: int,
    num_neighbors: int,
    min_edge_len: float,
    max_edge_len: float,
    resolution: float = 1.0,
    max_iterations: int = 10000,
    time_limit: int = 60,
    agent_radius: float = 0.0,
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
    # TODO: need to modify it for resolution other than 1.0
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
        "sample_num": sample_num,
        "num_neighbors": num_neighbors,
        "min_edge_len": min_edge_len,
        "max_edge_len": max_edge_len,
        "agent_radius": agent_radius,
        "resolution": resolution,
    }
    total_cells = int(math.prod(dimensions))
    num_dims = len(dimensions)
    num_cells_to_inflate = int(agent_radius/0.5)
    # no_resolution_dimensions = [int(dim*resolution) for dim in dimensions]
    # total_cells = int(math.prod(no_resolution_dimensions)) 
    # resolution_cells = int(1/resolution)
    # resolution_cells_shape = (resolution_cells,) * num_dims
    # num_resolution_cells = resolution_cells**num_dims
    # total_cells = int(math.prod(no_resolution_dimensions)) 
    if 0 < nb_obstacles < 1:
        nb_obstacles = int(total_cells * nb_obstacles)
    required_cells = nb_obstacles + 2 * nb_agents  # obstacles + starts + goals

    if required_cells > total_cells * 0.9:
        print(
            f"Warning: Requesting {required_cells} positions in {total_cells} cells (>90% fill)"
        )

    # Use set for O(1) lookup
    occupied_positions = set()

    # TODO: Only for 2D cases
    def get_random_position(
        exclude_set: set, max_attempts: int = 1000,
    ) -> Optional[Tuple[int, int]]:
        """Get random position not in exclude set."""
        # For sparse boards, use random sampling
        if len(exclude_set) < total_cells * 0.7:
            for _ in range(max_attempts):
                pos = tuple(np.random.randint(0, dimensions[ii]) for ii in range(num_dims))
                if pos not in exclude_set:
                    return pos
        else:
            # For dense boards, sample from available positions
            all_positions = set(product(*(range(d) for d in dimensions)))
            available = list(all_positions - exclude_set)
            if available:
                return available[np.random.randint(0, len(available))]

        return None

    # Place obstacles
    obstacles = []
    # cell_indices = np.unravel_index(np.arange(num_resolution_cells),resolution_cells_shape )
    for _ in range(nb_obstacles):
        obs_pos = get_random_position(occupied_positions)
        if obs_pos is None:
            print(
                f"Failed to place all obstacles (placed {len(obstacles)}/{nb_obstacles})"
            )
            return None
        
        occupied_positions.add(obs_pos)
        obstacles.append(obs_pos)
        input_dict["map"]["obstacles"].append(obs_pos)
        # for i in range(num_resolution_cells):
        #     resolution_obs_pos = tuple(int(cell_indices[j][i] + obs_pos[j]//resolution) for j in range(num_dims))
        #     obstacles.append(resolution_obs_pos)
        #     input_dict["map"]["obstacles"].append(resolution_obs_pos)
    
    if agent_radius > 0 and num_cells_to_inflate > 0:
        offset_range = range(-num_cells_to_inflate, num_cells_to_inflate + 1)
        for obs_pos in obstacles:
            for offset in product(*(offset_range for _ in range(num_dims))):
                inflated_obs_pos = tuple(obs_pos[j] + offset[j] for j in range(num_dims))
                occupied_positions.add(inflated_obs_pos)
            


    # Place agents
    for agent_id in range(nb_agents):
        # Get start position
        start_pos = get_random_position(occupied_positions)
        if start_pos is None:
            print(f"Failed to place agent {agent_id} start position")
            return None
        occupied_positions.add(start_pos)
        if agent_radius > 0 and num_cells_to_inflate > 0:
            offset_range = range(-num_cells_to_inflate, num_cells_to_inflate + 1)
            for offset in product(*(offset_range for _ in range(num_dims))):
                inflated_obs_pos = tuple(start_pos[j] + offset[j] for j in range(num_dims))
                occupied_positions.add(inflated_obs_pos)

        # Get goal position (can overlap with other goals but not starts/obstacles)
        goal_pos = get_random_position(occupied_positions)
        if goal_pos is None:
            print(f"Failed to place agent {agent_id} goal position")
            return None
        occupied_positions.add(goal_pos)
        if agent_radius > 0 and num_cells_to_inflate > 0:
            offset_range = range(-num_cells_to_inflate, num_cells_to_inflate + 1)
            for offset in product(*(offset_range for _ in range(num_dims))):
                inflated_obs_pos = tuple(goal_pos[j] + offset[j] for j in range(num_dims))
                occupied_positions.add(inflated_obs_pos)

        start_pos = tuple(int(start_pos[j]*resolution) for j in range(num_dims))
        goal_pos = tuple(int(goal_pos[j]*resolution) for j in range(num_dims))
        input_dict["agents"].append(
            {
                "start": list(start_pos),
                "goal": list(goal_pos),
                "name": f"agent{agent_id}",
            }
        )

    return input_dict

def create_map(param: Dict, generate_new_graph: bool = False,graph_file: Path ="graph_sampler.pkl" ):
    bounds = param["map"]["bounds"]
    resolution = param["map"]["resolution"]
    if graph_file.exists() and not generate_new_graph:
        map_ = GraphSampler(bounds=bounds, resolution=resolution, start=[], goal=[])
        map_.load_graph_sampler(graph_file)
    else:
        obstacles = np.array(param["map"]["obstacles"])
        agents = param["agents"]
        road_map_type = param["road_map_type"]
        if road_map_type == 'grid':
            map_ = GraphSampler(
                bounds=bounds, resolution=resolution, start=[], goal=[], use_discrete_space=True,
                sample_num=0,
                min_edge_len=1e-10,
                max_edge_len=(1+1e-10)*param['resolution'],
                num_neighbors=4.0
            )
        elif road_map_type == 'prm' or road_map_type == 'planar' or road_map_type == 'rrg':
            map_ = GraphSampler(
                bounds=bounds,
                resolution=resolution,
                start=[],
                goal=[],
                use_discrete_space=param["discrete_space"],
                sample_num=param["sample_num"],
                min_edge_len=param["min_edge_len"],
                max_edge_len=param["max_edge_len"],
                num_neighbors=param["num_neighbors"]
            )
        else:
            print("Invalid road map name provided")
        
        map_.set_obstacles(obstacles=obstacles)
        map_.set_inflation_radius(radius=param["agent_radius"]+np.sqrt(2)/2*param["resolution"])

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
        map_.save_graph_sampler(graph_file)
    return map_

def process_single_case_map_generation(args: Tuple) -> Optional[int]:
    """
    Generate map and permutations for a single case (no solving).
    Saves input.yaml, graph_sampler.pkl, and case-level agents/perm_{id}/agents.yaml.
    Used so all MAPF solvers can later run on the same map and task config.

    Args:
        args: Tuple of (case_id, path, config)

    Returns:
        case_id on success, None on failure
    """
    case_id, path, config,generate_new_graph,graph_file = args
    set_global_seed(config.get("seed", 42) + case_id)
    case_path = path / f"case_{case_id}"
    roadmap_path = case_path / config.get("road_map_type", "grid")/ f'radius{config.get("agent_radius", 0.0)}'
    roadmap_path.mkdir(parents=True, exist_ok=True)
    input_file = roadmap_path / "input.yaml"

    if generate_new_graph or not input_file.exists():
        inpt = gen_input(
            bounds=config.get("bounds", [[0, 32.0], [0, 32.0]]),
            nb_obstacles=config.get("nb_obstacles", 0.1),
            nb_agents=config.get("nb_agents", 4),
            resolution=config.get("resolution", 1.0),
            max_iterations=config.get("max_iterations", 10000),
            time_limit=config.get("time_limit", 60),
            road_map_type=config.get("road_map_type", "grid"),
            discrete_space=config.get("use_discrete_space", True),
            sample_num=config.get("sample_num", 0),
            num_neighbors=config.get("num_neighbors", 4.0),
            min_edge_len=config.get("min_edge_len", 1e-10),
            max_edge_len=config.get("max_edge_len", 1 + 1e-10),
            agent_radius=config.get("agent_radius", 0.0),
        )
        with open(input_file, "w") as f:
            yaml.safe_dump(_to_native_yaml(inpt), f)
    else:
        with open(input_file, "r") as f:
            inpt = yaml.load(f, Loader=yaml.FullLoader)
        if inpt is None:    
            print(f"Failed to load input for case {case_id}")
            assert False, f"Failed to load input for case {case_id}"
    ground_truth_path = roadmap_path / "ground_truth"
    ground_truth_path.mkdir(parents=True, exist_ok=True)
    if graph_file is None:
        graph_file = ground_truth_path / "graph_sampler.pkl"

    # Build and save graph once
    create_map(inpt, generate_new_graph,graph_file)

    # Generate and save permutation agent configs
    nb_permutations = config["nb_permutations"]
    nb_permutations_tries = config.get("nb_permutations_tries", nb_permutations * 2)
    max_permutations = math.factorial(2 * config["nb_agents"])
    if nb_permutations > max_permutations:
        nb_permutations = max_permutations

    start_goal_index = np.arange(0, config["nb_agents"] * 2, 1)
    unique_permutations = set()
    total_attempts = 0

    while len(unique_permutations) < nb_permutations and total_attempts < nb_permutations_tries:
        max_unique_attempts = 1000
        unique_attempts = 0
        while tuple(start_goal_index) in unique_permutations:
            start_goal_index = np.random.permutation(start_goal_index)
            unique_attempts += 1
            if unique_attempts >= max_unique_attempts:
                break
        if unique_attempts >= max_unique_attempts:
            break

        unique_permutations.add(tuple(start_goal_index))
        agents_shuffled = shuffle_agents_goals(inpt, start_goal_index)

        perm_id = len(unique_permutations) - 1
        agents_dir = case_path / "agents"
        perm_path = agents_dir / f"radius{config.get('agent_radius', 0.0)}" / f"perm_{perm_id}"
        agents_file = perm_path / "agents.yaml"
        perm_path.mkdir(parents=True, exist_ok=True)
        with open(agents_file, "w") as f:
            yaml.safe_dump(_to_native_yaml({"agents": agents_shuffled}), f)

        total_attempts += 1
        start_goal_index = np.random.permutation(start_goal_index)

    return case_id

def create_maps(path: Path, num_cases: int, config: Dict, num_workers: int = cpu_count(), generate_new_graph: bool = False) -> None:
    """
    Generate maps and permutations for all cases (no solving).
    Saves input.yaml, graph_sampler.pkl, and case-level agents/perm_{id}/agents.yaml per case
    so create_solutions can later run all solvers on the same maps.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    case_tasks = [(i, path, config, generate_new_graph,None) for i in range(num_cases)]

    if num_workers > 1 and len(case_tasks) > 1:
        with Pool(processes=num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(process_single_case_map_generation, case_tasks),
                total=len(case_tasks),
                desc="Generating maps",
            ))
    else:
        for task in tqdm(case_tasks, desc="Generating maps"):
            process_single_case_map_generation(task)

def process_single_permutation(args: Tuple, delete_failed_path: bool = True) -> Tuple[bool, int]:
    """
    Process a single permutation in parallel.

    Args:
        args: Tuple of (inpt_dict, agent_goal_index, case_path, perm_id, seed)

    Returns:
        Tuple of (success: bool, perm_id: int)
    """
    (
        param,
        map_,
        mapf_path,
        mapf_solver_config,
        solution_name_suffix
    ) = args

    # Generate solution
    mapf_path.mkdir(parents=True, exist_ok=True)
    agents = [
        {"start": tuple(agent['start']), "name": agent["name"], "goal": tuple(agent['goal'])}
        for i, agent in enumerate(param["agents"])
    ]

    solution_summary = solve_mapf(map_, agents, mapf_solver_config)
    success = solution_summary["success"]

    agent_velocity = round(mapf_solver_config.get("agent_velocity", 0.0), 2)

    # Write input parameters file
    parameters_file = mapf_path / "input.yaml"
    if not os.path.exists(parameters_file):
        with open(parameters_file, "w") as f:
            yaml.safe_dump(_to_native_yaml(param), f)

    solution_file = mapf_path / f"{solution_name_suffix}_velocity{round(agent_velocity, 2)}.yaml"
    with open(solution_file, "w") as f:
        yaml.safe_dump(_to_native_yaml(solution_summary), f)

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
    case_id, path, config, all, graph_file,verbose = args
    case_path = path / f"case_{case_id}"
    set_global_seed(config.get("seed", 42) + case_id)
    agent_radius = round(config.get("agent_radius", 0.0), 3)
    roadmap_path = case_path / config.get("road_map_type", "grid") / f"radius{agent_radius}"
    roadmap_path.mkdir(parents=True, exist_ok=True)
    input_file = roadmap_path / "input.yaml"
    ground_truth_path = roadmap_path / "ground_truth"
    graph_sampler_exists = ground_truth_path.exists() and (ground_truth_path / "graph_sampler.pkl").exists()
    delete_failed_path = config.get("delete_failed_path", False)

    assert input_file.exists(), f"Input file {input_file} does not exist"
    with open(input_file, "r") as f:
        inpt = yaml.load(f, Loader=yaml.FullLoader)
    ground_truth_path.mkdir(parents=True, exist_ok=True)

    # Prepare permutation generation
    start_goal_index = np.arange(0, config["nb_agents"]*2, 1)
    agent_start_goals = [tuple(agent["start"]) for agent in inpt["agents"]] + [tuple(agent["goal"]) for agent in inpt["agents"]]
    agent_start_goals = dict(zip(agent_start_goals, range(len(agent_start_goals))))
    nb_permutations = config["nb_permutations"]
    nb_permutations_tries = config.get("nb_permutations_tries", nb_permutations * 2)
    nb_permutations_tries = nb_permutations_tries if delete_failed_path else nb_permutations

    # Ensure tries >= permutations
    if nb_permutations_tries < nb_permutations:
        print(f"Warning: nb_permutations_tries ({nb_permutations_tries}) < nb_permutations ({nb_permutations})")
        print(f"Setting nb_permutations_tries to {nb_permutations * 2}")
        nb_permutations_tries = nb_permutations * 2 if delete_failed_path else nb_permutations

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
    agent_radius = round(config.get("agent_radius", 0.0), 3)
    agent_velocity = round(config.get("agent_velocity", 0.0), 2)
    delete_failed_path = config.get("delete_failed_path", False)
    if graph_file is None:
        graph_file = ground_truth_path / "graph_sampler.pkl"
        solution_name_suffix = "solution"
    else:
        solution_name_suffix = "solution_" + graph_file.stem

    perm_ids_unfinished = []
    if graph_sampler_exists:
        for perm_id in range(nb_permutations):
            perm_path = ground_truth_path / f"perm_{perm_id}"
            mapf_path = perm_path / mapf_solver_name
            # Use the same rounded naming convention as when writing solution files
            solution_file = mapf_path / f"{solution_name_suffix}_velocity{round(agent_velocity, 2)}.yaml"
            if not solution_file.exists():
                perm_ids_unfinished.append(perm_id)
            else:
                # Count as successful (same perm already solved)
                try:
                    with open(mapf_path / "input.yaml", "r") as f:
                        perm_input = yaml.load(f, Loader=yaml.FullLoader)
                    starts = [tuple(agent["start"]) for agent in perm_input["agents"]]
                    goals = [tuple(agent["goal"]) for agent in perm_input["agents"]]
                    perm_start_goals = starts + goals
                    unique_permutations.add(tuple([agent_start_goals[pos] for pos in perm_start_goals]))
                except Exception:
                    pass
    successful_permutations = len(unique_permutations)
    if verbose:
        print(f"Case {case_id}: Started with {nb_permutations-len(perm_ids_unfinished)} permutations (max {nb_permutations_tries} attempts)")
    map_ = create_map(inpt, graph_file = graph_file)

    mapf_solver_config = {
        "mapf_solver_name": config.get("mapf_solver_name", "cbs"),
        "time_limit": config.get("time_limit", 60),
        "agent_radius": config.get("agent_radius", 0.0),
        "agent_velocity": config.get("agent_velocity", 0.0),
        "max_iterations": config.get("max_iterations", 10000),
        "heuristic_type": config.get("heuristic_type", "manhattan"),
    }

    while (successful_permutations < nb_permutations and total_attempts < nb_permutations_tries): 

        # Determine perm_id first so agents_file and mapf_path use the correct permutation
        if len(perm_ids_unfinished) > 0:
            perm_id = perm_ids_unfinished[0]
        else:
            perm_id = total_attempts % nb_permutations

        max_unique_attempts = 1000
        unique_attempts = 0

        agents_file = case_path / "agents" /f"radius{agent_radius}" / f"perm_{perm_id}" / "agents.yaml"
        if agents_file.exists():
            with open(agents_file, "r") as f:
                agents_shuffled = yaml.load(f, Loader=yaml.FullLoader)["agents"]
        else:
            # Generate a new (unique) agent permutation for this perm_id, then persist it.
            while unique_attempts < max_unique_attempts:
                start_goal_index = np.random.permutation(start_goal_index)
                perm_key = tuple(start_goal_index.tolist())
                if perm_key not in unique_permutations:
                    unique_permutations.add(perm_key)
                    agents_shuffled = shuffle_agents_goals(inpt, start_goal_index)
                    break
                unique_attempts += 1
            else:
                raise RuntimeError(
                    f"Unable to generate a unique permutation for {case_path} after {max_unique_attempts} attempts"
                )

            perm_path = agents_file.parent
            perm_path.mkdir(parents=True, exist_ok=True)
            with open(agents_file, "w") as f:
                yaml.safe_dump(_to_native_yaml({"agents": agents_shuffled}), f)

        inpt_copy = copy.deepcopy(inpt)
        inpt_copy["agents"] = agents_shuffled

        mapf_path = ground_truth_path / f"perm_{perm_id}" / mapf_solver_config["mapf_solver_name"]
        task = (inpt_copy, map_, mapf_path, mapf_solver_config, solution_name_suffix)
        success = process_single_permutation(task, delete_failed_path=delete_failed_path)

        total_attempts += 1
        if success:
            successful_permutations += 1
        else:
            failed_permutations += 1
        if (not delete_failed_path or not success) and len(perm_ids_unfinished) > 0:
            perm_ids_unfinished.pop(0)

        start_goal_index = np.random.permutation(start_goal_index)

    # Summary
    if verbose:
        print(f"Case {case_id}: Completed with {successful_permutations}/{nb_permutations} successful permutations")
        print(f"Case {case_id}: Total attempts: {total_attempts}, Failed: {failed_permutations}")

    successful_ratio = successful_permutations / nb_permutations if nb_permutations > 0 else 0.0
    failed_ratio = 1.0 - successful_ratio
    
    return successful_ratio, failed_ratio, case_id

def create_solutions(path: Path, num_cases: int, config: Dict, all:bool, num_workers: int = cpu_count(),graph_files: Optional[List[Path]] = None, verbose: bool = True):
    """
    Create multiple MAPF instances and their solutions with parallel processing.

    Args:
        path: Base path for dataset
        num_cases: Number of cases to generate
        config: Configuration dictionary
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Generating solutions for {num_cases} cases")
    
    if graph_files is not None and len(graph_files) > 0:
        assert len(graph_files) == num_cases, "Number of graph files must match number of cases"

    # Prepare case tasks
    case_tasks = []
    for i in range(0, num_cases):
        if graph_files is not None:
            graph_file = graph_files[i]
        else:
            graph_file = None
        task = (i, path, config, all, graph_file,verbose)
        case_tasks.append(task)

    # Process cases in parallel
    successful = 0
    failed = 0
    solver_name = config.get("mapf_solver_name", "cbs")    
    agent_velocity = round(config.get("agent_velocity", 0.0), 2)
    solution_file_name = f"solution_velocity{agent_velocity}.yaml"
    if num_workers > 1 and len(case_tasks) > 1:
        with Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(process_single_case, case_tasks),
                total=len(case_tasks),
                desc=f"{solver_name} {solution_file_name}",
            ):
                results.append(result)

        # Aggregate results
        for successful_ratio, failed_ratio, case_id in results:
            successful += successful_ratio
            failed += failed_ratio
    else:
        # Sequential fallback
        for task in tqdm(case_tasks, desc=f"{solver_name} {solution_file_name}"):
            successful_ratio, failed_ratio, case_id = process_single_case(task)
            successful += successful_ratio
            failed += failed_ratio

    # Print summary
    total_permutations = num_cases * config.get("nb_permutations", 1)
    if verbose:
        print(f"Generation complete: {successful:.2f} successful cases, {failed:.2f} failed cases")
        print(f"Total Successful permutations: {int(successful * total_permutations)}")
        print(f"Total Failed permutations: {int(failed * total_permutations)}")
        print(f"Cases stored in {path} for {num_cases} cases")

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
            yaml.safe_dump(_to_native_yaml(config), f)
    return path
