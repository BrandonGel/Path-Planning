"""
Dataset generation for MAPF training.
Generates random MAPF instances and solves them using CBS.
Modified from the original implementation to work with the new common environment.

author: Brandon Ho
original author: Victor Retamal
"""

import os
from pathlib import Path
import yaml
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from path_planning.multi_agent_planner.centralized.icbs.icbs import IEnvironment, ICBS
from path_planning.common.environment.map.graph_sampler import GraphSampler
from python_motion_planning.common import TYPES
import math
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil


def shuffle_agents_goals(inpt: Dict, agent_goal_index: List[int]) -> Dict:
    """
    Shuffle the goals of the agents.
    """
    agents = inpt["agents"]
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
    resolution: float = 1.0,
    max_attempts: int = 10000,
) -> Optional[Dict]:
    """
    Generate random MAPF instance with agents and obstacles.

    Args:
        dimensions: (width, height) of the grid
        nb_obstacles: Number of obstacles to place
        nb_agents: Number of agents
        max_attempts: Maximum attempts for placement before giving up

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
        obs_pos = get_random_position(occupied_positions, max_attempts=max_attempts)
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


def process_single_permutation(args: Tuple) -> Tuple[bool, int]:
    """
    Process a single permutation in parallel.

    Args:
        args: Tuple of (inpt_dict, agent_goal_index, case_path, perm_id, seed)

    Returns:
        Tuple of (success: bool, perm_id: int)
    """
    (
        inpt,
        agent_goal_index,
        case_path,
        perm_id,
        seed,
        timeout,
        max_attempts,
        improved,
    ) = args

    # Set random seed for this worker process
    np.random.seed(seed)

    # Shuffle agents/goals
    agents = shuffle_agents_goals(inpt, agent_goal_index)
    inpt_copy = inpt.copy()
    inpt_copy["agents"] = agents

    # Generate solution
    perm_path = case_path / f"perm_{perm_id}"
    success = data_gen(
        inpt_copy,
        perm_path,
        improved=improved,
        timeout=timeout,
        max_attempts=max_attempts,
    )

    if not success and perm_path.exists():
        shutil.rmtree(perm_path)

    return success, perm_id


def process_single_case(args: Tuple) -> Tuple[float, float, int]:
    """
    Process a single case in parallel (including all its permutations).

    Args:
        args: Tuple of (case_id, path, config, seed)

    Returns:
        Tuple of (successful_ratio, failed_ratio, case_id)
    """
    case_id, path, config, seed = args

    # Set random seed for this worker process
    np.random.seed(seed)

    # Generate random instance
    inpt = gen_input(
        config["bounds"],
        config["nb_obstacles"],
        config["nb_agents"],
        config["resolution"],
    )
    timeout = config["timeout"]
    max_attempts = config["max_attempts"]
    if inpt is None:
        return 0.0, 1.0, case_id

    case_path = path / f"case_{case_id}"

    # Prepare permutation tasks
    permutation_tasks = []
    permutations = set()
    agent_goal_index = np.arange(0, config["nb_agents"] * 2, 1)

    # Check max permutations
    max_permutations = math.factorial(2 * config["nb_agents"])
    nb_permutations = config["nb_permutations"]
    if nb_permutations <= 0:
        assert False, "Number of permutations must be greater than 0"
    if nb_permutations > max_permutations:
        nb_permutations = max_permutations

    # Check for cbs vs. icbs
    improved = config["improved"]

    # Generate all unique permutations
    for perm_id in range(nb_permutations):
        attempts = 0
        while tuple(agent_goal_index) in permutations:
            agent_goal_index = np.random.permutation(agent_goal_index)
            attempts += 1
            if attempts > 1000:
                break

        permutations.add(tuple(agent_goal_index))
        perm_seed = np.random.randint(0, 2**31)
        task = (
            inpt.copy(),
            agent_goal_index.copy(),
            case_path,
            perm_id,
            perm_seed,
            timeout,
            max_attempts,
            improved,
        )
        permutation_tasks.append(task)
        agent_goal_index = np.random.permutation(agent_goal_index)

    # Process permutations sequentially
    # Note: Permutations are always processed sequentially within each case worker
    # to avoid nested multiprocessing complexity and daemon process errors.
    successful_permutations = 0
    failed_permutations = 0

    for task in permutation_tasks:
        success, _ = process_single_permutation(task)
        if success:
            successful_permutations += 1
        else:
            failed_permutations += 1

    successful_ratio = successful_permutations / nb_permutations
    failed_ratio = failed_permutations / nb_permutations

    return successful_ratio, failed_ratio, case_id


def data_gen(
    input_dict: Dict,
    output_path: Path,
    improved: str,
    timeout: int = 60,
    max_attempts: int = 10000,
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
    map_ = GraphSampler(
        bounds=bounds, resolution=resolution, start=[], goal=[], use_discrete_space=True
    )
    map_.set_obstacle_map(obstacles)

    map_.inflate_obstacles(radius=0)
    map_.set_parameters(
        sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1
    )

    start = [agent["start"] for agent in agents]
    goal = [agent["goal"] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    nodes = map_.generateRandomNodes(generate_grid_nodes=True)
    road_map = map_.generate_roadmap(nodes)

    start = [s.current for s in map_.get_start_nodes()]
    goal = [g.current for g in map_.get_goal_nodes()]
    agents = [
        {"start": start[i], "name": agent["name"], "goal": goal[i]}
        for i, agent in enumerate(agents)
    ]
    env = Environment(map_, agents)

    # TODO: call cbs and icbs here seperately
    if improved == "no":
        # Search for solution and measure runtime using cbs
        cbs = CBS(env, time_limit=timeout, max_iterations=max_attempts)
        start_time = time.time()
        solution = cbs.search()
        runtime = time.time() - start_time
    else:
        # Search for solution and measure runtime using icbs
        icbs = ICBS(env, time_limit=timeout, max_iterations=max_attempts)
        start_time = time.time()
        solution = icbs.search()
        runtime = time.time() - start_time

    if not solution:
        print(f"No solution found for case {output_path.name}")
        return False

    # Write solution file
    output = {
        "schedule": solution,
        "cost": env.compute_solution_cost(solution),
        "runtime": runtime,
    }

    solution_file = output_path / "solution.yaml"
    with open(solution_file, "w") as f:
        yaml.safe_dump(output, f)

    # Write input parameters file
    parameters_file = output_path / "input.yaml"
    with open(parameters_file, "w") as f:
        yaml.safe_dump(param, f)

    return True


def create_solutions(path: Path, num_cases: int, config: Dict):
    """
    Create multiple MAPF instances and their solutions with parallel processing.

    Args:
        path: Base path for dataset
        num_cases: Number of cases to generate
        config: Configuration dictionary
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Count existing cases
    existing_cases = [
        d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")
    ]
    cases_ready = len(existing_cases) if existing_cases else 0

    print(f"Generating solutions (starting from case {cases_ready})")

    # Get number of workers for cases
    num_workers = config.get("num_workers", cpu_count())

    # Prepare case tasks
    case_tasks = []
    for i in range(cases_ready, num_cases):
        seed = np.random.randint(0, 2**31)
        task = (i, path, config, seed)
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
    cases_processed = num_cases - cases_ready
    total_permutations = cases_processed * config["nb_permutations"]
    print(
        f"Generation complete: {successful:.2f} successful cases, {failed:.2f} failed cases"
    )
    print(f"Total Successful permutations: {int(successful * total_permutations)}")
    print(f"Total Failed permutations: {int(failed * total_permutations)}")
    print(f"Cases stored in {path}")


def create_path_parameter_directory(base_path: Path, config: Dict):
    """
    Create a path parameters file.
    """
    bounds = config["bounds"]
    nb_agents = config["nb_agents"]
    nb_obstacles = config["nb_obstacles"]
    resolution = config["resolution"]
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

    with open(path / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    return path
