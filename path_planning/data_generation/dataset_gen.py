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
from path_planning.common.environment.map.graph_sampler import GraphSampler
from python_motion_planning.common import TYPES
import math

def shuffle_agents_goals(inpt: Dict,agent_goal_index: List[int]) -> Dict:
    """
    Shuffle the goals of the agents.
    """
    agents = inpt["agents"]
    start_goals = [agent["start"] for agent in agents] + [agent["goal"] for agent in agents]
    for i in range(len(agents)):
        agents[i]["start"] = start_goals[agent_goal_index[i]]
        agents[i]["goal"] = start_goals[agent_goal_index[i+len(agents)]]
    return agents

def gen_input(
    bounds: List[List[float]],
    nb_obstacles: int,
    nb_agents: int,
    resolution: float = 1.0,
    max_attempts: int = 10000
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
    dimensions = [int((bounds[i][1] - bounds[i][0]) / resolution) for i in range(len(bounds))]
    input_dict = {
        "map": {
            "dimensions": dimensions,
            'bounds': bounds,
            'resolution': resolution,
            "obstacles": []
        },
        "agents": []
    }

    total_cells = dimensions[0] * dimensions[1]
    if 0 < nb_obstacles < 1:
        nb_obstacles = int(total_cells * nb_obstacles)
    required_cells = nb_obstacles + 2 * nb_agents  # obstacles + starts + goals

    if required_cells > total_cells * 0.9:
        print(f"Warning: Requesting {required_cells} positions in {total_cells} cells (>90% fill)")

    # Use set for O(1) lookup
    occupied_positions = set()

    def get_random_position(exclude_set: set, max_attempts: int = 1000) -> Optional[Tuple[int, int]]:
        """Get random position not in exclude set."""
        # For sparse boards, use random sampling
        if len(exclude_set) < total_cells * 0.7:
            for _ in range(max_attempts):
                pos = (
                    np.random.randint(0, dimensions[0]),
                    np.random.randint(0, dimensions[1])
                )
                if pos not in exclude_set:
                    return pos
        else:
            # For dense boards, sample from available positions
            all_positions = {(x, y) for x in range(dimensions[0]) for y in range(dimensions[1])}
            available = list(all_positions - exclude_set)
            if available:
                return available[np.random.randint(0, len(available))]

        return None

    # Place obstacles
    obstacles = []
    for _ in range(nb_obstacles):
        obs_pos = get_random_position(occupied_positions, max_attempts=max_attempts)
        if obs_pos is None:
            print(f"Failed to place all obstacles (placed {len(obstacles)}/{nb_obstacles})")
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
        goal_occupied = occupied_positions - {s for s in occupied_positions if s in obstacles}
        goal_pos = get_random_position(occupied_positions)
        if goal_pos is None:
            print(f"Failed to place agent {agent_id} goal position")
            return None
        occupied_positions.add(goal_pos)

        input_dict["agents"].append({
            "start": list(start_pos),
            "goal": list(goal_pos),
            "name": f"agent{agent_id}"
        })

    return input_dict


def data_gen(input_dict: Dict, output_path: Path,use_discrete_space: bool = True) -> bool:
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
    map_ = GraphSampler(bounds=bounds, resolution=resolution,start=[],goal=[],use_discrete_space=use_discrete_space)
    map_.type_map[obstacles[:,0], obstacles[:,1]] = TYPES.OBSTACLE 

    map_.inflate_obstacles(radius=0)
    map_.set_parameters(sample_num=0, num_neighbors=4.0, min_edge_len=0.0, max_edge_len=1.1)

    start = [agent['start'] for agent in agents]
    goal = [agent['goal'] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    nodes = map_.generateRandomNodes(generate_grid_nodes = True)
    road_map = map_.generate_roadmap(nodes)

    start = [s.current for s in map_.get_start_nodes()]
    goal = [g.current for g in map_.get_goal_nodes()]
    agents =[
         {
            "start": start[i],
            "name": agent['name'],
            "goal": goal[i]
        }
        for i, agent in enumerate(agents)
    ]
    env = Environment(map_, agents)

    # Search for solution
    cbs = CBS(env)
    solution = cbs.search()

    if not solution:
        print(f"No solution found for case {output_path.name}")
        return False

    # Write solution file
    output = {
        "schedule": solution,
        "cost": env.compute_solution_cost(solution)
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
    Create multiple MAPF instances and their solutions.

    Args:
        path: Base path for dataset
        num_cases: Number of cases to generate
        config: Configuration dictionary
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Count existing cases
    existing_cases = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")]
    cases_ready = len(existing_cases) if existing_cases else 0

    print(f"Generating solutions (starting from case {cases_ready})")

    successful = 0
    failed = 0

    for i in range(cases_ready, num_cases):
        if i % 25 == 0:
            print(f"Solution -- [{i}/{num_cases}] (Success: {successful}, Failed: {failed})")

        # Generate random instance
        inpt = gen_input(
            config["bounds"],
            config["nb_obstacles"],
            config["nb_agents"],
            config["resolution"],
        )

        if inpt is None:
            failed += 1
            continue

        # Generate and save solution
       
        successful_permutations = 0
        failed_permutations = 0
        agent_goal_index = np.arange(0,config["nb_agents"]*2,1)
        permutations = set[Any]()
        if config["nb_permutations"] > math.factorial(2*config["nb_agents"]):
            print(f"Warning: Requesting {config["nb_permutations"]} positions in {math.factorial(2*config["nb_agents"])} cells")
            print(f"Setting nb_permutations to {math.factorial(2*config["nb_agents"])}")
            config["nb_permutations"] = math.factorial(2*config["nb_agents"])
        for _ in range(config["nb_permutations"]):
            case_path = path / f"case_{i}"/f"perm_{successful_permutations}"  
            while tuple[Any, ...](agent_goal_index) in permutations:
                agent_goal_index = np.random.permutation(agent_goal_index)
            agents = shuffle_agents_goals(inpt,agent_goal_index)
            inpt["agents"] = agents

            if data_gen(inpt, case_path):
                successful_permutations += 1
            else:
                failed_permutations += 1
                # Remove failed case directory
                if case_path.exists():
                    import shutil
                    shutil.rmtree(case_path)

            permutations.add(tuple(agent_goal_index))
            agent_goal_index = np.random.permutation(agent_goal_index)

            
        
        successful += successful_permutations/config["nb_permutations"]
        failed += failed_permutations/config["nb_permutations"]

        

    print(f"Generation complete: {successful} successful, {failed} failed")
    print(f"Cases stored in {path}")

def create_path_parameter_directory(base_path: Path, config: Dict):
    """
    Create a path parameters file.
    """
    bounds = config["bounds"]
    nb_agents = config["nb_agents"]
    nb_obstacles = config["nb_obstacles"]
    resolution = config["resolution"]
    str_bounds ="" 
    for b in bounds:
        str_bounds+=f"{b[1]-b[0]}x"
    str_bounds = str_bounds[:-1]
    path = base_path / f"map{str_bounds}_resolution{resolution}" / f"agents{nb_agents}_obst{nb_obstacles}"
    os.makedirs(path, exist_ok=True)

    with open(path / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    return path

