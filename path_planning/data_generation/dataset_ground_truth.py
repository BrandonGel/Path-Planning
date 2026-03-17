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
from path_planning.utils.util import _to_native_yaml
from path_planning.data_generation.dataset_ground_truth_util import generate_base_path,generate_base_case_path, generate_input_perm_yaml_path, generate_mapf_path, get_input_file_path, get_graph_file_path, get_solution_file_path, get_config_file_path
from path_planning.data_generation.dataset_ground_truth_util import get_input_file_path, get_graph_file_path, get_solution_file_path, get_config_file_path

KEYS = ["bounds", "resolution", "time_limit", "max_iterations", "road_map_type", "use_discrete_space", "sample_num", "num_neighbors", "min_edge_len", "max_edge_len", "agent_radius", "nb_obstacles", "nb_agents"]
class InputFile:   
    def __init__(self, input_file: Path, case_id: int = 0):
        self.input_file = input_file
        self.case_id = case_id
        
    def check_keys_inside_input(self, inpt: Dict):
        for key in KEYS:
            if key in inpt:
                return False
            for subkey in inpt.keys():
                if type(inpt[subkey]) == dict and key in inpt[subkey]:  
                    return False
        print(f"Key {key} not found in input file {self.input_file}")
        return key
        
    def check_input_file(self):
        if not self.input_file.exists():
            return False
        with open(self.input_file, "r") as f:
            inpt = yaml.load(f, Loader=yaml.FullLoader)
        if inpt is None:
            print(f"Failed to load input for case {self.case_id}")
            return False
        key = self.check_keys_inside_input(inpt)    
        if key:
            print(f"Key {key} not found in input file {self.input_file}")
            return False
        return inpt

    def gen_input(self,**kwargs):
        key = self.check_keys_inside_input(kwargs)
        if key:
            print(f"Key {key} not found in kwargs")
            assert False, f"Key {key} not found in kwargs"
        bounds = kwargs.get("bounds", [[0, 32.0], [0, 32.0]])
        resolution = kwargs.get("resolution", 1.0)
        time_limit = kwargs.get("time_limit", 60)
        max_iterations = kwargs.get("max_iterations", 10000)
        road_map_type = kwargs.get("road_map_type", "grid")
        discrete_space = kwargs.get("discrete_space", True)
        sample_num = kwargs.get("sample_num", 0)
        num_neighbors = kwargs.get("num_neighbors", 4.0)
        min_edge_len = kwargs.get("min_edge_len", 1e-10)
        max_edge_len = kwargs.get("max_edge_len", 1 + 1e-10)
        agent_radius = kwargs.get("agent_radius", 0.0)
        nb_obstacles = kwargs.get("nb_obstacles", 0.1)
        nb_agents = kwargs.get("nb_agents", 4)
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
                assert False, f"Failed to place obstacle {_} (placed {len(obstacles)}/{nb_obstacles})"
            
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
                assert False, f"Failed to place agent {agent_id} start position"
            occupied_positions.add(start_pos)
            if agent_radius > 0 and num_cells_to_inflate > 0:
                offset_range = range(-num_cells_to_inflate, num_cells_to_inflate + 1)
                for offset in product(*(offset_range for _ in range(num_dims))):
                    inflated_obs_pos = tuple(start_pos[j] + offset[j] for j in range(num_dims))
                    occupied_positions.add(inflated_obs_pos)

            # Get goal position (can overlap with other goals but not starts/obstacles)
            goal_pos = get_random_position(occupied_positions)
            if goal_pos is None:
                assert False, f"Failed to place agent {agent_id} goal position"
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
            map_.generate_roadmap(nodes)
        elif road_map_type == 'prm':
            nodes = map_.generateRandomNodes()
            map_.generate_roadmap(nodes)
        elif road_map_type == 'planar':
            nodes = map_.generateRandomNodes()
            map_.generate_planar_map(nodes)
        elif road_map_type == 'rrg':
            # TODO: Implement RRG generation
            nodes = map_.generateRandomNodes()
            map_.generate_rrg(nodes)
        else:
            print("Invalid road map name provided")
        map_.save_graph_sampler(graph_file)
    return map_

def generate_permutation(inpt: Dict, config: Dict, case_path: Path):
    # Generate and save permutation agent configs
    nb_permutations = config["nb_permutations"]
    nb_permutations_tries = config["nb_permutations_tries"]
    solve_till_success = config["solve_till_success"]
    start_goal_index = np.arange(0, config["nb_agents"] * 2, 1)
    unique_permutations = set()

    num_tries = nb_permutations_tries if solve_till_success else nb_permutations
    for ii in range(num_tries):
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
        inpt_copy = copy.deepcopy(inpt)
        inpt_copy["agents"] = agents_shuffled

        perm_id = len(unique_permutations) - 1
        _,perm_file = generate_input_perm_yaml_path(case_path, perm_id)
        with open(perm_file, "w") as f:
            yaml.safe_dump(_to_native_yaml(inpt_copy), f)
        start_goal_index = np.random.permutation(start_goal_index)

def process_single_case_map_generation(args: Tuple) -> Optional[int]:
    """
    Generate map and permutations for a single case (no solving).
    Saves input.yaml, graph_sampler.pkl, and case-level agents
    Used so all MAPF solvers can later run on the same map and task config.

    Args:
        args: Tuple of (case_id, path, config)

    Returns:
        case_id on success, None on failure
    """
    case_id, path, config,generate_new_graph,graph_file,verbose = args
    set_global_seed(config.get("seed", 42) + case_id)
    road_map_type = config.get("road_map_type", "grid")
    case_path,map_path = generate_base_case_path(path, case_id, road_map_type)

    input_file = get_input_file_path(case_path)
    input_class = InputFile(input_file, case_id)
    inpt = input_class.check_input_file()
    if generate_new_graph or not inpt:
        inpt = input_class.gen_input(**config)
        if verbose:
            print(f"Generated input file {input_file}")
        with open(input_file, "w") as f:
            yaml.safe_dump(_to_native_yaml(inpt), f)   
    if graph_file is None:
        graph_file = get_graph_file_path(map_path)

    # Build and save graph once
    create_map(inpt, generate_new_graph,graph_file)

    # Generate and save permutation agent configs
    generate_permutation(inpt, config, case_path)

    return case_id

def create_maps(path: Path, num_cases: int, config: Dict, num_workers: int = cpu_count(), generate_new_graph: bool = False, verbose: bool = True) -> None:
    """
    Generate maps and permutations for all cases (no solving).
    Saves input.yaml, graph_sampler.pkl, and case-level agents per case
    so create_solutions can later run all solvers on the same maps.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    case_tasks = [(i, path, config, generate_new_graph,None,verbose) for i in range(num_cases)]
    if verbose:
        nb_permutations = config["nb_permutations"]
        print(f"Generating maps for {num_cases} cases with {num_workers} workers for {nb_permutations} permutations per case")
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

def process_single_permutation(args: Tuple) -> Tuple[bool, int]:
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
    agent_velocity = mapf_solver_config["agent_velocity"]

    # Write input parameters file
    parameters_file = get_input_file_path(mapf_path)
    if not os.path.exists(parameters_file):
        with open(parameters_file, "w") as f:
            yaml.safe_dump(_to_native_yaml(param), f)

    solution_file = get_solution_file_path(mapf_path, solution_name_suffix, agent_velocity)
    with open(solution_file, "w") as f:
        yaml.safe_dump(_to_native_yaml(solution_summary), f)
    return success

def process_single_case(args: Tuple) -> Tuple[float, float, int]:
    """
    Process a single case in parallel (including all its permutations).

    Args:
        args: Tuple of (case_id, path, config, graph_file,verbose)

    Returns:
        Tuple of (successful_ratio, failed_ratio, case_id)
    """
    case_id, path, config, graph_file,verbose = args
    set_global_seed(config.get("seed", 42) + case_id)
    road_map_type = config.get("road_map_type", "grid")
    case_path,map_path = generate_base_case_path(path, case_id, road_map_type)   
    
    graph_sampler_exists = get_graph_file_path(map_path).exists()
    assert graph_sampler_exists, f"Graph sampler {graph_sampler_exists} does not exist"
    solve_till_success = config.get("solve_till_success", False)

    input_file = get_input_file_path(case_path)
    assert input_file.exists(), f"Input file {input_file} does not exist"
    with open(input_file, "r") as f:
        inpt = yaml.load(f, Loader=yaml.FullLoader)

    # Read through permutations already generated
    nb_permutations = config["nb_permutations"]
    nb_permutations_tries = config['nb_permutations_tries']
    solve_till_success = config.get("solve_till_success", False)
    num_tries = nb_permutations_tries if solve_till_success else nb_permutations

    # Track unique permutations (as tuples of agent indices)
    num_success = 0
    total_attempts = 0
    perm_id = 0
    mapf_solver_name = config["mapf_solver_name"]
    agent_velocity = config["agent_velocity"]
    if graph_file is None:
        graph_file = get_graph_file_path(map_path)
        solution_name_suffix = "solution"
    else:
        solution_name_suffix = "solution_" + graph_file.stem

    perm_ids_unfinished = []
    num_attempts = 0
    for perm_id in range(num_tries):
        mapf_path = generate_mapf_path(case_path, perm_id, mapf_solver_name)
        solution_file = get_solution_file_path(mapf_path, solution_name_suffix, agent_velocity)
        if not solution_file.exists():
            perm_ids_unfinished.append(perm_id)
        else:
            with open(solution_file, "r") as f:
                solution_summary = yaml.load(f, Loader=yaml.FullLoader)
            if solution_summary["success"]:
                num_success += 1
        if num_success >= nb_permutations:
            num_attempts = perm_id + 1
            break

    map_ = create_map(inpt, graph_file = graph_file)

    mapf_solver_config = {
        "mapf_solver_name": config.get("mapf_solver_name", "cbs"),
        "time_limit": config.get("time_limit", 60),
        "agent_radius": config.get("agent_radius", 0.0),
        "agent_velocity": config.get("agent_velocity", 0.0),
        "max_iterations": config.get("max_iterations", 10000),
        "heuristic_type": config.get("heuristic_type", "manhattan"),
    }
    
    if num_success < nb_permutations:
        for perm_id in perm_ids_unfinished:
            _,perm_file = generate_input_perm_yaml_path(case_path, perm_id)
            assert perm_file.exists(), f"Permutation file {perm_file} does not exist"
            with open(perm_file, "r") as f:
                agents_shuffled = yaml.load(f, Loader=yaml.FullLoader)["agents"]

            inpt_copy = copy.deepcopy(inpt)
            inpt_copy["agents"] = agents_shuffled

            mapf_path = generate_mapf_path(case_path, perm_id, mapf_solver_config["mapf_solver_name"])
            task = (inpt_copy, map_, mapf_path, mapf_solver_config, solution_name_suffix)
            success = process_single_permutation(task)

            total_attempts += 1
            if success:
                num_success += 1
            if num_success >= nb_permutations:
                break
        num_attempts = perm_id + 1
        if verbose:
            if len(perm_ids_unfinished) > 0:
                print(f"Case {case_id}: Completed {num_success}/{nb_permutations} successful permutations, starting from permutation {num_tries-len(perm_ids_unfinished)} with {num_attempts} attempts")
            else:
                print(f"Case {case_id}: Completed {num_success}/{nb_permutations} successful permutations with {num_attempts} attempts")
    else:
        if verbose:
            print(f"Case {case_id}: Already Completed {num_success}/{nb_permutations} with {num_attempts} attempts")
    
    return num_success, num_attempts, case_id

def create_solutions(path: Path, num_cases: int, config: Dict,  num_workers: int = cpu_count(),graph_files: Optional[List[Path]] = None, verbose: bool = True):
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
        print(f"Generating solutions for {num_cases} cases with max {config['nb_permutations_tries']} attempts")
    
    if graph_files is not None and len(graph_files) > 0:
        assert len(graph_files) == num_cases, "Number of graph files must match number of cases"

    # Prepare case tasks
    case_tasks = []
    for i in range(0, num_cases):
        if graph_files is not None:
            graph_file = graph_files[i]
        else:
            graph_file = None
        task = (i, path, config,  graph_file,verbose)
        case_tasks.append(task)

    # Process cases in parallel
    successful = 0
    attempts = 0
    solver_name = config.get("mapf_solver_name", "cbs")    
    agent_velocity = config.get("agent_velocity", 0.0)
    if num_workers > 1 and len(case_tasks) > 1:
        with Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(process_single_case, case_tasks),
                total=len(case_tasks),
                desc=f"{solver_name} {agent_velocity}",
            ):
                results.append(result)

        # Aggregate results
        for num_success, num_attempts, case_id in results:
            successful += num_success
            attempts += num_attempts
    else:
        # Sequential fallback
        for task in tqdm(case_tasks, desc=f"{solver_name} {agent_velocity}"):
            num_success, num_attempts, case_id = process_single_case(task)
            successful += num_success
            attempts += num_attempts

    # Print summary
    if verbose:
        success_ratio = successful / attempts
        failed_ratio = 1.0 - success_ratio
        print(f"Generation complete: {success_ratio:.2f} success, {failed_ratio:.2f} failure")
        print(f"Total Successful permutations: {successful}")
        print(f"Total Failed permutations: {attempts - successful}")
        print(f"Cases stored in {path} for {num_cases} cases")

def create_path_parameter_directory(base_path: Path, config: Dict,dump_config: bool = True):
    """
    Create a path parameters file.
    """
    bounds = config.get("bounds", [[0,32.0],[0,32.0]])
    nb_agents = config.get("nb_agents", 4)
    nb_obstacles = config.get("nb_obstacles", 0.1)
    resolution = config.get("resolution", 1.0)
    agent_radius = config.get("agent_radius", 0.0)
    map_path = generate_base_path(base_path, config)

    base_config = {
        "bounds": bounds,
        "nb_agents": nb_agents,
        "nb_obstacles": nb_obstacles,
        "resolution": resolution,
        "agent_radius": agent_radius,
    }
    if dump_config:
        with open(get_config_file_path(map_path), "w") as f:
            yaml.safe_dump(_to_native_yaml(base_config), f)
    return map_path
