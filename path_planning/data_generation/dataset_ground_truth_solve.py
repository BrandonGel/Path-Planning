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
from path_planning.data_generation.dataset_util import *
from path_planning.data_generation.dataset_ground_truth_map import create_map

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
        roadmap_path,
        mapf_solver_config,
        solution_name_suffix
    ) = args

    # Generate solution
    roadmap_path.mkdir(parents=True, exist_ok=True)
    agents = [
        {"start": tuple(agent['start']), "name": agent["name"], "goal": tuple(agent['goal'])}
        for i, agent in enumerate(param["agents"])
    ]

    solution_summary = solve_mapf(map_, agents, mapf_solver_config)
    success = solution_summary["success"]
    agent_velocity = mapf_solver_config["agent_velocity"]

    # Write input parameters file
    parameters_file = get_input_file_path(roadmap_path)
    if not os.path.exists(parameters_file):
        with open(parameters_file, "w") as f:
            yaml.safe_dump(_to_native_yaml(param), f)

    solution_file = get_solution_file_path(roadmap_path, solution_name_suffix, agent_velocity)
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
    case_id, path, config, graph_file_name,verbose = args
    set_global_seed(config.get("seed", 42) + case_id)
    road_map_type = config.get("road_map_type", "grid")
    mapf_solver_name = config.get("mapf_solver_name", "cbs")
    case_path,map_path = generate_base_case_path(path, case_id, road_map_type)   
    solve_till_success = config.get("solve_till_success", False)
    generate_new_graph = config.get("generate_new_graph", False)

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

    solution_name_suffix = get_solution_name_suffix(graph_file_name)
    graph_file = get_graph_file_path(map_path,graph_file_name)

    perm_ids_unfinished = []
    num_attempts = 0
    for perm_id in range(num_tries):
        perm_path,_ = generate_input_perm_yaml_path(case_path, perm_id)
        mapf_path = generate_mapf_path(perm_path, mapf_solver_name)
        roadmap_path = generate_roadmap_path(mapf_path, road_map_type)
        solution_file = get_solution_file_path(roadmap_path, solution_name_suffix, agent_velocity)
        if not solution_file.exists() or generate_new_graph:
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

            perm_path,_ = generate_input_perm_yaml_path(case_path, perm_id)
            mapf_path = generate_mapf_path(perm_path, mapf_solver_name)
            roadmap_path = generate_roadmap_path(mapf_path, road_map_type)
            task = (inpt_copy, map_, roadmap_path, mapf_solver_config, solution_name_suffix)
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

def create_solutions(path: Path, num_cases: int, config: Dict,  num_workers: int = cpu_count(),graph_file_name: Path = None, verbose: bool = True):
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
    
    # Prepare case tasks
    case_tasks = []
    for i in range(0, num_cases):
        task = (i, path, config,  graph_file_name,verbose)
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