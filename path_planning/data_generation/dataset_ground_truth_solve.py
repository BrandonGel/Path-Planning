"""
Ground Truth Dataset generation for MAPF training.
Generates random MAPF instances and solves them using CBS.
Modified from the original implementation to work with the new common environment.

author: Brandon Ho
original author: Victor Retamal
"""

from pathlib import Path
import yaml
from typing import Dict, Tuple
from path_planning.multi_agent_planner.mapf_solver import POINT_SOLVERS, solve_mapf
from path_planning.utils.util import set_global_seed
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import copy
from path_planning.utils.util import _to_native_yaml
from path_planning.data_generation.dataset_util import (
    generate_base_case_path,
    generate_input_perm_yaml_path,
    generate_mapf_path,
    generate_roadmap_path,
    get_graph_file_path,
    get_input_file_path,
    get_solution_file_path,
    get_solution_name_suffix,
)
from path_planning.data_generation.dataset_ground_truth_map import create_map

#: Defaults of the keys ``solve_mapf`` requires; every other config key (body, solver switches, ...) is
#: passed through as given.
SOLVER_DEFAULTS = {"mapf_solver_name": "cbs", "time_limit": 60, "agent_radius": 0.0, "agent_velocity": 0.0,
                   "max_iterations": 10000, "heuristic_type": "manhattan"}


def process_single_permutation(args: Tuple) -> bool:
    """Solve one permutation and write its solution summary. ``args``: ``(param, map_, roadmap_path,
    mapf_solver_config, solution_name_suffix)``; ``param["agents"]`` go to ``solve_mapf`` as they are
    (world frame, with any per-agent body or heading). Returns whether it was solved."""
    param, map_, roadmap_path, mapf_solver_config, solution_name_suffix = args
    roadmap_path.mkdir(parents=True, exist_ok=True)
    solution_summary = solve_mapf(map_, param["agents"], mapf_solver_config)
    solution_file = get_solution_file_path(roadmap_path, solution_name_suffix, mapf_solver_config["agent_velocity"])
    with open(solution_file, "w") as f:
        yaml.safe_dump(_to_native_yaml(solution_summary), f)
    return solution_summary["success"]

def process_single_case(args: Tuple) -> Tuple[int, int, int]:
    """Solve the permutations of one case until ``nb_permutations`` succeed (or the tries run out).
    ``args``: ``(case_id, path, config, graph_file_name, verbose)``. Returns ``(solved, attempts, case_id)``."""
    case_id, path, config, graph_file_name,verbose = args
    set_global_seed(config.get("seed", 42) + case_id)
    road_map_type = config.get("road_map_type", "grid")
    case_path,map_path = generate_base_case_path(path, case_id, road_map_type)   
    solve_till_success = config.get("solve_till_success", False)
    generate_new_graph = config.get("generate_new_graph", False)
    resolve_solution = config.get("resolve_solution", False)

    input_file = get_input_file_path(case_path)
    assert input_file.exists(), f"Input file {input_file} does not exist"
    with open(input_file, "r") as f:
        inpt = yaml.load(f, Loader=yaml.FullLoader)

    # Read through permutations already generated
    nb_permutations = config["nb_permutations"]
    nb_permutations_tries = config['nb_permutations_tries']
    num_tries = nb_permutations_tries if solve_till_success else nb_permutations

    mapf_solver_config = {**SOLVER_DEFAULTS, **config}
    mapf_solver_name = mapf_solver_config["mapf_solver_name"]
    agent_velocity = mapf_solver_config["agent_velocity"]
    num_success = 0

    solution_name_suffix = get_solution_name_suffix(graph_file_name)
    graph_file = get_graph_file_path(map_path,graph_file_name)

    perm_ids_unfinished = []
    num_attempts = 0
    for perm_id in range(num_tries):
        perm_path,_ = generate_input_perm_yaml_path(case_path, perm_id)
        mapf_path = generate_mapf_path(perm_path, mapf_solver_name)
        roadmap_path = generate_roadmap_path(mapf_path, road_map_type)
        solution_file = get_solution_file_path(roadmap_path, solution_name_suffix, agent_velocity)
        if not solution_file.exists() or generate_new_graph or resolve_solution:
            perm_ids_unfinished.append(perm_id)
        else:
            with open(solution_file, "r") as f:
                solution_summary = yaml.load(f, Loader=yaml.FullLoader)
            if solution_summary["success"]:
                num_success += 1
        if num_success >= nb_permutations:
            num_attempts = perm_id + 1
            break

    map_ = create_map(inpt, graph_file = graph_file,verbose=verbose,args={"use_collision_sweep": True})

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
            if process_single_permutation(task):
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

def create_solutions(path: Path, num_cases: int, config: Dict,  num_workers: int = cpu_count(),graph_files: list = None, verbose: bool = True):
    """Solve the cases under ``path`` (maps and permutations from ``create_maps``) with
    ``config["mapf_solver_name"]``, in parallel; one solution summary per permutation. ``graph_files``: per
    case, the name of the graph pickle to solve on (default: the case's ground-truth map). ``config`` is
    passed to ``solve_mapf`` (see ``SOLVER_DEFAULTS``)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    solver_config = {**SOLVER_DEFAULTS, **config}
    if solver_config["mapf_solver_name"] in POINT_SOLVERS and (
            solver_config["agent_radius"] > 0 or solver_config.get("agent") is not None
            or config.get("road_map_type", "grid") != "grid"):
        raise ValueError(f"{solver_config['mapf_solver_name']} plans point particles on a grid roadmap only "
                         "(agent_radius 0, no agent body, road_map_type 'grid')")
    if verbose:
        print(f"Generating solutions for {num_cases} cases with max {config['nb_permutations_tries']} attempts")
    
    # Prepare case tasks
    case_tasks = []
    for i in range(0, num_cases):
        graph_file= graph_files[i] if graph_files is not None else None
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
        success_ratio = successful / attempts if attempts else 0.0
        failed_ratio = 1.0 - success_ratio
        print(f"Generation complete: {success_ratio:.2f} success, {failed_ratio:.2f} failure")
        print(f"Total Successful permutations: {successful}")
        print(f"Total Failed permutations: {attempts - successful}")
        print(f"Cases stored in {path} for {num_cases} cases")