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
from typing import Any, Dict, List, Tuple, Optional
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
from path_planning.common.environment.map.graph_sampler import validate_roadmap_type

KEYS = ["bounds", "resolution", "time_limit", "max_iterations", "road_map_type", "use_discrete_space", "sample_num", "num_neighbors", "min_edge_len", "max_edge_len", "agent_radius", "nb_obstacles", "nb_agents"]
class InputFile:   
    def __init__(self, input_file: Path, case_id: int = 0):
        self.input_file = input_file
        self.case_id = case_id
        self.inpt = self.check_input_file()
        
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

    def get_input_content(self):
        return self.inpt

    def check_difference_between_input(self, config: Dict) -> bool:
        keys = ["bounds", "resolution", "road_map_type", "use_discrete_space", "sample_num", "num_neighbors", "min_edge_len", "max_edge_len", "agent_radius", "nb_obstacles", "nb_agents"]
        inpt = self.get_input_content()
        if not inpt:
            return True

        for key in keys:
            if inpt.get(key, None) is not None and config.get(key, None) is not None and inpt[key] != config[key]:
                return True
        return False

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
        use_discrete_space = kwargs.get("use_discrete_space", True)
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
            "use_discrete_space": use_discrete_space,
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

def create_map(param: Dict, generate_new_graph: bool = False,graph_file: Path =None,verbose: bool = True):
    bounds = param["map"]["bounds"]
    resolution = param["map"]["resolution"]
    if graph_file and graph_file.exists() and not generate_new_graph:
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
        elif validate_roadmap_type(road_map_type):
            map_ = GraphSampler(
                bounds=bounds,
                resolution=resolution,
                start=[],
                goal=[],
                use_discrete_space=param["use_discrete_space"],
                sample_num=param["sample_num"],
                min_edge_len=param["min_edge_len"],
                max_edge_len=param["max_edge_len"],
                num_neighbors=param["num_neighbors"]
            )
        else:
            assert False, f"Invalid road map name provided: {road_map_type}"
        
        map_.set_obstacles(obstacles=obstacles)
        map_.set_inflation_radius(radius=param["agent_radius"]+np.sqrt(2)/2*param["resolution"])

        start = [agent["start"] for agent in agents]
        goal = [agent["goal"] for agent in agents]
        map_.set_start(start)
        map_.set_goal(goal)

        if road_map_type == 'grid':
            nodes = map_.generateRandomNodes(generate_grid_nodes=True)
        else:
            nodes = map_.generateRandomNodes()
        map_.generate_map(road_map_type,nodes)
        map_.save_graph_sampler(graph_file)
        if verbose:
            print(f"Generated and saved graph to {graph_file}")
    return map_

def generate_permutation(inpt: Dict, config: Dict, case_path: Path, generate_new_graph: bool = False, verbose: bool = True):
    # Generate and save permutation agent configs
    nb_permutations = config["nb_permutations"]
    nb_permutations_tries = config["nb_permutations_tries"]
    solve_till_success = config.get("solve_till_success", False)
    start_goal_index = np.arange(0, config["nb_agents"] * 2, 1)
    unique_permutations = set()

    # Count how many permutation input files already exist on disk and, if we are
    # not regenerating a new graph, load any previously stored permutations so we
    # do not create duplicates across runs.
    perm_base = generate_perm_base_path(case_path)
    existing_count = 0
    if perm_base.exists():
        for p in perm_base.iterdir():
            if p.is_dir() and (p / "input.yaml").exists():
                existing_count += 1
                if not generate_new_graph:
                    try:
                        with open(p / "input.yaml", "r") as f:
                            existing_data = yaml.safe_load(f)
                    except Exception:
                        existing_data = None
                    if isinstance(existing_data, dict) and "start_goal_index" in existing_data:
                        try:
                            perm_tuple = tuple(int(v) for v in existing_data["start_goal_index"])
                            unique_permutations.add(perm_tuple)
                        except TypeError:
                            # If the stored value is not iterable/int-castable, just skip it.
                            pass

    # If we already have enough permutations and the graph is unchanged, do nothing.
    if existing_count >= nb_permutations and not generate_new_graph:
        return

    # Determine how many additional permutations we still need (if any).
    remaining = max(0, nb_permutations - existing_count)
    if remaining == 0 and not generate_new_graph:
        return

    # For a new graph, we start from scratch; otherwise we only add missing perms.
    start_perm_id = 0 if generate_new_graph else existing_count

    num_tries = nb_permutations_tries if solve_till_success else remaining
    for ii in range(num_tries):
        max_unique_attempts = 1000
        unique_attempts = 0
        while tuple(start_goal_index) in unique_permutations:
            start_goal_index = np.random.permutation(start_goal_index)
            unique_attempts += 1
            if unique_attempts >= max_unique_attempts:
                break
        if unique_attempts >= max_unique_attempts:
            if verbose:
                print(f"Failed to generate unique permutation after {max_unique_attempts} attempts")
            break

        unique_permutations.add(tuple(start_goal_index))
        agents_shuffled = shuffle_agents_goals(inpt, start_goal_index)
        inpt_copy = copy.deepcopy(inpt)
        inpt_copy["agents"] = agents_shuffled
        # Persist the permutation so future runs can avoid duplicates.
        try:
            inpt_copy["start_goal_index"] = start_goal_index.tolist()
        except AttributeError:
            inpt_copy["start_goal_index"] = list(start_goal_index)

        perm_id = start_perm_id + len(unique_permutations) - 1
        _,perm_file = generate_input_perm_yaml_path(case_path, perm_id)
        with open(perm_file, "w") as f:
            yaml.safe_dump(_to_native_yaml(inpt_copy), f)
        if verbose:
            print(f"Generated and saved permutation {perm_id} to {perm_file}")
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
    inpt = input_class.get_input_content()
    if graph_file is None:
        graph_file = get_graph_file_path(map_path)
    if generate_new_graph or not inpt or not graph_file.exists():
        inpt = input_class.gen_input(**config)
        if generate_new_graph or not graph_file.exists() or not input_file.exists():
            if verbose:
                print(f"Generated new input file {input_file}")
            with open(input_file, "w") as f:
                yaml.safe_dump(_to_native_yaml(inpt), f)   
    

    # Build and save graph once
    create_map(inpt, generate_new_graph,graph_file,verbose)

    # Generate and save permutation agent configs
    generate_permutation(inpt, config, case_path, generate_new_graph=generate_new_graph, verbose=verbose)

    return case_id

def create_maps(path: Path, num_cases: int, config: Dict, num_workers: int = cpu_count(), generate_new_graph: bool = False, verbose: bool = True,graph_files: list = None) -> None:
    """
    Generate maps and permutations for all cases (no solving).
    Saves input.yaml, graph_sampler.pkl, and case-level agents per case
    so create_solutions can later run all solvers on the same maps.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if graph_files and len(graph_files) != num_cases:
        case_tasks = [(i, path, config, generate_new_graph,graph_files[i],verbose) for i in range(num_cases)]
    else:
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
