from pathlib import Path
from typing import Dict

def generate_base_case_path(base_path: Path, case_id: int, road_map_type: str):
    case_path = base_path / f"case_{case_id}"
    case_path.mkdir(parents=True, exist_ok=True)
    map_path = base_path / f"case_{case_id}" / "maps"/ road_map_type
    map_path.mkdir(parents=True, exist_ok=True)
    return case_path,map_path

def generate_input_perm_yaml_path(base_path: Path, perm_id: int):
    perm_path = base_path /  f"perm_{perm_id}"
    perm_path.mkdir(parents=True, exist_ok=True)
    perm_file = perm_path / "input.yaml"
    return perm_path,perm_file

def generate_mapf_path(base_path: Path, perm_id: int,mapf_solver_name:str):
    mapf_path = base_path / f"perm_{perm_id}" / mapf_solver_name
    mapf_path.mkdir(parents=True, exist_ok=True)
    return mapf_path

def get_input_file_path(base_path: Path):
    input_file = base_path / "input.yaml"
    return input_file

def get_graph_file_path(base_path: Path):
    graph_file = base_path /  "graph_sampler.pkl"
    return graph_file

def get_solution_file_path(base_path: Path, solution_name_suffix: str = "solution", agent_velocity: float = 0.0):
    solution_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}.yaml"
    return solution_file

def get_config_file_path(base_path: Path):
    config_file = base_path / "config.yaml"
    return config_file

def generate_base_path(base_path: Path, config: Dict):
    bounds = config.get("bounds", [[0,32.0],[0,32.0]])
    nb_agents = config.get("nb_agents", 4)
    nb_obstacles = config.get("nb_obstacles", 0.1)
    resolution = config.get("resolution", 1.0)
    agent_radius = config.get("agent_radius", 0.0)
    str_bounds = ""
    for b in bounds:
        str_bounds += f"{b[1]-b[0]}x"
    str_bounds = str_bounds[:-1]
    path = base_path/ f"map{str_bounds}_resolution{resolution}"/ f"agents{nb_agents}_obst{nb_obstacles}"/ f"radius{agent_radius}"
    path.mkdir(parents=True, exist_ok=True)
    return path
