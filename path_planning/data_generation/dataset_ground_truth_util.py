from pathlib import Path
from typing import Dict

def generate_base_case_path(base_path: Path, case_id: int, road_map_type: str):
    case_path = base_path / f"case_{case_id}"
    case_path.mkdir(parents=True, exist_ok=True)
    map_path = generate_map_path(case_path, road_map_type)
    return case_path,map_path

def generate_map_path(base_path: Path, road_map_type: str):
    map_path = base_path / "maps" / road_map_type
    map_path.mkdir(parents=True, exist_ok=True)
    return map_path

def generate_perm_base_path(base_path: Path):
    perm_path = base_path / "perm"
    perm_path.mkdir(parents=True, exist_ok=True)
    return perm_path

def generate_input_perm_yaml_path(base_path: Path, perm_id: int):
    perm_path = generate_perm_base_path(base_path) / f"perm_{perm_id}"
    perm_path.mkdir(parents=True, exist_ok=True)
    perm_file = perm_path / "input.yaml"
    return perm_path,perm_file

def generate_mapf_path(base_path: Path, mapf_solver_name:str):
    mapf_path = base_path / mapf_solver_name
    mapf_path.mkdir(parents=True, exist_ok=True)
    return mapf_path

def generate_ground_truth_path(base_path: Path):
    ground_truth_path = base_path / "ground_truth"
    ground_truth_path.mkdir(parents=True, exist_ok=True)
    return ground_truth_path

def generate_roadmap_path(base_path: Path,roadmap_type:str):
    roadmap_path = base_path / roadmap_type
    roadmap_path.mkdir(parents=True, exist_ok=True)
    return roadmap_path

def generate_sample_base_path(base_path: Path):
    sample_base_path = base_path / "sample"
    sample_base_path.mkdir(parents=True, exist_ok=True)
    return sample_base_path

def generate_sample_path(base_path: Path, sample_id: int, augmentation_id: int = 0):
    sample_path = base_path/ f"graph_{sample_id}_{augmentation_id}"
    sample_path.mkdir(parents=True, exist_ok=True)
    return sample_path

def get_input_file_path(base_path: Path):
    input_file = base_path / "input.yaml"
    return input_file

def get_start_goal_file(base_path: Path):
    start_goal_file = base_path /  'start_goal_locations.npy'
    return start_goal_file

def get_graph_file_path(base_path: Path,graph_file_name:str =None):
    if graph_file_name is None:
        graph_file_name = "graph_map.pkl"
    graph_file = base_path / graph_file_name
    return graph_file

def get_graph_gnn_file_path(base_path: Path):
    graph_gnn_file = base_path / "graph.npz"
    return graph_gnn_file

def get_graph_gnn_file_path(base_path: Path):
    graph_gnn_file = base_path / "graph.npz"
    return graph_gnn_file

def get_target_file_path(base_path: Path, y_type_name: str):
    target_file = base_path / "target_{y_type_name}.npy"
    return target_file

def get_solution_file_path(base_path: Path, solution_name_suffix: str = "solution_graph_map", agent_velocity: float = 0.0):
    solution_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}.yaml"
    return solution_file

def get_config_file_path(base_path: Path):
    config_file = base_path / "config.yaml"
    return config_file

def get_path_visualization_file(base_path: Path, solution_name_suffix: str = "solution_graph_map", agent_velocity: float = 0.0):
    path_visualization_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}_path.png"
    return path_visualization_file

def get_heatmap_visualization_file(base_path: Path, solution_name_suffix: str = "solution_graph_map", agent_velocity: float = 0.0):
    heatmap_visualization_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}_heatmap.png"
    return heatmap_visualization_file

def get_path_animation_file(base_path: Path, solution_name_suffix: str = "solution_graph_map", agent_velocity: float = 0.0):
    path_animation_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}_path.gif"
    return path_animation_file

def get_trajectory_map_file(base_path: Path, solution_name_suffix: str = "solution_graph_map", agent_velocity: float = 0.0):
    trajectory_map_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}_trajectory_map.npy"
    return trajectory_map_file

def get_obstacle_map_file(base_path: Path, solution_name_suffix: str = "solution_graph_map", agent_velocity: float = 0.0):
    obstacle_map_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}_obstacle_map.npy"
    return obstacle_map_file

def get_density_map_file(base_path: Path, solution_name_suffix: str = "solution_graph_map", agent_velocity: float = 0.0):
    density_map_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}_density_map.npy"
    return density_map_file

def get_density_map_visualization_file(base_path: Path, solution_name_suffix: str = "solution_graph_map", agent_velocity: float = 0.0):
    density_map_visualization_file = base_path / f"{solution_name_suffix}_velocity{agent_velocity}_density_map.png"
    return density_map_visualization_file

def get_solution_name_suffix(graph_file: Path = None):
    if graph_file is None:
        solution_name_suffix = "solution_graph_map"
    else:
        solution_name_suffix = "solution_" + graph_file.stem
    return solution_name_suffix

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

