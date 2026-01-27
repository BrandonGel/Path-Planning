from typing import Any


import python_motion_planning as pmp
from python_motion_planning.common import Grid, TYPES
from path_planning.common.environment.map.graph_sampler import GraphSampler
import numpy as np
import yaml
import os
def convert_to_pixel(x,y):
    pixel_x = int(np.round(x))
    pixel_y = int(np.round(-1 - y))
    return pixel_x, pixel_y


def convert_from_pixel(pixel_x, pixel_y, ylen):
    """
    Convert pixel coordinates back to grid coordinates.
    Reverse of convert_to_pixel.
    Args:
        pixel_x: pixel x coordinate
        pixel_y: pixel y coordinate
    Returns:
        x, y: grid coordinates
    """
    x = pixel_x
    y = ylen - 1 - pixel_y
    return x, y

def convert_grid_to_yaml(env: pmp.common.Grid, agents: list = [], filename: str = None):
    """
    Convert a grid to a YAML file.
    Args:
        env: The environment to convert.
        filename: The filename to save the YAML file.
    Returns:
        None
    """
    dimensions = list(env.shape)
    
    # Convert bounds to list of lists with float values
    # Handle different possible formats of bounds
    if isinstance(env.bounds, np.ndarray):
        bounds = env.bounds.tolist()
    elif isinstance(env.bounds, (list, tuple)):
        bounds = [list(bound) if isinstance(bound, (list, tuple, np.ndarray)) else [bound] for bound in env.bounds]
    else:
        bounds = [[env.bounds]]
    
    # Ensure all values are floats
    bounds = [[float(b) for b in bound] for bound in bounds]

    # Convert the obstacle_map to . and T
    obstacles =  np.stack(np.where(env.type_map.data == TYPES.OBSTACLE)).T
    obstacles = obstacles.tolist()
    
    # Prepare YAML data structure
    yaml_data = {
        'map': {
            'dimensions': dimensions,
            'bounds': bounds,
            'resolution': env.resolution,
            'obstacles': obstacles
        },
        'agents': agents
    }
    
    # Write to YAML file
    if filename is None:
        filename = 'obstacle_map.yaml'
    elif not filename.endswith('.yaml'):
        filename = filename + '.yaml'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def read_grid_from_yaml(filename: str):
    """
    Read a YAML file and recreate a Grid environment.
    Args:
        filename: The filename of the YAML file to read.
    Returns:
        env: A Grid object with obstacles loaded from the YAML file.
    """

    with open(filename, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    dimensions = yaml_data['map']['dimensions'] if 'dimensions' in yaml_data['map'] else [0, 0]
    bounds = yaml_data['map']['bounds'] if 'bounds' in yaml_data['map'] else [[0, dim] for dim in dimensions]
    resolution = yaml_data['map']['resolution'] if 'resolution' in yaml_data['map'] else 1.0
    obstacles = np.array(yaml_data['map']['obstacles']) if 'obstacles' in yaml_data['map'] else []
    
    env = Grid(bounds=bounds, resolution=resolution)
    if len(dimensions) == 2:
        env.type_map[obstacles[:,0], obstacles[:,1]] = TYPES.OBSTACLE 
    elif len(dimensions) == 3:
        env.type_map[obstacles[:,0], obstacles[:,1], obstacles[:,2]] = TYPES.OBSTACLE 
    else:
        raise ValueError(f"Unsupported dimensions: {len(dimensions)}")
    return env

def read_graph_sampler_from_yaml(filename: str):
    """
    Read a YAML file and recreate a GraphSampler environment.
    Args:
        filename: The filename of the YAML file to read.
    Returns:
        env: A GraphSampler object with obstacles loaded from the YAML file.
    """

    with open(filename, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
    dimensions = yaml_data['map']['dimensions'] if 'dimensions' in yaml_data['map'] else [0, 0]
    bounds = yaml_data['map']['bounds'] if 'bounds' in yaml_data['map'] else [[0, dim] for dim in dimensions]
    resolution = yaml_data['map']['resolution'] if 'resolution' in yaml_data['map'] else 1.0
    obstacles = np.array(yaml_data['map']['obstacles']) if 'obstacles' in yaml_data['map'] else []
    
    
    env = GraphSampler(bounds=bounds, resolution=resolution,start=[],goal=[])
    if len(dimensions) == 2:
        env.type_map[obstacles[:,0], obstacles[:,1]] = TYPES.OBSTACLE 
    elif len(dimensions) == 3:
        env.type_map[obstacles[:,0], obstacles[:,1], obstacles[:,2]] = TYPES.OBSTACLE 
    else:
        raise ValueError(f"Unsupported dimensions: {len(dimensions)}")
    return env

def read_map_from_yaml(filename: str):
    """
    Read a YAML file and recreate a GraphSampler environment.
    Args:
        filename: The filename of the YAML file to read.
    Returns:
        env: A GraphSampler object with obstacles loaded from the YAML file.
    """

    with open(filename, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    map = yaml_data['map']
    return map

def read_agents_from_yaml(filename: str):
    """
    Read a YAML file and recreate a list of agents.
    Args:
        filename: The filename of the YAML file to read.
    Returns:
        agents: A list of agents.
    """
    with open(filename, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_data['agents']

def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_to_yaml(obj, filename: str):
    if len(os.path.dirname(filename)) > 0 and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    output_serializable = to_builtin(obj)
    with open(filename, "w") as f:
        yaml.safe_dump(output_serializable, f, sort_keys=False)