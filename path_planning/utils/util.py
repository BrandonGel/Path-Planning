from typing import Any


import python_motion_planning as pmp
from python_motion_planning.common import Grid, TYPES
from path_planning.global_planner.sample_search.graph_sampler import GraphSampler
import numpy as np
import yaml

def convert_to_pixel(x,y):
    pixel_x = int(np.round(x))
    pixel_y = int(np.round(-1 - y))
    return pixel_x, pixel_y

def convert_grid_to_yaml(env: pmp.common.Grid, filename: str = None):
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
    obstacle =  np.stack(np.where(env.type_map.data == TYPES.OBSTACLE)).T
    obstacle = [list(point.tolist()) for point in obstacle]
    
    # Prepare YAML data structure
    yaml_data = {
        'dimensions': dimensions,
        'bounds': bounds,
        'resolution': env.resolution,
        'obstacle': obstacle
    }
    
    # Write to YAML file
    if filename is None:
        filename = 'obstacle_map.yaml'
    elif not filename.endswith('.yaml'):
        filename = filename + '.yaml'
    
    with open(filename, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

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

def read_grid_from_yaml(filename: str):
    """
    Read a YAML file and recreate a Grid environment.
    Args:
        filename: The filename of the YAML file to read.
    Returns:
        env: A Grid object with obstacles loaded from the YAML file.
    """

    with open(filename, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    
    dimensions = yaml_data['dimensions']
    bounds = yaml_data['bounds']
    resolution = yaml_data['resolution']
    obstacle = np.array(yaml_data['obstacle'])
    
    
    env = Grid(bounds=bounds, resolution=resolution)
    if len(dimensions) == 2:
        env.type_map[obstacle[:,0], obstacle[:,1]] = TYPES.OBSTACLE 
    elif len(dimensions) == 3:
        env.type_map[obstacle[:,0], obstacle[:,1], obstacle[:,2]] = TYPES.OBSTACLE 
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
        yaml_data = yaml.safe_load(yaml_file)
    
    dimensions = yaml_data['dimensions']
    bounds = yaml_data['bounds']
    resolution = yaml_data['resolution']
    obstacle = np.array(yaml_data['obstacle'])
    
    
    env = GraphSampler(bounds=bounds, resolution=resolution,start=[],goal=[])
    if len(dimensions) == 2:
        env.type_map[obstacle[:,0], obstacle[:,1]] = TYPES.OBSTACLE 
    elif len(dimensions) == 3:
        env.type_map[obstacle[:,0], obstacle[:,1], obstacle[:,2]] = TYPES.OBSTACLE 
    else:
        raise ValueError(f"Unsupported dimensions: {len(dimensions)}")
    return env