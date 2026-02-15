from typing import Any


import python_motion_planning as pmp
from python_motion_planning.common import Grid, TYPES
from path_planning.common.environment.map.graph_sampler import GraphSampler
import numpy as np
import yaml
import os
import torch
import argparse
from pathlib import Path
from typing import List

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

def read_graph_sampler_from_yaml(filename: str,use_discrete_space: bool = True):
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
    
    
    env = GraphSampler(bounds=bounds, resolution=resolution,start=[],goal=[],use_discrete_space=use_discrete_space)
    if len(dimensions) == 2 or len(dimensions) == 3:
        env.set_obstacle_map(obstacles)
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

def set_global_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_train_config(train_config: dict = None,args: argparse.Namespace = None):
    """
    Set the train config.
    Args:
        args: The arguments to set the train config.
    Returns:
        train_config: The train config.
    """
    assert args is not None, "Arguments must be provided"

    if train_config is None:
        with open(args.config_file, 'r') as f:
            train_config = yaml.load(f,Loader=yaml.FullLoader)

    # Setting the seed
    if args.seed is not None:
        train_config['seed'] = args.seed
    else:
        train_config['seed'] = train_config['seed'] if train_config['seed'] is not None else 42

    # Setting the dataset folder paths
    if args.folder_paths is not None:
        folder_path = [Path(p) for p in args.folder_paths]
        train_config['dataset']['folder_path'] = folder_path
        assert len(train_config['dataset']['folder_path']) > 0, "Dataset folder path must be provided"
        assert type(train_config['dataset']['folder_path']) == List, "Dataset folder path must be a list"

    # Setting the dataset load file
    if args.load_file is not None:
        load_file = Path(args.load_file)
        train_config['dataset']['load_file'] = load_file
        assert os.path.exists(train_config['dataset']['load_file']), "Dataset load file must exist"
    
    # Setting the dataset save file
    if args.save_file is not None:
        save_file = Path(args.save_file)
        train_config['dataset']['save_file'] = save_file
        assert os.path.exists(train_config['dataset']['save_file']), "Dataset save file must exist"
    else:
        print("No save file provided, will not save the dataset")
    assert train_config['dataset']['folder_path'] is not None or train_config['dataset']['load_file'] is not None, f"Folder paths must be provided either in the {args.config_file} or in the arguments"

    # Setting the dataset config
    if args.use_discrete_space is not None and args.road_map_type is not None and args.target_space is not None:
        config = {
            'use_discrete_space':args.use_discrete_space,
            'road_map_type':args.road_map_type,
            'target_space':args.target_space,
        }
        train_config['dataset']['config'] = config
    assert train_config['dataset']['config'] is not None, "Dataset config must be provided"
    assert type(train_config['dataset']['config']) == dict, "Dataset config must be a dictionary"
    assert len(train_config['dataset']['config']) > 0, "Dataset config must be a non-empty dictionary"
    assert 'use_discrete_space' in train_config['dataset']['config'] and 'road_map_type' in train_config['dataset']['config'] and 'target_space' in train_config['dataset']['config'], f"Dataset config must contain use_discrete_space, road_map_type, and target_space"
    
    # Check the batch size and test size
    batch_size = train_config['train']['batch_size']    
    test_size = train_config['train']['test_size']
    assert batch_size > 0 and test_size > 0, "Batch size and test size must be greater than 0 "

    # Set the compile and compile dynamic
    compile = args.compile  or train_config['model'].get('compile',False)
    compile_dynamic = args.compile_dynamic or train_config['model'].get('compile_dynamic',True)
    train_config['model']['compile'] = compile
    train_config['model']['compile_dynamic'] = compile_dynamic

    # Check & set the resume epoch and load folder
    if 'resume_epoch' not in train_config['train']:
        train_config['train']['resume_epoch'] = 0
        print("Resume epoch not provided in the config file, setting to 0")
    if 'load_folder' not in train_config['train']:
        train_config['train']['load_folder'] = None
    resume_epoch = train_config['train']['resume_epoch']
    model_load_folder = train_config['train']
    if resume_epoch > 0:
        assert model_load_folder is not None, "Load folder must be provided if resume epoch is greater than 0"
        assert os.path.exists(model_load_folder), "Load folder must exist"
        model_load_file = f"epoch_{resume_epoch}.pth"        
        model_load_path = os.path.join(model_load_folder, model_load_file)
        assert os.path.exists(model_load_path), "Model file must exist"
    return train_config