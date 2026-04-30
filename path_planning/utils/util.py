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
import math
import random
from multiprocessing import cpu_count

def convert_to_pixel(x,y):
    pixel_x = int(np.round(x))
    pixel_y = int(np.round(-1 - y))
    return pixel_x, pixel_y

def _to_native_yaml(obj):
    """Convert numpy scalars/arrays in nested structures to native Python for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _to_native_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native_yaml(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

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

def read_graph_sampler_from_yaml(filename: str,use_discrete_space: bool = True,graph_file: str = None, args: dict = {}):
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
    if graph_file is None:
        if len(dimensions) == 2 or len(dimensions) == 3:
            env.set_obstacles(obstacles)
        else:
            raise ValueError(f"Unsupported dimensions: {len(dimensions)}")
    else:
        if os.path.exists(graph_file):
            env.load_graph_sampler(graph_file, args=args)
        else:
            print(f"Graph file {graph_file} not found")
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_map_config(map_config: dict = None,args: argparse.Namespace = None):
    """
    Set the map config.
    Args:
        args: The arguments to set the map config.
    Returns:
        map_config: The map config.
    """
    assert args is not None, "Arguments must be provided"
    if map_config is None:
        with open(args.config_file, 'r') as f:
            map_config = yaml.load(f,Loader=yaml.FullLoader)
    args_dict = vars(args)

    # Setting the seed
    if args_dict.get('seed') is not None:
        map_config['seed'] = args.seed
    else:
        map_config['seed'] = map_config['seed'] if map_config['seed'] is not None else 42
    
    # Setting the path
    if args_dict.get('path') is not None and os.path.exists(args_dict.get('path')):
        map_config['path'] = args.path
    else:
        map_config['path'] = map_config['path'] if map_config['path'] is not None else os.path.join(os.getcwd(), 'benchmark', 'train')
    map_config['path'] = Path(map_config['path'])

    # Setting the bounds
    if args_dict.get('bounds') is not None:
        # Convert bounds from flat list to nested list format
        if isinstance(args.bounds, list):
            if len(args.bounds) == 2:
                bounds = [[0, args.bounds[1]], [0, args.bounds[0]]]
            elif len(args.bounds) == 3:
                bounds = [[0, args.bounds[1]], [0, args.bounds[0]], [0, args.bounds[2]]]
            elif len(args.bounds) == 4:
                bounds = [[args.bounds[0], args.bounds[1]], [args.bounds[2], args.bounds[3]]]
            elif len(args.bounds) == 6:
                bounds = [[args.bounds[0], args.bounds[1]], [args.bounds[2], args.bounds[3]], [args.bounds[4], args.bounds[5]]]
            else:
                raise ValueError(f"Invalid bounds: {args.bounds}")
        else:
            bounds = args.bounds
        map_config['bounds'] = bounds
    else:
        map_config['bounds'] = map_config['bounds'] if map_config['bounds'] is not None else [[0, 32.0], [0, 32.0]]
    
    # Setting the nb_agents
    if args_dict.get('nb_agents') is not None:
        map_config['nb_agents'] = args.nb_agents
    else:
        map_config['nb_agents'] = map_config['nb_agents'] if map_config['nb_agents'] is not None else 4

    # Setting the nb_obstacles
    if args_dict.get('nb_obstacles') is not None:
        map_config['nb_obstacles'] = args.nb_obstacles
    else:
        map_config['nb_obstacles'] = map_config['nb_obstacles'] if map_config['nb_obstacles'] is not None else 0.1
    
    # Setting the resolution
    if args_dict.get('resolution') is not None:
        map_config['resolution'] = args.resolution
    else:
        map_config['resolution'] = map_config['resolution'] if map_config['resolution'] is not None else 1.0
    
    # Setting the agent_radius
    if args_dict.get('agent_radius') is not None:
        map_config['agent_radius'] = round(args.agent_radius, 3)
    else:
        map_config['agent_radius'] = round(map_config['agent_radius'] if map_config['agent_radius'] is not None else 0.0, 3)

    # Setting the agent_velocity
    if args_dict.get('agent_velocity') is not None:
        map_config['agent_velocity'] = args.agent_velocity
    else:
        map_config['agent_velocity'] = map_config['agent_velocity'] if map_config['agent_velocity'] is not None else 0.0

    # Setting the num cases
    if args_dict.get('num_cases') is not None:
        map_config['num_cases'] = args.num_cases
    else:
        map_config['num_cases'] = map_config['num_cases'] if map_config['num_cases'] is not None else 1
    
    # Setting the solve_till_success
    if args_dict.get('solve_till_success') is not None:
        map_config['solve_till_success'] = args.solve_till_success
    else:
        map_config['solve_till_success'] = map_config['solve_till_success'] if map_config['solve_till_success'] is not None else False

    # Setting the nb_permutations
    if args_dict.get('nb_permutations') is not None:
        map_config['nb_permutations'] = args.nb_permutations
    else:
        map_config['nb_permutations'] = map_config['nb_permutations'] if map_config['nb_permutations'] is not None else 1
    if map_config['nb_permutations'] <= 0:
        assert False, "Number of permutations must be greater than 0"
    max_permutations = math.factorial(2 * map_config['nb_agents'])
    if map_config['nb_permutations'] > max_permutations:
        print(f"Warning: Requested {map_config['nb_permutations']} permutations, but only {max_permutations} possible")
        map_config['nb_permutations'] = max_permutations

    # Setting the nb_permutations_tries
    if args_dict.get('nb_permutations_tries') is not None:
        map_config['nb_permutations_tries'] = args.nb_permutations_tries
    else:
        map_config['nb_permutations_tries'] = map_config['nb_permutations_tries'] if map_config['nb_permutations_tries'] is not None else map_config['nb_permutations']
    if map_config['solve_till_success']:
        if map_config['nb_permutations_tries'] < map_config['nb_permutations']:
            print(f"Warning: nb_permutations_tries ({map_config['nb_permutations_tries']}) < nb_permutations ({map_config['nb_permutations']})")
            print(f"Setting nb_permutations_tries to {map_config['nb_permutations'] * 2}")
        map_config['nb_permutations_tries'] = max(map_config['nb_permutations_tries'], map_config['nb_permutations'] * 2) 
    else:
        map_config['nb_permutations_tries'] = map_config['nb_permutations']

    # Setting the time_limit
    if args_dict.get('time_limit') is not None:
        map_config['time_limit'] = args.time_limit
    else:
        map_config['time_limit'] = map_config['time_limit'] if map_config['time_limit'] is not None else 60

    # Setting the max_iterations
    if args_dict.get('max_iterations') is not None:
        map_config['max_iterations'] = args.max_iterations
    else:
        map_config['max_iterations'] = map_config['max_iterations'] if map_config['max_iterations'] is not None else 10000

    # Setting the mapf_solver_name
    if args_dict.get('mapf_solver_name') is not None:
        map_config['mapf_solver_name'] = args.mapf_solver_name
    else:
        map_config['mapf_solver_name'] = map_config['mapf_solver_name'] if map_config['mapf_solver_name'] is not None else 'cbs'

    # Setting the road_map_type
    if args_dict.get('road_map_type') is not None:
        map_config['road_map_type'] = args.road_map_type
    else:
        map_config['road_map_type'] = map_config['road_map_type'] if map_config['road_map_type'] is not None else 'grid'

    # Setting the use_discrete_space
    if args_dict.get('use_discrete_space') is not None:
        map_config['use_discrete_space'] = args.use_discrete_space
    else:
        map_config['use_discrete_space'] = map_config['use_discrete_space'] if map_config['use_discrete_space'] is not None else True

    # Setting the generate_new_graph
    if args_dict.get('generate_new_graph') is not None:
        map_config['generate_new_graph'] = args.generate_new_graph
    else:
        map_config['generate_new_graph'] = map_config['generate_new_graph'] if map_config['generate_new_graph'] is not None else True

    # Setting the sample_num
    if args_dict.get('num_samples') is not None:
        map_config['sample_num'] = args.num_samples
    else:
        map_config['sample_num'] = map_config['num_samples'] if map_config['num_samples'] is not None else 0

    # Setting the num_neighbors
    if args_dict.get('num_neighbors') is not None:
        map_config['num_neighbors'] = args.num_neighbors
    else:
        map_config['num_neighbors'] = map_config['num_neighbors'] if map_config['num_neighbors'] is not None else 4.0

    # Setting the min_edge_len
    if args_dict.get('min_edge_len') is not None:
        map_config['min_edge_len'] = args.min_edge_len
    else:
        map_config['min_edge_len'] = map_config['min_edge_len'] if map_config['min_edge_len'] is not None else 1e-10

    # Setting the max_edge_len
    if args_dict.get('max_edge_len') is not None:
        map_config['max_edge_len'] = args.max_edge_len
    else:
        map_config['max_edge_len'] = map_config['max_edge_len'] if map_config['max_edge_len'] is not None else 1.1

    # Setting the heuristic_type
    if args_dict.get('heuristic_type') is not None:
        map_config['heuristic_type'] = args.heuristic_type
    else:
        map_config['heuristic_type'] = map_config['heuristic_type'] if map_config['heuristic_type'] is not None else 'manhattan'

    # Setting the is_start_goal_discrete
    if args_dict.get('is_start_goal_discrete') is not None:
        map_config['is_start_goal_discrete'] = args.is_start_goal_discrete
    else:
        map_config['is_start_goal_discrete'] = map_config['is_start_goal_discrete'] if map_config['is_start_goal_discrete'] is not None else True

    # Setting the weighted_sampling
    if args_dict.get('weighted_sampling') is not None:
        map_config['weighted_sampling'] = args.weighted_sampling
    else:
        map_config['weighted_sampling'] = map_config['weighted_sampling'] if map_config['weighted_sampling'] is not None else True

    # Setting the samp_from_prob_map_ratio
    if args_dict.get('samp_from_prob_map_ratio') is not None:
        map_config['samp_from_prob_map_ratio'] = args.samp_from_prob_map_ratio
    else:
        map_config['samp_from_prob_map_ratio'] = map_config['samp_from_prob_map_ratio'] if map_config['samp_from_prob_map_ratio'] is not None else 0.5

    # Setting the num_graph_samples
    if args_dict.get('num_graph_samples') is not None:
        map_config['num_graph_samples'] = args.num_graph_samples
    else:
        map_config['num_graph_samples'] = map_config['num_graph_samples'] if map_config['num_graph_samples'] is not None else 1

    # Setting the target_space
    if args_dict.get('target_space') is not None:
        map_config['target_space'] = args.target_space
    else:
        map_config['target_space'] = map_config['target_space'] if map_config['target_space'] is not None else 'convolution_binary'

    # Setting the num_workers
    if args_dict.get('num_workers') is not None:
        map_config['num_workers'] = args.num_workers
    else:
        map_config['num_workers'] = map_config['num_workers'] if map_config['num_workers'] is not None else cpu_count()
   
    # Setting the verbose
    if args_dict.get('verbose') is not None:
        map_config['verbose'] = args.verbose
    else:
        map_config['verbose'] = map_config['verbose'] if map_config['verbose'] is not None else False

    return map_config


def get_model_epoch_file(base_path: Path, epoch: int = 0, model_name_suffix: str = ""):
    if model_name_suffix != "":
        model_file = base_path / f"epoch_{epoch}_{model_name_suffix}.pth"
    else:   
        model_file = base_path / f"epoch_{epoch}.pth"
    if not model_file.exists():
        return False
    return model_file

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
        assert type(train_config['dataset']['folder_path']) == list, "Dataset folder path must be a list"

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
    compile = args.compile  or train_config['device'].get('compile',False)
    compile_dynamic = args.compile_dynamic or train_config['device'].get('compile_dynamic',True)
    train_config['device']['compile'] = compile
    train_config['device']['compile_dynamic'] = compile_dynamic

    # Check & set the resume epoch and load folder
    if 'resume_epoch' not in train_config['train']:
        train_config['train']['resume_epoch'] = 0
        print("Resume epoch not provided in the config file, setting to 0")
    if 'load_folder' not in train_config['train']:
        train_config['train']['load_folder'] = None
    resume_epoch = train_config['train']['resume_epoch']
    model_load_folder = Path(train_config['train']['load_folder'])
    if resume_epoch > 0:
        assert model_load_folder is not None, "Load folder must be provided if resume epoch is greater than 0"
        assert os.path.exists(model_load_folder), "Load folder must exist"
        model_file = get_model_epoch_file(model_load_folder, resume_epoch)
        assert model_file, "Model file must exist"

        if train_config['threshold']['use']:
            model_threshold_file = get_model_epoch_file(model_load_folder, resume_epoch)
            assert model_threshold_file, "Model threshold file must exist"
    return train_config