"""
Dataset generation for MAPF training using the ground truth dataset.
"""

from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
import os
import tempfile
import numpy as np
from pathlib import Path
from tqdm import tqdm
from path_planning.common.environment.map.graph_sampler import GraphSampler, Node
from typing import List, Tuple
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool, cpu_count
from scipy.ndimage import generic_filter,generate_binary_structure
from path_planning.utils.util import set_global_seed
from path_planning.data_generation.dataset_util import *


def _clone_graph_sampler(map_: GraphSampler) -> GraphSampler:
    """
    Clone a GraphSampler without deepcopy. CGAL/SWIG objects in constraint_sweep
    are not copyable; save_graph_sampler only stores plain data and load
    reconstructs a fresh CGAL_Sweep.
    """
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        map_.save_graph_sampler(path)
        clone = GraphSampler(
            bounds=map_.bounds,
            resolution=map_.resolution,
            start=[],
            goal=[],
            sample_num=0,
            num_neighbors=map_.num_neighbors,
            min_edge_len=map_.min_edge_length,
            max_edge_len=map_.max_edge_length,
            use_discrete_space=map_.use_discrete_space,
            use_constraint_sweep=map_.use_constraint_sweep,
            record_sweep=map_.record_sweep,
            use_exact_collision_check=map_.use_exact_collision_check,
        )
        clone.load_graph_sampler(path)
        return clone
    finally:
        os.unlink(path)


TARGET_SPACE_TYPE_TO_NAME = {
    'binary',
    'bilinear',
    'distribution',
    'fuzzy_binary',
    'fuzzy_distribution',
    'convolution_binary',
    'convolution_distribution',
}

def weighted_max_op(window, weights,ignore_value = -1):
    # 'window' is passed as a 1D array by scipy
    return np.max(window * weights) if window[len(window)//2] != ignore_value else ignore_value

def get_prob_map(target_space_type:str, map:GraphSampler,density_map:np.ndarray,config:dict = None):
    if target_space_type == 'binary':
        prob_map = np.clip(density_map,0,1)/np.clip(density_map,0,1).sum()
    elif target_space_type == 'bilinear':
        prob_map = density_map/np.sum(density_map)
    elif target_space_type == 'distribution':
        prob_map = density_map/np.sum(density_map)
    elif 'fuzzy' in target_space_type:
        if target_space_type == 'fuzzy_binary':
            prob_map = np.clip(density_map,0,1)/np.clip(density_map,0,1).sum()
        elif target_space_type == 'fuzzy_distribution':
            prob_map = density_map/np.sum(density_map)
    elif 'convolution' in  target_space_type:
        num_hops = int(config["num_hops"]) if config is not None and "num_hops" in config else 1
        if num_hops < 1:
            num_hops = 1
        resolution = map.resolution
        kernel = generate_binary_structure(map.dim,1).astype(int)/(1+resolution)
        center_pt = (1,)*map.dim
        kernel[tuple(center_pt)] = 1
        if target_space_type == 'convolution_binary':
            new_density_map = np.clip(density_map,0,1)
        elif target_space_type == 'convolution_distribution':
            new_density_map = density_map / density_map.sum()
        new_density_map[map.get_obstacle_map()] = -1
        for _ in range(num_hops):     
            new_density_map = generic_filter(new_density_map, weighted_max_op, size=kernel.shape, 
                    extra_keywords={'weights': kernel.flatten()})    
        new_density_map[map.get_obstacle_map()] = 0
        prob_map = new_density_map/new_density_map.sum()
    else:
        prob_map = np.clip(density_map,0,1)/np.clip(density_map,0,1).sum()
    return prob_map

def generate_target_space(target_space_type:str, map:GraphSampler,density_map:np.ndarray,ignore_generate: bool = False,config:dict = None):
    y = []
    if target_space_type.lower() in TARGET_SPACE_TYPE_TO_NAME:
        discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
        y = np.clip([density_map[s] for s in discrete_pos],0,1)
    elif target_space_type.lower() in TARGET_SPACE_TYPE_TO_NAME:
        discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
        mesh_space = tuple((np.arange(0,map.shape[i],1) for i in range(map.dim)))
        interp_func = RegularGridInterpolator(mesh_space, density_map/np.sum(density_map), method="linear")
        y = np.array([interp_func(discrete_pos[i]) for i in range(len(discrete_pos))])
    elif target_space_type.lower() in TARGET_SPACE_TYPE_TO_NAME:
        discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
        y = np.array([density_map[s] for s in discrete_pos])
        y = y / density_map.sum()
    elif 'fuzzy' in TARGET_SPACE_TYPE_TO_NAME[target_space_type.lower()]:
        num_hops = int(config["num_hops"]) if config is not None and "num_hops" in config else 1
        if num_hops < 1:
            num_hops = 1
        discrete_pos = [map.world_to_map(node.current, discrete=True) for node in map.nodes]            
        y_density = np.array([density_map[s] for s in discrete_pos]) 
        if target_space_type.lower() in TARGET_SPACE_TYPE_TO_NAME:
            y_density = np.clip(y_density,0,1)
        elif target_space_type.lower() in TARGET_SPACE_TYPE_TO_NAME:
            y_density = y_density / np.sum(density_map)
        y = y_density.copy()
        
        # PRE-COMPUTE: Build graph structure once (OPTIMIZATION)
        num_nodes = len(map.nodes)
        y_node_index = {map.node_index_list[node]: ii for ii, node in enumerate(map.nodes)}
        
        # Pre-build neighbor arrays for efficient access
        max_neighbors = max(len(map.road_map[map.node_index_list[node]]) 
                        for node in map.nodes) if num_nodes > 0 else 0
        if max_neighbors > 0:
            neighbor_indices = np.full((num_nodes, max_neighbors), -1, dtype=np.int32)
            neighbor_costs = np.zeros((num_nodes, max_neighbors), dtype=np.float32)
            neighbor_counts = np.zeros(num_nodes, dtype=np.int32)
            
            for i, node in enumerate(map.nodes):
                node_idx = map.node_index_list[node]
                neighbors = list(map.road_map[node_idx])
                neighbor_counts[i] = len(neighbors)
                
                for j, neighbor_idx in enumerate(neighbors):
                    neighbor_indices[i, j] = y_node_index[neighbor_idx]
                    neighbor_costs[i, j] = map.cost_matrix[(node_idx, neighbor_idx)]
            
            # PROPAGATION LOOP (now much faster)
            for _ in range(num_hops):
                y_inverse_cost = np.zeros(num_nodes)
                
                for i in range(num_nodes):
                    if neighbor_counts[i] > 0:
                        # Get valid neighbors
                        valid_neighbors = neighbor_indices[i, :neighbor_counts[i]]
                        neighbor_values = y[valid_neighbors]/(neighbor_costs[i,:neighbor_counts[i]] + 1)
                        
                        # Find best neighbor
                        y_inverse_cost[i] = np.max(neighbor_values)
                
                # Vectorized update
                y = np.clip(np.maximum(y_density, y_inverse_cost), 0, 1)
    elif 'convolution' in  TARGET_SPACE_TYPE_TO_NAME[target_space_type.lower()]:
        num_hops = int(config["num_hops"]) if config is not None and "num_hops" in config else 1
        if num_hops < 1:
            num_hops = 1
        if target_space_type.lower() in TARGET_SPACE_TYPE_TO_NAME:
            new_density_map = np.clip(density_map,0,1)
        elif target_space_type.lower() in TARGET_SPACE_TYPE_TO_NAME:
            new_density_map = density_map / density_map.sum()
        new_density_map[map.get_obstacle_map()] = -1

        resolution = map.resolution
        kernel = generate_binary_structure(map.dim,1).astype(int)/(1+resolution)
        center_pt = (1,)*map.dim
        kernel[tuple(center_pt)] = 1
        for _ in range(num_hops):     
            new_density_map = generic_filter(new_density_map, weighted_max_op, size=kernel.shape, 
                    extra_keywords={'weights': kernel.flatten(),'ignore_value':-1})
        discrete_pos = [map.world_to_map(node.current, discrete=True) for node in map.nodes]            
        y = np.array([new_density_map[s] for s in discrete_pos]) 
    else:
        discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
        y = np.clip([density_map[s] for s in discrete_pos],0,1)
    return y

def transform_graph_map_to_gnn(map_: GraphSampler):
    # Get Edges & Edge Weights
    edge_weights = map_.edge_weights
    start_goal_edges_dict = {**map_.get_start_nodes_with_all_edges(), **map_.get_goal_nodes_with_all_edges()}
    start_goal_nodes = map_.get_start_nodes() + map_.get_goal_nodes()

    # Generate Node Data ('node')
    pos = np.array([node.current for node in map_.nodes])
    # Concatenate Position & zero class vector
    ndata = np.concatenate((pos, np.zeros((pos.shape[0], 2))), axis=1)
    start_goal_mask = np.full(len(ndata),fill_value=0).astype(bool)
    start_goal_idx = [map_.get_node_index(node) for node in start_goal_nodes]
    start_goal_mask[start_goal_idx] = True
    # Specify Start & Goal Nodes to have the one class value
    ndata[start_goal_mask, map_.dim] = 1
    ndata[~start_goal_mask, map_.dim+1] = 1

    # Generate Edges ('node', 'to', 'node')
    edges = np.array(map_.edges)
    edata = np.array(edge_weights)

    # Generate Start & Goal Edges ('node', 'approx', 'node')
    start_goal_edges_list = []
    start_goal_weights_list = []
    for u_idx, u_to_v_edge in start_goal_edges_dict.items():
        for v_idx, edge_weight in u_to_v_edge:
            start_goal_edges_list.append((u_idx, v_idx))
            start_goal_weights_list.append(edge_weight)

    
    
    # Convert to numpy arrays and save as compressed npz (much smaller than pickle)
    node_to_node_edges_arr = edges.astype(np.int32)
    node_to_node_weights_arr = edata.astype(np.float32)
    
    start_goal_edges_arr = np.array(start_goal_edges_list, dtype=np.int32)
    start_goal_weights_arr = np.array(start_goal_weights_list, dtype=np.float32)

    return ndata,node_to_node_edges_arr, node_to_node_weights_arr, start_goal_edges_arr, start_goal_weights_arr

def process_single_case_graphs(args: Tuple[Path, dict]) -> Tuple[bool, Path]:
    """
    Generate graph samples for a single case.

    Args:
        args: Tuple of (case_dir, config)

    Returns:
        Tuple of (success: bool, case_dir: Path)
    """
    case_dir, config = args


    use_discrete_space = config["use_discrete_space"] if "use_discrete_space" in config else False
    num_samples = config["num_samples"] if "num_samples" in config else 1000
    num_neighbors = config["num_neighbors"] if "num_neighbors" in config else 4.0
    min_edge_len = config["min_edge_len"] if "min_edge_len" in config else 0.0
    max_edge_len = config["max_edge_len"] if "max_edge_len" in config else 1 + 1e-10
    num_graph_samples = config["num_graph_samples"] if "num_graph_samples" in config else 10
    roadmap_type = config["roadmap_type"] if "roadmap_type" in config else "prm"
    roadmap_type_gt = config["roadmap_type_gt"] if "roadmap_type_gt" in config else "grid"
    target_space = config["target_space"] if "target_space" in config else "binary"
    generate_new_graph = config["generate_new_graph"] if "generate_new_graph" in config else True
    agent_velocity = config["agent_velocity"] if "agent_velocity" in config else 0.0
    is_start_goal_discrete = config["is_start_goal_discrete"] if "is_start_goal_discrete" in config else True
    graph_file_name = config["graph_file_name"] if "graph_file_name" in config else None
    agent_radius = config["agent_radius"] if "agent_radius" in config else 0.0
    resolution = config["resolution"] if "resolution" in config else 1.0

    input_file = get_input_file_path(case_dir)
    map_ = read_graph_sampler_from_yaml(
        input_file, use_discrete_space=use_discrete_space
    )
    map_.set_inflation_radius(radius=agent_radius+np.sqrt(2)/2*resolution)
    agents = read_agents_from_yaml(input_file)
    gt_dir = generate_roadmap_path(generate_ground_truth_path(case_dir), roadmap_type_gt)
    solution_name_suffix = get_solution_name_suffix(graph_file=graph_file_name)
    density_map_file = get_density_map_file(gt_dir, solution_name_suffix,agent_velocity)
    density_map = np.load(density_map_file)

    # Starts and Goals are assumed to be in grid space
    map_.set_parameters(
        sample_num=0 if use_discrete_space else num_samples,
        num_neighbors=num_neighbors,
        min_edge_len=min_edge_len,
        max_edge_len=max_edge_len,
    )
    if is_start_goal_discrete and not use_discrete_space:
        start = [map_.map_to_world(agent["start"]) for agent in agents]
        goal = [map_.map_to_world(agent["goal"]) for agent in agents]
    elif not is_start_goal_discrete and use_discrete_space:
        start = [map_.world_to_map(agent["start"],discrete=True) for agent in agents]
        goal = [map_.world_to_map(agent["goal"],discrete=True) for agent in agents]
    else:
        start = [agent["start"] for agent in agents]
        goal = [agent["goal"] for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)
    

    sample_base_dir = generate_roadmap_path(generate_sample_base_path(case_dir), roadmap_type)
    for ii in range(num_graph_samples):
        # Check if graph gnn file exists
        graph_sample_path = generate_sample_path(sample_base_dir, ii, 0)
        graph_sample_path.mkdir(parents=True, exist_ok=True)
        graph_file = get_graph_file_path(graph_sample_path)
        graph_gnn_file = get_graph_gnn_file_path(graph_sample_path)
        use_exisiting_graph = graph_gnn_file.exists() and graph_file.exists() and not generate_new_graph
        if use_exisiting_graph:
            map_.load_graph_sampler(graph_file)
            data_dict = np.load(graph_gnn_file)
            ndata = data_dict['node_features']
            node_to_node_edges_arr = data_dict['edge_index']
            node_to_node_weights_arr = data_dict['edge_attr']
            start_goal_edges_arr = data_dict['approx_edge_index']
            start_goal_weights_arr = data_dict['approx_edge_attr']
            binary_id = data_dict['binary_id']
        else:
            # Generates Nodes & Edges
            weighted_sampling = config["weighted_sampling"] if "weighted_sampling" in config else False
            if weighted_sampling:
                prob_map = get_prob_map(target_space, map_, density_map,config=config)
            else:
                prob_map = None
            samp_from_prob_map_ratio = config["samp_from_prob_map_ratio"] if "samp_from_prob_map_ratio" in config else 0.5
            map_.generateRandomNodes(generate_grid_nodes=use_discrete_space,prob_map=prob_map,samp_from_prob_map_ratio=samp_from_prob_map_ratio)
            map_.generate_map(roadmap_type,map_.nodes)

            ndata,node_to_node_edges_arr, node_to_node_weights_arr, start_goal_edges_arr, start_goal_weights_arr = transform_graph_map_to_gnn(map_)
            # Save as compressed npz (10x smaller than pickle)
            np.savez_compressed(
                graph_gnn_file,
                node_features=ndata.astype(np.float32),
                edge_index=node_to_node_edges_arr,
                edge_attr=node_to_node_weights_arr,
                approx_edge_index=start_goal_edges_arr,
                approx_edge_attr=start_goal_weights_arr,
                binary_id=np.binary_repr(0, width=map_.dim)
            )

            map_.save_graph_sampler(graph_file)

        # Generate Target Space ('y')
        if target_space.lower() in TARGET_SPACE_TYPE_TO_NAME:
            y_type_name = target_space.lower()
        else:
            raise ValueError(f"Target space type {target_space} not supported")
        target_file = get_target_file_path(graph_sample_path, y_type_name)
        if not (use_exisiting_graph and target_file.exists()):
            y = generate_target_space(target_space, map_, density_map,config=config)
            np.save(target_file, y)

        # 4 rotation augmentations: k = 1, 2, 3  (90°, 180°, 270° CCW)
        num_augmentations = 4
        for augmentation_id in range(1, num_augmentations):
            graph_sample_path = generate_sample_path(sample_base_dir, ii, augmentation_id)
            gnn_graph_file = get_graph_gnn_file_path(graph_sample_path)
            graph_file = get_graph_file_path(graph_sample_path)
            target_file = get_target_file_path(graph_sample_path, y_type_name)

            use_existing_aug = gnn_graph_file.exists() and graph_file.exists() and not generate_new_graph
            if use_existing_aug and target_file.exists():
                continue

            graph_sample_path.mkdir(parents=True, exist_ok=True)

            # Rotate a fresh copy of the map by augmentation_id * 90° CCW
            map_aug = _clone_graph_sampler(map_)
            map_aug.rotate(augmentation_id, axes=(0, 1))

            # Rotate the density map to match the new orientation
            rotated_density = np.rot90(density_map, k=augmentation_id, axes=(0, 1))

            # Generate node features and graph arrays for the rotated map
            (ndata_aug,
             node_to_node_edges_arr_aug,
             node_to_node_weights_arr_aug,
             start_goal_edges_arr_aug,
             start_goal_weights_arr_aug) = transform_graph_map_to_gnn(map_aug)

            np.savez_compressed(
                gnn_graph_file,
                node_features=ndata_aug.astype(np.float32),
                edge_index=node_to_node_edges_arr_aug,
                edge_attr=node_to_node_weights_arr_aug,
                approx_edge_index=start_goal_edges_arr_aug,
                approx_edge_attr=start_goal_weights_arr_aug,
                binary_id=np.binary_repr(augmentation_id, width=2),
            )
            map_aug.save_graph_sampler(graph_file)

            y = generate_target_space(target_space, map_aug, rotated_density, config=config)
            np.save(target_file, y)
        map_.clear_data()

    return True, case_dir

def generate_graph_samples(path: Path, config: dict, num_workers: int | None = None) -> None:
    """
    Generate graph samples for all cases in a dataset directory.

    Args:
        path: Path to dataset directory containing case folders
        config: Configuration dict for graph generation
        num_workers: Number of parallel workers (default: auto-detect CPU cores)
    """
    set_global_seed(config.get("seed", 42))
    path = Path(path)
    cases = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")])
    if not cases:
        print("No cases found to process")
        return

    print(f"Generating graph samples for {len(cases)} cases")

    # Get number of workers
    if num_workers is None:
        num_workers = cpu_count()

    # Prepare case tasks
    case_tasks = [(case_dir, config) for case_dir in cases]

    # Process cases in parallel
    successful = 0
    failed = 0

    if num_workers > 1 and len(case_tasks) > 1:
        with Pool(processes=num_workers) as pool:
            for success, case_dir in tqdm(
                pool.imap_unordered(process_single_case_graphs, case_tasks),
                total=len(case_tasks),
                desc="Generating graph samples",
            ):
                if success:
                    successful += 1
                else:
                    failed += 1
    else:
        # Sequential fallback
        for task in tqdm(case_tasks, desc="Generating graph samples"):
            success, case_dir = process_single_case_graphs(task)
            if success:
                successful += 1
            else:
                failed += 1

    print(f"Graph samples -- [{successful}/{len(cases)}] Complete!")
    if failed > 0:
        print(f"Warning: {failed} cases failed to process")


