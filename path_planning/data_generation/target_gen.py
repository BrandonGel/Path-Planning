from path_planning.utils.util import read_graph_sampler_from_yaml, read_agents_from_yaml
import numpy as np
import networkx as nx
import numpy as np
from pathlib import Path
from tqdm import tqdm
from path_planning.common.environment.map.graph_sampler import GraphSampler, Node
from typing import List, Tuple
from scipy.interpolate import RegularGridInterpolator
import pickle
from multiprocessing import Pool, cpu_count

def generate_target_space(target_space_type:str, map:GraphSampler,density_map:np.ndarray,ignore_generate: bool = False,config:dict = None):
    y = []
    name = 'binary'
    if target_space_type == 'binary':
        if not ignore_generate:
            discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
            y = np.clip([density_map[s] for s in discrete_pos],0,1)
        name ='binary'
    elif target_space_type == 'bilinear':
        if not ignore_generate:
            discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
            mesh_space = tuple((np.arange(0,map.shape[i],1) for i in range(map.dim)))
            interp_func = RegularGridInterpolator(mesh_space, density_map/np.sum(density_map), method="linear")
            y = np.array([interp_func(discrete_pos[i]) for i in range(len(discrete_pos))])
        name ='bilinear'
    elif target_space_type == 'distribution':
        if not ignore_generate:
            discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
            y = np.array([density_map[s] for s in discrete_pos])
            y = y / np.sum(y)
        name ='distribution'
    elif 'fuzzy' in target_space_type:
        if not ignore_generate:
            num_hops = int(config["num_hops"]) if config is not None and "num_hops" in config else 1
            if num_hops < 1:
                num_hops = 1
            discrete_pos = [map.world_to_map(node.current, discrete=True) for node in map.nodes]            
            y_density = np.array([density_map[s] for s in discrete_pos]) 
            if target_space_type == 'fuzzy_binary':
                y_density = np.clip(y_density,0,1)
            elif target_space_type == 'fuzzy_distribution':
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
        if target_space_type == 'fuzzy_binary':
            name = 'fuzzy_binary'
        elif target_space_type == 'fuzzy_distribution':
            name = 'fuzzy_distribution'
    else:
        if not ignore_generate:
            discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
            y = np.clip([density_map[s] for s in discrete_pos],0,1)
        name ='binary'
    return y,name

def generate_roadmap(road_map_type: str, map: GraphSampler, nodes: List[Node], ignore_generate: bool = False):
    if road_map_type == 'prm':
        if not ignore_generate:
            map.generate_roadmap(nodes)
        return 'prm'
    else:
        if not ignore_generate:
            map.generate_planar_map(nodes)
        return 'planar'

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
    road_map_type = config["road_map_type"] if "road_map_type" in config else "planar"
    target_space = config["target_space"] if "target_space" in config else "binary"
    generate_new_graph = config["generate_new_graph"] if "generate_new_graph" in config else True
    is_start_goal_discrete = config["is_start_goal_discrete"] if "is_start_goal_discrete" in config else True

    map_ = read_graph_sampler_from_yaml(
        case_dir / "input.yaml", use_discrete_space=use_discrete_space
    )
    agents = read_agents_from_yaml(case_dir / "input.yaml")
    density_map = np.load(case_dir / "ground_truth" / "density_map.npy")

    # Starts and Goals are assumed to be in grid space
    if use_discrete_space:
        map_.set_parameters(
            sample_num=0,
            num_neighbors=num_neighbors,
            min_edge_len=min_edge_len,
            max_edge_len=max_edge_len,
        )
        
    else:
        map_.set_parameters(
            sample_num=num_samples,
            num_neighbors=num_neighbors,
            min_edge_len=min_edge_len,
            max_edge_len=max_edge_len,
        )

    if is_start_goal_discrete:
        start = [agent["start"] for agent in agents]
        goal = [agent["goal"] for agent in agents]
    else:
        start = [map_.map_to_world(agent["start"],discrete=True) for agent in agents]
        goal = [map_.map_to_world(agent["goal"],discrete=True) for agent in agents]
    map_.set_start(start)
    map_.set_goal(goal)

    for ii in range(num_graph_samples):
        # Check if graph file exists
        road_map_type_name = generate_roadmap(road_map_type, None, [], True)
        graph_sample_path = case_dir / "samples" / f"{road_map_type_name}" / f"graph_{ii}"
        graph_sample_path.mkdir(parents=True, exist_ok=True)
        # Check if npz graph already exists
        graph_file = graph_sample_path / f"graph.npz"
        use_exisiting_graph = graph_file.exists() and not generate_new_graph
        if use_exisiting_graph:
            # Load from npz format
            data_dict = np.load(graph_file)
            position = data_dict['node_features'][:,:-1]
            edge_index = data_dict['edge_index']
            edge_attr = data_dict['edge_attr']
            use_roadmap = False
            if 'fuzzy' in target_space:
                use_roadmap = True
            map_.read_from_numpy(position, edge_index, edge_attr, use_roadmap)
        else:
            # Generates Nodes & Edges
            nodes = map_.generateRandomNodes(generate_grid_nodes=use_discrete_space)
            generate_roadmap(road_map_type, map_, nodes)

            # Get Edges & Edge Weights
            edge_weights = map_.edge_weights
            start_goal_edges_dict = map_.start_to_all_edges_dict | map_.goal_to_all_edges_dict
            start_goal_nodes = map_.get_start_nodes() + map_.get_goal_nodes()

            # Generate Node Data ('node')
            pos = np.array([node.current for node in nodes])
            # Concatenate Position & zero class vector
            ndata = np.concatenate((pos, np.zeros((pos.shape[0], 1))), axis=1)
            start_goal_idx = [map_.get_node_index(node) for node in start_goal_nodes]
            # Specify Start & Goal Nodes to have the one class value
            ndata[start_goal_idx, -1] = 1

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

            # Ensure directory exists
            graph_sample_path.mkdir(parents=True, exist_ok=True)
            
            # Convert to numpy arrays and save as compressed npz (much smaller than pickle)
            node_to_node_edges_arr = edges.astype(np.int32)
            node_to_node_weights_arr = edata.astype(np.float32)
            
            start_goal_edges_arr = np.array(start_goal_edges_list, dtype=np.int32)
            start_goal_weights_arr = np.array(start_goal_weights_list, dtype=np.float32)
            
            # Save as compressed npz (10x smaller than pickle)
            np.savez_compressed(
                graph_sample_path / f"graph.npz",
                node_features=ndata.astype(np.float32),
                edge_index=node_to_node_edges_arr,
                edge_attr=node_to_node_weights_arr,
                approx_edge_index=start_goal_edges_arr,
                approx_edge_attr=start_goal_weights_arr
            )

        # Generate Target Space ('y')
        y, y_type_name = generate_target_space(target_space, map_, density_map,config=config)
        np.save(graph_sample_path / f"target_{y_type_name}.npy", y)

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


