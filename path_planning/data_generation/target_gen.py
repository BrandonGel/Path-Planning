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

def generate_target_space(target_space_type:str, map:GraphSampler,density_map:np.ndarray,config:dict):
    if target_space_type == 'binary':
        discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
        y = np.clip([density_map[s] for s in discrete_pos],0,1)
        return y,'binary'
    elif target_space_type == 'frequency':
        discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
        y = np.array([density_map[s] for s in discrete_pos]) 
        return y,'frequency'
    elif target_space_type == 'bilinear':
        discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
        mesh_space = tuple((np.arange(0,map.shape[i],1) for i in range(map.dim)))
        interp_func = RegularGridInterpolator(mesh_space, density_map, method="linear")
        y = np.array([interp_func(discrete_pos[i]) for i in range(len(discrete_pos))])
        return y,'bilinear'
    else:
        discrete_pos = [map.world_to_map(node.current,discrete=True) for node in map.nodes]
        y = np.clip([density_map[s] for s in discrete_pos],0,1)
        return y,'binary'

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

    try:
        use_discrete_space = config["use_discrete_space"] if "use_discrete_space" in config else False
        num_samples = config["num_samples"] if "num_samples" in config else 1000
        num_neighbors = config["num_neighbors"] if "num_neighbors" in config else 4.0
        min_edge_len = config["min_edge_len"] if "min_edge_len" in config else 0.0
        max_edge_len = config["max_edge_len"] if "max_edge_len" in config else 1 + 1e-10
        num_graph_samples = config["num_graph_samples"] if "num_graph_samples" in config else 10
        road_map_type = config["road_map_type"] if "road_map_type" in config else "planar"
        target_space = config["target_space"] if "target_space" in config else "binary"
        generate_new_graph = config["generate_new_graph"] if "generate_new_graph" in config else True

        map_ = read_graph_sampler_from_yaml(
            case_dir / "input.yaml", use_discrete_space=use_discrete_space
        )
        agents = read_agents_from_yaml(case_dir / "input.yaml")
        density_map = np.load(case_dir / "ground_truth" / "density_map.npy")
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

        start = [agent["start"] for agent in agents]
        goal = [agent["goal"] for agent in agents]
        map_.set_start(start)
        map_.set_goal(goal)

        for ii in range(num_graph_samples):
            # Check if graph file exists
            road_map_type_name = generate_roadmap(road_map_type, None, [], True)
            graph_sample_path = case_dir / "samples" / f"{road_map_type_name}" / f"graph_{ii}"
            graph_sample_path.mkdir(parents=True, exist_ok=True)
            graph_file = graph_sample_path / f"graph_{ii}.pickle"
            use_exisiting_graph = graph_file.exists() and not generate_new_graph
            if use_exisiting_graph:
                with open(graph_file, "rb") as file:
                    G = pickle.load(file)
                nodes = []
                for n in G.nodes:
                    nodes.append(Node(current=G.nodes[n]["ndata"][:-1]))
            else:
                # Generates Nodes & Edges
                nodes = map_.generateRandomNodes(generate_grid_nodes=use_discrete_space)
                generate_roadmap(road_map_type, map_, nodes)

                # Get Edges & Edge Weights
                edge_weights = map_.edge_weights
                start_goal_edges_dict = map_.start_to_all_edges_dict
                start_goal_nodes = map_.get_start_nodes() + map_.get_goal_nodes()

                # Generate Node Data ('node')
                pos = np.array([node.current for node in nodes])
                # Concatenate Position & zero class vector
                ndata = np.concatenate((pos, np.zeros((pos.shape[0], 1))), axis=1)
                start_goal_idx = [map_.get_node_index(node) for node in start_goal_nodes]
                # Specify Start & Goal Nodes to have the one class value
                ndata[start_goal_idx, -1] = 1

                # Generate Edges ('node', 'to', 'node')
                edges = []
                for edge in map_.edges:
                    if edge[0] in start_goal_idx or edge[1] in start_goal_idx:
                        continue
                    edges.append(edge)
                edata = np.array(edge_weights)

                # Generate Start & Goal Edges ('node', 'approx', 'node')
                start_goal_edata = []
                for start_idx, start_edge in start_goal_edges_dict.items():
                    for goal_idx, edge_weight in start_edge:
                        start_goal_edata.append(
                            (start_idx, goal_idx, {"distance": edge_weight})
                        )

                G = nx.MultiDiGraph()
                G.add_nodes_from((i, {"ndata": ndata[i]}) for i in range(len(nodes)))
                G.add_weighted_edges_from(
                    [
                        (int(edge[0]), int(edge[1]), {"cost": edata[i]})
                        for i, edge in enumerate(edges)
                    ]
                )
                G.add_weighted_edges_from(start_goal_edata)

                graph_sample_path = (
                    case_dir / "samples" / f"{road_map_type_name}" / f"graph_{ii}"
                )
                graph_sample_path.mkdir(parents=True, exist_ok=True)
                with open(graph_sample_path / f"graph_{ii}.pickle", "wb") as f:
                    pickle.dump(G, f)

            # Generate Target Space ('y')
            y, y_type_name = generate_target_space(target_space, map_, density_map, config)
            np.save(graph_sample_path / f"target_{y_type_name}.npy", y)

            map_.clear_data()

        return True, case_dir
    except Exception as e:
        print(f"Error processing {case_dir}: {e}")
        return False, case_dir


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


