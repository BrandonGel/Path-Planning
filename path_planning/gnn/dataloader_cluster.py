
import numpy as np
from typing import Callable, List, Optional, Tuple
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from torch_geometric.transforms import AddSelfLoops
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from path_planning.data_generation.cluster_dataset_generate import get_target_space_type_name
import os
import torch
from path_planning.data_generation.dataset_util import generate_cluster_sample_base_path,generate_roadmap_path
# One graph container and one normalisation for the score and the cluster loaders (re-exported here:
# cluster/gnn_cluster_map.py imports them from this module).
from path_planning.gnn.dataloader import (
    HeteroData, _min_max_normalize_columns, bounds_of_dataset_path, normalize_positions,
)


def _load_single_graph(files: Tuple[Path, Path]) -> dict:
    """Read and normalise one cluster sample (worker function for multiprocessing).
    ``files``: ``(graph.npz, target.npy)``. Returns numpy arrays; the graph is built in the main process."""
    graph_file, target_file = files
    bounds = bounds_of_dataset_path(graph_file)
    dims = len(bounds)

    data_dict = np.load(graph_file)
    node_ndata = data_dict['node_features']
    node_ndata[:,:dims] = normalize_positions(node_ndata[:,:dims], bounds)
    node_to_node_edges = data_dict['edge_index'].T
    node_to_node_edata = data_dict['edge_attr'].reshape(len(data_dict['edge_attr']),-1)
    node_to_node_edata = _min_max_normalize_columns(node_to_node_edata)
    node_approx_node_edges = data_dict['approx_edge_index'].T
    node_aprox_node_edata = data_dict['approx_edge_attr'].reshape(len(data_dict['approx_edge_attr']),-1)
    node_aprox_node_edata = _min_max_normalize_columns(node_aprox_node_edata)
    # Boundary self-loops ('node','boundary','node'); attr = Euclidean distance to the obstacle
    # boundary, normalized like the other relations' distances.
    node_boundary_node_edges = data_dict['boundary_edge_index'].reshape(-1, 2).T
    node_boundary_node_edata = _min_max_normalize_columns(data_dict['boundary_edge_attr'].reshape(-1, 1))
    # Shortest-path supervision pairs (L_SP, cluster.md §11); attr = graph
    # shortest-path distance, min-max normalized like the other relations.
    node_sp_node_edges = data_dict['sp_pair_index'].reshape(-1, 2).T
    node_sp_node_edata = _min_max_normalize_columns(data_dict['sp_pair_dist'].reshape(-1, 1))
    return {
        'node_features': node_ndata,
        'node_to_node_edges': node_to_node_edges,
        'node_to_node_edata': node_to_node_edata,
        'node_approx_node_edges': node_approx_node_edges,
        'node_approx_node_edata': node_aprox_node_edata,
        'node_boundary_node_edges': node_boundary_node_edges,
        'node_boundary_node_edata': node_boundary_node_edata,
        'node_sp_node_edges': node_sp_node_edges,
        'node_sp_node_edata': node_sp_node_edata,
        'y': np.load(target_file),
    }

TASK_RELATION = ('node', 'approx', 'node')      # start/goal -> all-node Dijkstra edges
BOUNDARY_RELATION = ('node', 'boundary', 'node')  # boundary-node self-loops


def apply_edge_ablation(data: HeteroData, use_boundary_edges: bool = True,
                        use_task_edges: bool = True,
                        use_node_type_features: bool = True,
                        use_boundary_node_features: bool = True) -> HeteroData:
    """Ablate a cluster HeteroData in place (idempotent, returns data):
    - use_task_edges=False / use_boundary_edges=False drop the approx /
      boundary relation (message passing only; the node one-hot and the
      cluster target y still carry the information);
    - use_node_type_features=False flattens the [start/goal, free, boundary]
      one-hot to [0, 1, 0] for every node (positions untouched), so the
      encoder input carries no obstacle / task information at all. The
      cluster loss then cannot exclude start/goal nodes; inference anchors
      stay protected via the map's start/goal indices;
    - use_boundary_node_features=False re-labels boundary nodes as free
      ([.., 0, 1] -> [.., 1, 0]) while keeping the start/goal column, i.e.
      the input carries task information but no obstacle information."""
    if not use_task_edges and TASK_RELATION in data.edge_types:
        del data[TASK_RELATION]
    if not use_boundary_edges and BOUNDARY_RELATION in data.edge_types:
        del data[BOUNDARY_RELATION]
    if not use_node_type_features:
        x = data['node'].x
        dim = x.shape[1] - 3
        x[:, dim:] = 0.0
        x[:, dim + 1] = 1.0
    elif not use_boundary_node_features:
        x = data['node'].x
        dim = x.shape[1] - 3
        b = x[:, dim + 2] > 0.5
        x[b, dim + 1] = 1.0
        x[b, dim + 2] = 0.0
    return data


def get_dummy_sample_data(dim: int = 2, use_boundary_edges: bool = True,
                          use_task_edges: bool = True):
    """Minimal HeteroData whose metadata() matches real cluster samples:
    to (+ approx / boundary when the corresponding flag is on). The model is
    to_hetero'd on this, so the flags must match the data the model sees."""
    data = HeteroData()
    num_nodes = 100
    # dim + 3: [pos | start/goal | free | boundary]
    data['node'].x = torch.tensor(np.zeros((num_nodes,dim+3)), dtype=torch.float)
    data['node','to','node'].edge_index = torch.tensor(np.zeros((2,num_nodes)), dtype=torch.long)
    data['node','to','node'].edge_attr = torch.tensor(np.zeros((num_nodes,1)), dtype=torch.float)
    data['node','to','node'].edge_weight = None
    if use_task_edges:
        data['node','approx','node'].edge_index = torch.tensor(np.zeros((2,num_nodes)), dtype=torch.long)
        data['node','approx','node'].edge_attr = torch.tensor(np.zeros((num_nodes,1)), dtype=torch.float)
        data['node','approx','node'].edge_weight = None
    data['node'].y = torch.tensor(np.zeros((num_nodes,1)), dtype=torch.float)
    data = AddSelfLoops('edge_attr', fill_value=0.0)(data)
    # After AddSelfLoops so the boundary relation keeps only its own loops;
    # present in the dummy so metadata() matches real cluster data.
    if use_boundary_edges:
        data['node','boundary','node'].edge_index = torch.tensor(np.tile(np.arange(num_nodes), (2, 1)), dtype=torch.long)
        data['node','boundary','node'].edge_attr = torch.tensor(np.zeros((num_nodes,1)), dtype=torch.float)
        data['node','boundary','node'].edge_weight = None
    return data

def _build_cluster_data(arrays: dict) -> HeteroData:
    """The HeteroData of one sample read by ``_load_single_graph`` (before the ablations)."""
    data = HeteroData()
    data['node'].x = torch.tensor(arrays['node_features'], dtype=torch.float)
    for relation in ('to', 'approx'):
        data['node', relation, 'node'].edge_index = torch.tensor(arrays[f'node_{relation}_node_edges'], dtype=torch.long)
        data['node', relation, 'node'].edge_attr = torch.tensor(arrays[f'node_{relation}_node_edata'], dtype=torch.float)
        data['node', relation, 'node'].edge_weight = None
    data['node'].y = torch.tensor(arrays['y'], dtype=torch.float)
    data = AddSelfLoops('edge_attr', fill_value=0.0)(data)
    # Boundary self-loops attach AFTER AddSelfLoops: the transform adds (i,i) for every node to every
    # existing relation, which would flood the boundary signal. 'sp' is supervision-only: L_SP reads it,
    # but the model never message-passes over it - to_hetero is built from
    # get_dummy_sample_data().metadata(), which omits 'sp', and extra relations in a batch are ignored
    # by the generated forward. PyG batching still offsets its indices.
    for relation in ('boundary', 'sp'):
        data['node', relation, 'node'].edge_index = torch.tensor(arrays[f'node_{relation}_node_edges'], dtype=torch.long)
        data['node', relation, 'node'].edge_attr = torch.tensor(arrays[f'node_{relation}_node_edata'], dtype=torch.float)
        data['node', relation, 'node'].edge_weight = None
    return data


class GraphDataset(InMemoryDataset):  
    def __init__(self,
                data_files:List[Path]=None,
                load_file:Path = None,
                save_file:Path = None,
                root:Optional[str] = None,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None,
                num_workers: Optional[int] = 1,
                use_boundary_edges: bool = True,
                use_task_edges: bool = True,
                use_node_type_features: bool = True,
                use_boundary_node_features: bool = True):
        self.data_files = data_files
        self.load_file = load_file
        # Ablation flags (see apply_edge_ablation); applied per graph below.
        self.use_boundary_edges = bool(use_boundary_edges)
        self.use_task_edges = bool(use_task_edges)
        self.use_node_type_features = bool(use_node_type_features)
        self.use_boundary_node_features = bool(use_boundary_node_features)
        if load_file is not None and not (self.use_boundary_edges and self.use_task_edges
                                          and self.use_node_type_features
                                          and self.use_boundary_node_features):
            raise ValueError(
                "load_file caches un-ablated graphs; set load_file to null for an "
                "ablation run (use_boundary_edges/use_task_edges/use_node_type_features)")
        self.save_file = save_file if save_file is not None and len(str(save_file)) > 0 else os.path.join(root, 'data.pt') if root is not None else 'data.pt'
        super().__init__(root, transform, pre_transform, pre_filter)

        if load_file is not None:
            if os.path.exists(load_file):
                self.load(load_file)
                return
            print(f"Loading {load_file} does not exist, processing data")

        if not data_files:
            raise ValueError("No data files provided")

        # Get number of workers
        if num_workers is None:
            num_workers = cpu_count()

        if num_workers > 1 and len(data_files) > 1:
            with Pool(processes=num_workers) as pool:       # read in parallel, build the graphs here
                arrays = list(tqdm(pool.imap(_load_single_graph, data_files), total=len(data_files), desc="Loading graphs"))
        else:
            arrays = [_load_single_graph(files) for files in tqdm(data_files, desc="Loading graphs")]
        self.data, self.slices = self.collate([
            apply_edge_ablation(_build_cluster_data(a), self.use_boundary_edges, self.use_task_edges,
                                self.use_node_type_features, self.use_boundary_node_features)
            for a in arrays])

    @property
    def raw_file_names(self):
        return self.data_files 

    @property
    def processed_file_names(self):
        return [self.save_file]

    def save(self):
        """Write the collated dataset to ``save_file`` in PyG's own format (what ``load`` reads)."""
        fs.torch_save((self._data.to_dict(), self.slices, self._data.__class__), str(self.save_file))


def get_graph_dataset_file_paths(paths:List[Path],config:dict, max_cases:list[int]=None,max_graphs:list[int]=None):
    data_files = []

    if max_cases is not None:
        if isinstance(max_cases,int):
            max_cases = [max_cases]*len(paths)
        if len(max_cases) != len(paths):
            raise ValueError("max_cases must be a list of the same length as paths")
    if max_graphs is not None:
        if isinstance(max_graphs,int):
            max_graphs = [max_graphs]*len(paths)
        if len(max_graphs) != len(paths):
            raise ValueError("max_graphs must be a list of the same length as paths")

    road_map_types = list(config["road_map_types"])        # (config/train_cluster.yaml writes the list)
    target_space = config["target_space"].lower()

    #Iterate over all paths
    for road_map_type in road_map_types:
        for ii,path in enumerate(paths):
            path = Path(path)
            cases = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")],key=lambda x: int(x.name.split('_')[-1]))
            if not cases:
                print("No cases found to process")
                return

            if max_cases is not None:   
                cases_end = min(max_cases[ii],len(cases))
                cases = cases[:cases_end]
            #Iterate over all cases
            for case_dir in cases:
                
                sample_base_path = generate_cluster_sample_base_path(case_dir)
                roadmap_dir = generate_roadmap_path(sample_base_path,road_map_type)
                y_type_name = get_target_space_type_name(target_space)
                if not roadmap_dir.exists():
                    continue

                graphs_dirs = sorted([d for d in roadmap_dir.iterdir()], key = lambda x: int(x.name.split('_')[-1]))
                if max_graphs is not None:
                    graphs_dirs_end = min(max_graphs[ii],len(graphs_dirs))
                    graphs_dirs = graphs_dirs[:graphs_dirs_end]

                #Iterate over all graphs
                for graph_dir in graphs_dirs:
                    graph_file = graph_dir / f'graph.npz'
                    target_file = graph_dir / f'target_{y_type_name}.npy'
                    assert graph_file.exists(), f"{graph_file} does not exist"
                    assert target_file.exists(), f"{target_file} does not exist"
                    data_files.append((graph_file,target_file))
    return data_files
