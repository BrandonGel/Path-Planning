
import numpy as np
from torch_geometric.data import HeteroData as HeteroDataBase
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from torch_geometric.transforms import AddSelfLoops
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from path_planning.data_generation.dataset_generate import get_target_space_type_name
import os
import torch
import torch_geometric
from torch_geometric.typing import EdgeType
from path_planning.data_generation.dataset_util import generate_sample_base_path,generate_roadmap_path
# Allowlist PyG's base storage to bypass the security check
torch.serialization.add_safe_globals([torch_geometric.data.storage.BaseStorage])
torch.serialization.add_safe_globals([torch_geometric.data.storage.NodeStorage])
torch.serialization.add_safe_globals([torch_geometric.data.storage.EdgeStorage])


def _min_max_normalize_columns(x: np.ndarray) -> np.ndarray:
    """Scale each column to [0, 1] via (x - min) / (max - min). Constant columns -> 0 (no NaNs); an empty
    array (a relation without edges) is returned as it is."""
    orig_dtype = np.asarray(x).dtype
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        return x.astype(orig_dtype, copy=False)
    mn = x.min(axis=0, keepdims=True)
    mx = x.max(axis=0, keepdims=True)
    rng = mx - mn
    out = np.divide(x - mn, rng, out=np.zeros_like(x), where=rng > 0)
    return out.astype(orig_dtype, copy=False)


def normalize_positions(pos: np.ndarray, bounds) -> np.ndarray:
    """Positions into [0, 1]: shifted by the map's origin (the per-axis lower bound), then divided by the
    largest axis extent (one scale for every axis, so the aspect ratio is kept). ``bounds``: ``(dim, 2)``."""
    b = np.asarray(bounds, dtype=float).reshape(-1, 2)
    scale = float(np.max(b[:, 1] - b[:, 0]))
    return (np.asarray(pos, dtype=float) - b[:, 0]) / (scale if scale > 0 else 1.0)


def bounds_of_dataset_path(graph_file) -> np.ndarray:
    """The map bounds a sample's folder name records (``map<W>x<H>[x<D>]_resolution<r>``): ``(dim, 2)``
    from 0 to each extent (the folder name does not record the origin, so it is taken as 0)."""
    part = next(p for p in Path(graph_file).parts if p.startswith('map') and 'resolution' in p)
    extents = [float(e) for e in part.split('_')[0][len('map'):].split('x')]
    return np.array([[0.0, e] for e in extents])


class HeteroData(HeteroDataBase):
    def __init__(self,):
        super().__init__()
    
    @property
    def edge_attr_dict(self) -> Dict[EdgeType, Any]:
        r"""Returns a dictionary of all edge attributes of the graph."""
        return {edge_type: getattr(self[edge_type], 'edge_attr', None) 
                for edge_type in self.edge_index_dict.keys()}

    @property
    def edge_weight_dict(self) -> Dict[EdgeType, Any]:
        r"""Returns a dictionary of all edge weights of the graph."""
        return {edge_type: getattr(self[edge_type], 'edge_weight', None) 
                for edge_type in self.edge_index_dict.keys()}


torch.serialization.add_safe_globals([HeteroData])


def build_graph_data(node_features, bounds, edge_index, edge_attr, approx_edge_index, approx_edge_attr, y=None) -> HeteroData:
    """The model's input graph, built the same way for training (``GraphDataset``) and inference
    (``dataset_prune.normalize_data``).

    - ``node_features``: ``(N, dim + k)``, positions first (world coordinates), then the class columns;
      the positions are normalised with ``normalize_positions(., bounds)``.
    - ``edge_index`` / ``approx_edge_index``: ``(E, 2)`` node index pairs of the roadmap edges and of the
      start / goal -> node edges; ``edge_attr`` / ``approx_edge_attr``: their lengths, min-max normalised
      per relation.
    - Every relation gets a self-loop per node (attribute 0). ``y``: the per-node target, when there is one."""
    x = np.array(node_features, dtype=np.float32)
    dim = np.asarray(bounds).reshape(-1, 2).shape[0]
    x[:, :dim] = normalize_positions(x[:, :dim], bounds)
    data = HeteroData()
    data['node'].x = torch.tensor(x, dtype=torch.float)
    for relation, index, attr in (('to', edge_index, edge_attr), ('approx', approx_edge_index, approx_edge_attr)):
        attr = np.asarray(attr, dtype=np.float32).reshape(len(attr), -1)
        data['node', relation, 'node'].edge_index = torch.tensor(np.asarray(index).reshape(-1, 2).T, dtype=torch.long)
        data['node', relation, 'node'].edge_attr = torch.tensor(_min_max_normalize_columns(attr) if attr.size else attr,
                                                               dtype=torch.float)
        data['node', relation, 'node'].edge_weight = None
    if y is not None:
        data['node'].y = torch.tensor(y, dtype=torch.float)
    return AddSelfLoops('edge_attr', fill_value=0.0)(data)


def _load_single_graph(files: Tuple[Path, Path]) -> dict:
    """Read one sample (worker function for multiprocessing): the arrays ``build_graph_data`` takes.
    ``files``: ``(graph.npz, target.npy)``."""
    graph_file, target_file = files
    data_dict = np.load(graph_file)
    return {
        'node_features': data_dict['node_features'],
        'bounds': bounds_of_dataset_path(graph_file),
        'edge_index': data_dict['edge_index'],
        'edge_attr': data_dict['edge_attr'],
        'approx_edge_index': data_dict['approx_edge_index'],
        'approx_edge_attr': data_dict['approx_edge_attr'],
        'y': np.load(target_file),
    }


class GraphDataset(InMemoryDataset):  
    def __init__(self,
                data_files:List[Path]=None,
                load_file:Path = None,
                save_file:Path = None,
                root:Optional[str] = None,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None,
                num_workers: Optional[int] = 1):
        self.data_files = data_files
        self.load_file = load_file
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
        self.data, self.slices = self.collate([build_graph_data(**a) for a in arrays])

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

    # ``road_map_type``: one type or a list (config/train.yaml, util.set_train_config, run_train's -rmt).
    road_map_types = config["road_map_type"]
    road_map_types = [road_map_types] if isinstance(road_map_types, str) else list(road_map_types)
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
                
                sample_base_path = generate_sample_base_path(case_dir)
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
