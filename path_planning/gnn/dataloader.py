
import numpy as np
from torch_geometric.data import HeteroData
import torch
from typing import Optional, Callable
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.transforms import NormalizeFeatures
from pathlib import Path
from tqdm import tqdm
from typing import List,Tuple
from multiprocessing import Pool, cpu_count
from path_planning.data_generation.target_gen import generate_roadmap,generate_target_space
import os
def _load_single_graph(files: Tuple[Path, Path]) -> HeteroData:
    """
    Load and process a single graph file (worker function for multiprocessing).
    
    Args:
        files: Tuple of (graph_file, target_file)
    
    Returns:
        HeteroData object containing the processed graph
    """
    graph_file, target_file = files

    # Normalize node features
    graph_file_split = str(graph_file).split('/')
    ind = [ii for ii in range(len(graph_file_split)) if 'map' in graph_file_split[ii] and 'resolution' in graph_file_split[ii]][0]
    resolution = float(graph_file_split[ind].split('_')[1][len('resolution'):])
    bounds = graph_file_split[ind].split('_')[0][len('map'):].split('x')
    bounds = np.array([float(b) for b in bounds])
    
    # Load optimized npz format
    data_dict = np.load(graph_file)
    node_ndata = data_dict['node_features']
    node_ndata[:,:-1] = node_ndata[:,:-1] / bounds
    node_to_node_edges = data_dict['edge_index'].T
    node_to_node_edata = data_dict['edge_attr']
    node_approx_node_edges = data_dict['approx_edge_index'].T
    node_aprox_node_edata = data_dict['approx_edge_attr']
    
    # Load target
    y: np.ndarray = np.load(target_file)
    
    # Create HeteroData
    data = HeteroData()
    data['node'].x = torch.tensor(node_ndata, dtype=torch.float)
    data['node','to','node'].edge_index = torch.tensor(node_to_node_edges, dtype=torch.long)
    data['node','to','node'].x = torch.tensor(node_to_node_edata, dtype=torch.float)
    data['node','approx','node'].edge_index = torch.tensor(node_approx_node_edges, dtype=torch.long)
    data['node','approx','node'].x = torch.tensor(node_aprox_node_edata, dtype=torch.float)
    data['node'].y = torch.tensor(y, dtype=torch.float)
    
    return data

class GraphDataset(InMemoryDataset):  
    def __init__(self,
                data_files:List[Path],
                load_file:Path = None,
                save_file:Path = '',
                root:Optional[str] = None,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None,
                num_workers: Optional[int] = 1):
        self.data_files = data_files
        self.load_file = load_file
        self.save_file = save_file if len(save_file) > 0 else os.path.join(root, 'data.pt') if root is not None else 'data.pt'
        super().__init__(root, transform, pre_transform, pre_filter)
        
        

        if load_file is not None:
            if os.path.exists(load_file):
                self.load(load_file)
                return
            print(f"Loading {load_file} does not exist, processing data")

        if len(data_files) == 0:
            raise ValueError("No data files provided")

        # Get number of workers
        if num_workers is None:
            num_workers = cpu_count()

        data_list = []

        # Parallel processing
        if num_workers > 1 and len(data_files) > 1:
            with Pool(processes=num_workers) as pool:
                for data in tqdm(
                    pool.imap(_load_single_graph, data_files),
                    total=len(data_files),
                    desc="Loading graphs"
                ):
                    data_list.append(data)
        else:
            # Sequential fallback (original behavior)
            for files in tqdm(data_files, desc="Loading graphs"):
                data = _load_single_graph(files)
                data_list.append(data)
        
        self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        return self.data_files 

    @property
    def processed_file_names(self):
        return [self.save_paths]

    def save(self):
        torch.save((self.data, self.slices), self.save_file)

def create_graph_dataset_loader(paths:List[Path],config:dict):
    data_files = []

    #Iterate over all paths
    for path in paths:
        path = Path(path)
        cases = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")])
        if not cases:
            print("No cases found to process")
            return

        #Iterate over all cases
        for case_dir in cases:
            road_map_type = config["road_map_type"] if "road_map_type" in config else "planar"
            target_space = config["target_space"] if "target_space" in config else "binary"
            road_map_type_name = generate_roadmap(road_map_type, None, [], True)
            roadmap_dir = case_dir / "samples"  / f"{road_map_type_name}" 
            _,y_type_name = generate_target_space(target_space,None,None,True)
            if not roadmap_dir.exists():
                continue
            graphs_dirs = sorted([d for d in roadmap_dir.iterdir()], key = lambda x: int(x.name.split('_')[-1]))

            #Iterate over all graphs
            for ii in range(len(graphs_dirs)):
                # Check if graph file exists
                graph_dir = graphs_dirs[ii]
                graph_file = graph_dir / f'graph.npz'
                target_file = graph_dir / f'target_{y_type_name}.npy'
                assert graph_file.exists(), f"{graph_file} does not exist"
                assert target_file.exists(), f"{target_file} does not exist"
                data_files.append((graph_file,target_file))
    return data_files



def get_graph_dataset_file_paths(paths:List[Path],config:dict):
    data_files = []

    #Iterate over all paths
    for path in paths:
        path = Path(path)
        cases = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")])
        if not cases:
            print("No cases found to process")
            return

        #Iterate over all cases
        for case_dir in cases:
            road_map_type = config["road_map_type"] if "road_map_type" in config else "planar"
            target_space = config["target_space"] if "target_space" in config else "binary"
            road_map_type_name = generate_roadmap(road_map_type, None, [], True)
            roadmap_dir = case_dir / "samples"  / f"{road_map_type_name}" 
            _,y_type_name = generate_target_space(target_space,None,None,True)
            if not roadmap_dir.exists():
                continue
            graphs_dirs = sorted([d for d in roadmap_dir.iterdir()], key = lambda x: int(x.name.split('_')[-1]))

            #Iterate over all graphs
            for ii in range(len(graphs_dirs)):
                # Check if graph file exists
                graph_dir = graphs_dirs[ii]
                graph_file = graph_dir / f'graph.npz'
                target_file = graph_dir / f'target_{y_type_name}.npy'
                assert graph_file.exists(), f"{graph_file} does not exist"
                assert target_file.exists(), f"{target_file} does not exist"
                data_files.append((graph_file,target_file))
    return data_files
