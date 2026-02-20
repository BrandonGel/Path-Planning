
import numpy as np
from torch_geometric.data import HeteroData as HeteroDataBase
from typing import Any, Optional, Callable
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import AddSelfLoops, RemoveDuplicatedEdges
from torch_geometric.utils import k_hop_subgraph,coalesce
from pathlib import Path
from tqdm import tqdm
from typing import List,Tuple
from multiprocessing import Pool, cpu_count
from path_planning.data_generation.dataset_generate import generate_roadmap,generate_target_space
import os
import torch
import torch_geometric
from torch_geometric.typing import EdgeType
from typing import Dict, Any
from path_planning.common.environment.map.graph_sampler import GraphSampler
# Allowlist PyG's base storage to bypass the security check
torch.serialization.add_safe_globals([torch_geometric.data.storage.BaseStorage])
torch.serialization.add_safe_globals([torch_geometric.data.storage.NodeStorage])
torch.serialization.add_safe_globals([torch_geometric.data.storage.EdgeStorage])

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

def add_self_loops_to_edge_type(data, edge_type, fill_value=0.0):
    """
    Adds self-loops to a specific edge type in a HeteroData object.

    Args:
        data (HeteroData): The input heterogeneous graph data.
        edge_type (tuple): The edge type (src_node_type, edge_type, dst_node_type).
        fill_value (float): The value to use for the self-loop edge attributes.
    """
    src_type, rel_type, dst_type = edge_type
    
    # 1. Isolate the specific edge type data
    edge_index = data[edge_type].edge_index
    
    # Handle existing edge attributes if they exist
    edge_attr_name = 'edge_attr' if 'edge_attr' in data[edge_type] else 'edge_weight'
    edge_attr = data[edge_type][edge_attr_name] if edge_attr_name in data[edge_type] else None
    
    # Determine the number of nodes for source and destination types
    num_src_nodes = data[src_type].num_nodes
    num_dst_nodes = data[dst_type].num_nodes
    
    # Ensure source and destination nodes are the same for self-loops
    if num_src_nodes != num_dst_nodes:
        print(f"Cannot add self-loops to asymmetric edge type {edge_type}")
        return data
        
    num_nodes = num_src_nodes

    # 2. Add self-loops
    # add_self_loops can return edge_attr, so we capture both
    new_edge_index, new_edge_attr = add_self_loops(
        edge_index,
        edge_attr,
        num_nodes=num_nodes,
        fill_value=fill_value
    )
    
    # Optionally, coalesce edges to remove duplicates if necessary
    # (add_self_loops might add duplicate (i, i) if they already exist)
    new_edge_index, new_edge_attr = coalesce(new_edge_index, new_edge_attr, num_nodes=num_nodes)

    # 3. Update the HeteroData object
    data[edge_type].edge_index = new_edge_index
    if new_edge_attr is not None:
        data[edge_type][edge_attr_name] = new_edge_attr

    return data

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
    dims = len(bounds)
    bounds = np.array([float(b) for b in bounds]).max()
    
    # Load optimized npz format
    data_dict = np.load(graph_file)
    node_ndata = data_dict['node_features']
    node_ndata[:,:dims] = node_ndata[:,:dims] / bounds
    node_to_node_edges = data_dict['edge_index'].T
    node_to_node_edata = data_dict['edge_attr'].reshape(len(data_dict['edge_attr']),-1)
    node_to_node_edata = (node_to_node_edata-node_to_node_edata.min(axis=0,keepdims=True))/node_to_node_edata.max(axis=0,keepdims=True)
    node_approx_node_edges = data_dict['approx_edge_index'].T
    node_aprox_node_edata = data_dict['approx_edge_attr'].reshape(len(data_dict['approx_edge_attr']),-1)
    node_aprox_node_edata = (node_aprox_node_edata-node_aprox_node_edata.min(axis=0,keepdims=True))/node_aprox_node_edata.max(axis=0,keepdims=True)
    binary_id = str(data_dict['binary_id'])
    # Load target
    y: np.ndarray = np.load(target_file)
    

    # Return as dictionary of numpy arrays instead of HeteroData
    return {
        'node_features': node_ndata,
        'node_to_node_edges': node_to_node_edges,
        'node_to_node_edata': node_to_node_edata,
        'node_approx_node_edges': node_approx_node_edges,
        'node_approx_node_edata': node_aprox_node_edata,
        'y': y,
        'binary_id': binary_id
    }   

class GraphDataset(InMemoryDataset):  
    def __init__(self,
                data_files:List[Path]=None,
                load_file:Path = None,
                save_file:Path = None,
                root:Optional[str] = None,
                num_hops: int = 2,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None,
                num_workers: Optional[int] = 1):
        self.data_files = data_files
        self.load_file = load_file
        self.save_file = save_file if save_file is not None and len(save_file) > 0 else os.path.join(root, 'data.pt') if root is not None else 'data.pt'
        self.data_binary_id = []
        super().__init__(root, transform, pre_transform, pre_filter)
        
        

        if load_file is not None:
            if os.path.exists(load_file):
                self.load(load_file)
                return
            print(f"Loading {load_file} does not exist, processing data")

        if len(data_files) == 0:
            raise ValueError("No data files provided")


        if data_files is None:
            raise ValueError("No data files provided")

        # Get number of workers
        if num_workers is None:
            num_workers = cpu_count()

        data_list = []

        # Parallel processing
        if num_workers > 1 and len(data_files) > 1:
            with Pool(processes=num_workers) as pool:
                for result_dict in tqdm(
                    pool.imap(_load_single_graph, data_files),
                    total=len(data_files),
                    desc="Loading graphs"
                ):
                    # Convert numpy arrays to HeteroData in main process
                    data = HeteroData()
                    data['node'].x = torch.tensor(result_dict['node_features'], dtype=torch.float)
                    data['node','to','node'].edge_index = torch.tensor(result_dict['node_to_node_edges'], dtype=torch.long)
                    data['node','to','node'].edge_attr = torch.tensor(result_dict['node_to_node_edata'], dtype=torch.float)
                    data['node','to','node'].edge_weight = None
                    data['node','approx','node'].edge_index = torch.tensor(result_dict['node_approx_node_edges'], dtype=torch.long)
                    data['node','approx','node'].edge_attr = torch.tensor(result_dict['node_approx_node_edata'], dtype=torch.float)
                    data['node','approx','node'].edge_weight = None
                    data['node'].y = torch.tensor(result_dict['y'], dtype=torch.float)
                    self.data_binary_id.append(result_dict['binary_id'])
                    transform = torch_geometric.transforms.Compose([AddSelfLoops('edge_attr',fill_value=0.0)])
                    data = transform(data)
                    # data['node','to','node'].edge_weight = 1/(1 + data['node','to','node'].edge_attr) 
                    # data['node','approx','node'].edge_weight = 1/(1 + data['node','approx','node'].edge_attr)
                    if num_hops >= 0:
                        node_mask = data['node'].y > 0 
                        node_idx = torch.where(node_mask)[0]
                        if num_hops > 0:
                            subset, _, _, _ = k_hop_subgraph(
                                node_idx, 
                                num_hops=num_hops, 
                                edge_index=data['node','to','node'].edge_index, 
                                relabel_nodes=False
                            )
                        else:
                            subset = node_idx
            

                        node_mask[:] = False
                        node_mask[subset] = True
                        subset_dict = {
                            'node': node_mask # Pass the boolean mask directly
                        }
                        for edge_type, edge_index in data.edge_index_dict.items():
                            if edge_type[0] == 'node' and  edge_type[2] != 'node':
                                edge_mask = node_mask[edge_index[0]]
                            elif edge_type[0] != 'node' and  edge_type[2] == 'node':
                                edge_mask = node_mask[edge_index[1]]
                            elif edge_type[0] == 'node' and  edge_type[2] == 'node':
                                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                            else:
                                edge_mask = torch.ones(node_mask.shape,dtype=bool)
                            subset_dict[edge_type] = edge_mask

                        data = data.subgraph(subset_dict)

                    data_list.append(data)
        else:
            # Sequential fallback (original behavior)
            for files in tqdm(data_files, desc="Loading graphs"):
                result_dict = _load_single_graph(files)
                # Convert numpy arrays to HeteroData
                data = HeteroData()
                data['node'].x = torch.tensor(result_dict['node_features'], dtype=torch.float)
                data['node','to','node'].edge_index = torch.tensor(result_dict['node_to_node_edges'], dtype=torch.long)
                data['node','to','node'].edge_attr = torch.tensor(result_dict['node_to_node_edata'], dtype=torch.float)
                data['node','to','node'].edge_weight = None
                data['node','approx','node'].edge_index = torch.tensor(result_dict['node_approx_node_edges'], dtype=torch.long)
                data['node','approx','node'].edge_attr = torch.tensor(result_dict['node_approx_node_edata'], dtype=torch.float)
                data['node','approx','node'].edge_weight = None
                data['node'].y = torch.tensor(result_dict['y'], dtype=torch.float)  
                self.data_binary_id.append(result_dict['binary_id'])
                transform = torch_geometric.transforms.Compose([AddSelfLoops('edge_attr',fill_value=0.0)])
                data = transform(data)

                if num_hops >= 0:
                    node_mask = data['node'].y > 0 
                    node_idx = torch.where(node_mask)[0]
                    if num_hops > 0:
                        subset, _, _, _ = k_hop_subgraph(
                                node_idx, 
                                num_hops=num_hops, 
                                edge_index=data['node','to','node'].edge_index, 
                                relabel_nodes=False
                            )
                    else:
                        subset = node_idx

                    node_mask[:] = False
                    node_mask[subset] = True
                    subset_dict = {
                        'node': node_mask # Pass the boolean mask directly
                    }
                    for edge_type, edge_index in data.edge_index_dict.items():
                        if edge_type[0] == 'node' and  edge_type[2] != 'node':
                            edge_mask = node_mask[edge_index[0]]
                        elif edge_type[0] != 'node' and  edge_type[2] == 'node':
                            edge_mask = node_mask[edge_index[1]]
                        elif edge_type[0] == 'node' and  edge_type[2] == 'node':
                            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                        else:
                            edge_mask = torch.ones(node_mask.shape,dtype=bool)
                        edge_mask_index = torch.where(edge_mask)[0]
                        subset_dict[edge_type] = edge_mask

                    data = data.subgraph(subset_dict)
                data_list.append(data)
        
        self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        return self.data_files 

    @property
    def processed_file_names(self):
        return [self.save_paths]

    def save(self):
        torch.save((self._data, self.slices), self.save_file)


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

    #Iterate over all paths
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
            road_map_type = config["road_map_type"] if "road_map_type" in config else "planar"
            target_space = config["target_space"] if "target_space" in config else "binary"
            road_map_type_name = generate_roadmap(road_map_type, None, [], True)
            roadmap_dir = case_dir / "samples"  / f"{road_map_type_name}" 
            _,y_type_name = generate_target_space(target_space,None,None,True)
            if not roadmap_dir.exists():
                continue

            graphs_dirs = sorted([d for d in roadmap_dir.iterdir()], key = lambda x: int(x.name.split('_')[-1]))
            if max_graphs is not None:
                graphs_dirs_end = min(max_graphs[ii],len(graphs_dirs))
                graphs_dirs = graphs_dirs[:graphs_dirs_end]

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
