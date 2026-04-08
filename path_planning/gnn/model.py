import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import scatter as pyg_scatter
import torch_geometric as pyg
import torch.nn.functional as F
import inspect
from typing import Optional

def get_model(model_type:str,**kwargs):
    if model_type == 'gcn':
        # Filter kwargs to only include parameters accepted by GCN
        gcn_params = set(inspect.signature(GCN.__init__).parameters.keys())
        gcn_params.discard('self')  # Remove 'self' from the set
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in gcn_params}
        return GCN(**filtered_kwargs)
    elif model_type == 'gat':
        # Filter kwargs to only include parameters accepted by GAT
        gat_params = set(inspect.signature(GAT.__init__).parameters.keys())
        gat_params.discard('self')  # Remove 'self' from the set
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in gat_params}
        return GAT(**filtered_kwargs)
    elif model_type == 'gatv2':
        # Filter kwargs to only include parameters accepted by GATv2
        gatv2_params = set(inspect.signature(GATv2.__init__).parameters.keys())
        gatv2_params.discard('self')  # Remove 'self' from the set
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in gatv2_params}
        return GATv2(**filtered_kwargs)
    elif model_type == 'transformer':
        # Filter kwargs to only include parameters accepted by Transformer
        transformer_params = set(inspect.signature(Transformer.__init__).parameters.keys())
        transformer_params.discard('self')  # Remove 'self' from the set
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in transformer_params}
        return Transformer(**filtered_kwargs)
    # elif model_type == 'batch_gnn':
    #     # Pass through all kwargs so backbone-specific args (e.g. ``num_heads`` for GAT)
    #     # reach ``Batch_GNN``'s ``**backbone_kwargs``; strict param filtering would drop them.
    #     kw = {k: v for k, v in kwargs.items() if k != 'type'}
    #     return Batch_GNN(**kw)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

# Classes so each layer gets its own module (required for ``to_hetero`` / torch.fx: shared
# instances across ``ModuleList``s become a single ``ModuleDict`` and break iteration).
ACTIVATION_CLASSES: dict[str, type[nn.Module]] = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'selu': nn.SELU,
}

ACTIVATION_FUNCTIONS = {name: cls() for name, cls in ACTIVATION_CLASSES.items()}

POOLING_FUNCTIONS = {
    'mean': pyg.nn.aggr.MeanAggregation() ,
    'max': pyg.nn.aggr.MaxAggregation() ,
    'sum': pyg.nn.aggr.SumAggregation() ,
}

class GNN(nn.Module):
    def __init__(self, 
                    node_in_channels:int=-1, node_out_channels:int=1,
                    num_mlp_start_layers:int=1,node_mlp_start_channels:int=16,
                    num_gnn_blocks:int=3,gnn_hidden_channels:int=64,
                    num_mlp_end_layers:int=2, node_mlp_end_channels:int=16,
                    edge_in_channels:int=None,edge_use_mlp_start:bool=True,edge_mlp_start_channels:int=16,use_edge_dim:bool=False,
                    activation_function:str='relu',pooling_function: str = None,):
        super().__init__()
        
        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.num_mlp_start_layers = num_mlp_start_layers
        self.node_mlp_start_channels = node_mlp_start_channels
        self.num_gnn_blocks = num_gnn_blocks
        self.gnn_hidden_channels = gnn_hidden_channels
        self.num_mlp_end_layers = num_mlp_end_layers
        self.node_mlp_end_channels = node_mlp_end_channels
        self.edge_in_channels = edge_in_channels
        self.edge_use_mlp_start = edge_use_mlp_start
        self.edge_mlp_start_channels = edge_mlp_start_channels
        self.use_edge_dim = use_edge_dim
        self.edge_gnn_in_channels = (self.edge_mlp_start_channels if self.edge_use_mlp_start else self.edge_in_channels) if self.use_edge_dim else None
        if activation_function not in ACTIVATION_CLASSES:
            raise ValueError(f"Invalid activation_function: {activation_function}")
        self._activation_name = activation_function
        self.pooling_function = POOLING_FUNCTIONS[pooling_function] if pooling_function is not None else None
        self.start_linear = nn.ModuleList()
        self.edge_linear = nn.ModuleList()
        self.gnn = nn.ModuleList()
        self.last_linear = nn.ModuleList()

    def _new_activation(self) -> nn.Module:
        return ACTIVATION_CLASSES[self._activation_name]()

    def build_model(self):
        self.build_start_linear()
        self.build_edge_linear()
        self.build_gnn()
        self.build_last_linear()

    def build_start_linear(self):
        # start linear layers: output must be gnn_hidden_channels so first GNN layer gets correct input dim
        self.start_linear = nn.ModuleList()
        if self.num_mlp_start_layers  == 1:
            if self.node_in_channels != -1:
                self.start_linear.append(nn.Linear(self.node_in_channels, self.gnn_hidden_channels ))
            else:
                self.start_linear.append(nn.LazyLinear(self.gnn_hidden_channels ))
        else:
            if self.node_in_channels != -1:
                self.start_linear.append(nn.Linear(self.node_in_channels, self.node_mlp_start_channels))
            else:
                self.start_linear.append(nn.LazyLinear(self.node_mlp_start_channels))
            self.start_linear.append(self._new_activation())
            for _ in range(1, self.num_mlp_start_layers - 1):
                self.start_linear.append(nn.Linear(self.node_mlp_start_channels, self.node_mlp_start_channels))
                self.start_linear.append(self._new_activation())
            self.start_linear.append(nn.Linear(self.node_mlp_start_channels, self.gnn_hidden_channels ))

    def build_edge_linear(self):
        # edge linear layers
        self.edge_linear = nn.ModuleList()
        if self.edge_use_mlp_start and self.use_edge_dim:
            # When `use_edge_dim=True`, downstream conv layers expect edge features
            # of size `self.edge_gnn_in_channels` (e.g. TransformerConv's `edge_dim`).
            edge_hidden_channels = self.edge_gnn_in_channels
            if self.edge_in_channels != -1:
                self.edge_linear.append(
                    nn.Linear(self.edge_in_channels, edge_hidden_channels))
            else:
                self.edge_linear.append(nn.LazyLinear(edge_hidden_channels))
            for _ in range(1,self.num_mlp_start_layers):
                self.edge_linear.append(self._new_activation())
                self.edge_linear.append(nn.Linear(edge_hidden_channels, edge_hidden_channels))

    def build_gnn(self):
        pass # to be implemented
            
    def build_last_linear(self):
        # last linear layers
        self.last_linear = nn.ModuleList()
        if self.num_mlp_end_layers == 0:
            return
        if self.num_mlp_end_layers == 1:
            self.last_linear.append(nn.Linear(self.gnn_hidden_channels, self.node_out_channels))
        else:
            self.last_linear.append(nn.Linear(self.gnn_hidden_channels, self.node_mlp_end_channels))
            self.last_linear.append(self._new_activation())
            for _ in range(1,self.num_mlp_end_layers-1):
                self.last_linear.append(nn.Linear(self.node_mlp_end_channels, self.node_mlp_end_channels))
                self.last_linear.append(self._new_activation())
            self.last_linear.append(nn.Linear(self.node_mlp_end_channels, self.node_out_channels))

    def forward(self, x, edge_index,edge_attr=None,batch=None):
        for linear in self.start_linear:
            x = linear(x)
        if edge_attr is not None:
            if self.edge_gnn_in_channels is not None:
                for linear in self.edge_linear:
                    edge_attr = linear(edge_attr)
            else:
                edge_attr = None
        for gnn in self.gnn:
            # Check if it's an activation function (only takes x) or a GNN layer (takes x, edge_index, edge_attr)
            if isinstance(gnn, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.SELU)):
                x = gnn(x)
            else:
                x = gnn(x, edge_index, edge_attr)
        if self.pooling_function is not None and batch is not None:
            x = self.pooling_function(x, batch)
        for linear in self.last_linear:
            x = linear(x)
        # x = 5*torch.tanh(x)
        return x

    def __repr__(self):
        return f"GNN"
    
    def __str__(self):
        return self.__repr__()

class GCN(GNN):
    def __init__(self, 
                    node_in_channels:int=-1, node_out_channels:int=1,
                    num_mlp_start_layers:int=1,node_mlp_start_channels:int=16,
                    num_gnn_blocks:int=3,gnn_hidden_channels:int=64,
                    num_mlp_end_layers:int=2, node_mlp_end_channels:int=16,
                    edge_in_channels:int=None,edge_use_mlp_start:bool=True,edge_mlp_start_channels:int=16,use_edge_dim:bool=False,
                    activation_function:str='relu',pooling_function: str = None,):
        super().__init__(node_in_channels=node_in_channels, node_out_channels=node_out_channels,
                         gnn_hidden_channels=gnn_hidden_channels,
                         num_gnn_blocks=num_gnn_blocks,
                         num_mlp_start_layers=num_mlp_start_layers,node_mlp_start_channels=node_mlp_start_channels,
                         num_mlp_end_layers=num_mlp_end_layers,node_mlp_end_channels=node_mlp_end_channels,
                         edge_in_channels=edge_in_channels,edge_use_mlp_start=edge_use_mlp_start,edge_mlp_start_channels=edge_mlp_start_channels,
                         use_edge_dim=use_edge_dim,activation_function=activation_function,pooling_function=pooling_function)
         # Build the model structure
        self.build_model()
        
    def build_gnn(self):
        # GNN layers
        self.gnn = nn.ModuleList()
        if self.num_gnn_blocks == 1:
            self.gnn.append(GCNConv(self.gnn_hidden_channels, self.gnn_hidden_channels,normalize=False))
        else:
            self.gnn.append(GCNConv(self.gnn_hidden_channels, self.gnn_hidden_channels,normalize=False))
            for _ in range(1,self.num_gnn_blocks-1):
                self.gnn.append(self._new_activation())
                self.gnn.append(GCNConv(self.gnn_hidden_channels,self.gnn_hidden_channels,normalize=False))
            self.gnn.append(self._new_activation())
            self.gnn.append(GCNConv(self.gnn_hidden_channels,self.gnn_hidden_channels,normalize=False))
            
    def __repr__(self):
        return f"GCNConv"

class GAT(GNN):
    def __init__(self, 
                    node_in_channels:int=-1, node_out_channels:int=1,
                    num_mlp_start_layers:int=1,node_mlp_start_channels:int=16,
                    num_gnn_blocks:int=3,gnn_hidden_channels:int=64,
                    num_mlp_end_layers:int=2, node_mlp_end_channels:int=16,
                    edge_in_channels:int=None,edge_use_mlp_start:bool=True,edge_mlp_start_channels:int=16,
                    num_heads:int=4,concat:bool=True,residual:bool=False,dropout:float=0.0,use_edge_dim:bool=False,
                    activation_function:str='relu',pooling_function: str = None,):
        # Set GAT-specific attributes before calling super().__init__() 
        # because build_last_linear() (called in super) needs them
        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual
        self.dropout = dropout
        super().__init__(node_in_channels=node_in_channels, node_out_channels=node_out_channels,
                         num_gnn_blocks=num_gnn_blocks,gnn_hidden_channels=gnn_hidden_channels,
                         num_mlp_start_layers=num_mlp_start_layers,node_mlp_start_channels=node_mlp_start_channels,
                         num_mlp_end_layers=num_mlp_end_layers,node_mlp_end_channels=node_mlp_end_channels,
                         edge_in_channels=edge_in_channels,edge_use_mlp_start=edge_use_mlp_start,edge_mlp_start_channels=edge_mlp_start_channels,
                         use_edge_dim=use_edge_dim,activation_function=activation_function,pooling_function=pooling_function    )
        self.num_gnn_input = self.gnn_hidden_channels*self.num_heads if self.concat else self.gnn_hidden_channels
        # Build the model structure
        self.build_model()

    def build_gnn(self):
        # GNN layers
        self.gnn = nn.ModuleList()
        if self.num_gnn_blocks == 1:
            self.gnn.append(GATConv(self.gnn_hidden_channels, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,residual=self.residual,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
        else:
            self.gnn.append(GATConv(self.gnn_hidden_channels, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,residual=self.residual,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
            for _ in range(1,self.num_gnn_blocks-1):
                self.gnn.append(self._new_activation())
                self.gnn.append(GATConv(self.num_gnn_input, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,residual=self.residual,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
            self.gnn.append(self._new_activation())
            self.gnn.append(GATConv(self.num_gnn_input, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,residual=self.residual,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
            
    def build_last_linear(self):
        # last linear layers
        self.last_linear = nn.ModuleList()
        if self.num_mlp_end_layers == 0:
            return
        if self.num_mlp_end_layers == 1:
            self.last_linear.append(nn.Linear(self.num_gnn_input, self.node_out_channels))
        else:
            self.last_linear.append(nn.Linear(self.num_gnn_input, self.node_mlp_end_channels))
            for _ in range(1,self.num_mlp_end_layers-1):
                self.last_linear.append(self._new_activation())
                self.last_linear.append(nn.Linear(self.node_mlp_end_channels, self.node_mlp_end_channels))
            self.last_linear.append(self._new_activation())
            self.last_linear.append(nn.Linear(self.node_mlp_end_channels, self.node_out_channels))

    def __repr__(self):
        return f"GATConv"

class GATv2(GAT):

    def __init__(self, 
                    node_in_channels:int=-1, node_out_channels:int=1,
                    num_mlp_start_layers:int=1,node_mlp_start_channels:int=16,
                    num_gnn_blocks:int=3,gnn_hidden_channels:int=64,
                    num_mlp_end_layers:int=2, node_mlp_end_channels:int=16,
                    edge_in_channels:int=None,edge_use_mlp_start:bool=True,edge_mlp_start_channels:int=16,
                    num_heads:int=4,concat:bool=True,residual:bool=False,dropout:float=0.0,use_edge_dim:bool=False,
                    activation_function:str='relu',pooling_function: str = None,):
        super().__init__(node_in_channels=node_in_channels, node_out_channels=node_out_channels,
                         num_gnn_blocks=num_gnn_blocks,gnn_hidden_channels=gnn_hidden_channels,
                         num_mlp_start_layers=num_mlp_start_layers,node_mlp_start_channels=node_mlp_start_channels,
                         num_mlp_end_layers=num_mlp_end_layers,node_mlp_end_channels=node_mlp_end_channels,
                         edge_in_channels=edge_in_channels,edge_use_mlp_start=edge_use_mlp_start,edge_mlp_start_channels=edge_mlp_start_channels,
                         use_edge_dim=use_edge_dim,activation_function=activation_function,pooling_function=pooling_function)
        self.num_gnn_input = self.gnn_hidden_channels*self.num_heads if self.concat else self.gnn_hidden_channels
        # Build the model structure
        self.build_model()

    def build_gnn(self):
        # GNN layers
        self.gnn = nn.ModuleList()
        if self.num_gnn_blocks == 1:
            self.gnn.append(GATv2Conv(self.gnn_hidden_channels, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,residual=self.residual,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
        else:
            self.gnn.append(GATv2Conv(self.gnn_hidden_channels, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,residual=self.residual,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
            
            for _ in range(1,self.num_gnn_blocks-1):
                self.gnn.append(self._new_activation())
                self.gnn.append(GATv2Conv(self.num_gnn_input, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,residual=self.residual,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
            self.gnn.append(self._new_activation())
            self.gnn.append(GATv2Conv(self.num_gnn_input, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,residual=self.residual,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
            
    def __repr__(self):
        return f"GATv2Conv"

class Transformer(GNN):

    def __init__(self, 
                    node_in_channels:int=-1, node_out_channels:int=1,
                    num_mlp_start_layers:int=1,node_mlp_start_channels:int=16,
                    num_gnn_blocks:int=3,gnn_hidden_channels:int=64,
                    num_mlp_end_layers:int=2, node_mlp_end_channels:int=16,
                    edge_in_channels:int=None,edge_use_mlp_start:bool=True,edge_mlp_start_channels:int=16,
                    num_heads:int=4,concat:bool=True,beta=False,dropout:float=0.0,use_edge_dim:bool=False,
                    activation_function:str='relu',pooling_function: str = None,):
        self.num_heads = num_heads
        self.concat = concat
        self.beta = beta
        self.dropout = dropout
        super().__init__(node_in_channels=node_in_channels, node_out_channels=node_out_channels,
                         num_gnn_blocks=num_gnn_blocks,gnn_hidden_channels=gnn_hidden_channels,
                         num_mlp_start_layers=num_mlp_start_layers,node_mlp_start_channels=node_mlp_start_channels,
                         num_mlp_end_layers=num_mlp_end_layers,node_mlp_end_channels=node_mlp_end_channels,
                         edge_in_channels=edge_in_channels,edge_use_mlp_start=edge_use_mlp_start,edge_mlp_start_channels=edge_mlp_start_channels,
                         use_edge_dim=use_edge_dim,activation_function=activation_function,pooling_function=pooling_function)
        self.num_gnn_input = self.gnn_hidden_channels*self.num_heads if self.concat else self.gnn_hidden_channels
        # Build the model structure
        self.build_model()

    def build_gnn(self):
        # GNN layers
        self.gnn = nn.ModuleList()
        if self.num_gnn_blocks == 1:
            self.gnn.append(TransformerConv(self.gnn_hidden_channels, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,beta=self.beta,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
        else:
            self.gnn.append(TransformerConv(self.gnn_hidden_channels, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,beta=self.beta,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
            
            for _ in range(1,self.num_gnn_blocks-1):
                self.gnn.append(self._new_activation())
                self.gnn.append(TransformerConv(self.num_gnn_input, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,beta=self.beta,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))
            self.gnn.append(self._new_activation())
            self.gnn.append(TransformerConv(self.num_gnn_input, self.gnn_hidden_channels,heads=self.num_heads,concat=self.concat,beta=self.beta,dropout=self.dropout,edge_dim=self.edge_gnn_in_channels))

    def build_last_linear(self):
        # last linear: input is num_gnn_input (heads * gnn_hidden_channels), not gnn_hidden_channels
        self.last_linear = nn.ModuleList()
        if self.num_mlp_end_layers == 0:
            return
        if self.num_mlp_end_layers == 1:
            self.last_linear.append(nn.Linear(self.num_gnn_input, self.node_out_channels))
        else:
            self.last_linear.append(nn.Linear(self.num_gnn_input, self.node_mlp_end_channels))
            self.last_linear.append(self._new_activation())
            for _ in range(1, self.num_mlp_end_layers - 1):
                self.last_linear.append(nn.Linear(self.node_mlp_end_channels, self.node_mlp_end_channels))
                self.last_linear.append(self._new_activation())
            self.last_linear.append(nn.Linear(self.node_mlp_end_channels, self.node_out_channels))

    def __repr__(self):
        return f"TransformerConv"


# def _backbone_embed_dim(gnn: nn.Module) -> int:
#     """Hidden size after conv blocks (before optional last_linear)."""
#     if hasattr(gnn, 'num_gnn_input'):
#         return int(gnn.num_gnn_input)
#     return int(gnn.gnn_hidden_channels)


# class Batch_GNN(nn.Module):
#     """Backbone GNN (node embeddings) + optional global pooling + MLP head.

#     Use ``gnn_type`` in ``('gcn','gat','gatv2','transformer')``. The inner backbone is built
#     with ``num_mlp_end_layers=0`` so embeddings come straight from the message-passing stack.
#     If ``pooling_function`` is set, we apply global pooling (needs ``batch``); otherwise the
#     head is applied per-node. Optional ``pooling_size`` matches a fixed batch size for pooling.
#     """

#     def __init__(self,
#                  node_in_channels: int = -1,
#                  node_out_channels: int = 1,
#                  num_mlp_start_layers: int = 1,
#                  node_mlp_start_channels: int = 16,
#                  num_gnn_blocks: int = 3,
#                  gnn_hidden_channels: int = 64,
#                  num_mlp_end_layers: int = 2,
#                  node_mlp_end_channels: int = 16,
#                  edge_in_channels: int = None,
#                  edge_use_mlp_start: bool = True,
#                  edge_mlp_start_channels: int = 16,
#                  use_edge_dim: bool = False,
#                  activation_function: str = 'relu',
#                  pooling_function: str = None,
#                  gnn_type: str = 'gcn',
#                  **backbone_kwargs):
#         super().__init__()
#         # Input feature dim (for LazyLinear / tooling); same as passed to the inner backbone.
        
#         self.node_out_channels = node_out_channels
#         self.num_mlp_end_layers = num_mlp_end_layers
#         self.node_mlp_end_channels = node_mlp_end_channels
#         self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
#         self.pooling_name = pooling_function
#         # self.pooling_fn = POOLING_FUNCTIONS[pooling_function] if pooling_function is not None else None
#         self.pooling_fn = pyg.nn.aggr.MeanAggregation()

#         # Inner backbone: no final MLP here; head is ``pooling_last_linear`` (and optional pool).
#         inner_kw = dict[str, int | str | None](
#             node_in_channels=node_in_channels,
#             node_out_channels=1,
#             num_mlp_start_layers=num_mlp_start_layers,
#             node_mlp_start_channels=node_mlp_start_channels,
#             num_gnn_blocks=num_gnn_blocks,
#             gnn_hidden_channels=gnn_hidden_channels,
#             num_mlp_end_layers=0,
#             node_mlp_end_channels=0,
#             edge_in_channels=edge_in_channels,
#             edge_use_mlp_start=edge_use_mlp_start,
#             edge_mlp_start_channels=edge_mlp_start_channels,
#             use_edge_dim=use_edge_dim,
#             activation_function=activation_function,
#             pooling_function=None,
#         )
#         inner_kw.update(backbone_kwargs)
#         self.gnn = get_model(model_type=gnn_type, **inner_kw)
#         self.node_in_channels = self.gnn.node_in_channels
#         self.embed_dim = _backbone_embed_dim(self.gnn)

#         self.pooling_last_linear = nn.ModuleList()
#         self.build_pooling_last_linear()

#     def _pool_dim_size(self, batch: torch.Tensor) -> int:
#         """Number of graphs in the batch (explicit dim_size for scatter, no config needed)."""
#         return int(batch.max()) + 1 if batch is not None else 0

#     def build_pooling_last_linear(self):
#         self.pooling_last_linear = nn.ModuleList()
#         if self.num_mlp_end_layers == 0:
#             return
#         if self.num_mlp_end_layers == 1:
#             self.pooling_last_linear.append(nn.Linear(self.embed_dim, self.node_out_channels))
#         else:
#             self.pooling_last_linear.append(nn.Linear(self.embed_dim, self.node_mlp_end_channels))
#             self.pooling_last_linear.append(self.activation_function)
#             for _ in range(1, self.num_mlp_end_layers - 1):
#                 self.pooling_last_linear.append(nn.Linear(self.node_mlp_end_channels, self.node_mlp_end_channels))
#                 self.pooling_last_linear.append(self.activation_function)
#             self.pooling_last_linear.append(nn.Linear(self.node_mlp_end_channels, self.node_out_channels))

#     def forward(self, x, edge_index, edge_attr=None, batch=None):
#         h = self.gnn(x, edge_index, edge_attr, batch)

        
#         h = self.pooling_fn(h, batch)
#         # # dim_size = self._pool_dim_size(batch)
#         # if self.pooling_fn is not None and batch is not None:
#         #     if self.pooling_name == 'mean':
#         #         h = pyg_scatter(h, batch, dim=0,reduce='mean')
#         #     elif self.pooling_name == 'sum':
#         #         h = pyg_scatter(h, batch, dim=0, reduce='sum')
#         #     elif self.pooling_name == 'max':
#         #         h = pyg_scatter(h, batch, dim=0, reduce='max')
#         #     else:
#         #         h = pyg_scatter(h, batch, dim=0, reduce='mean')
#         for linear in self.pooling_last_linear:
#             h = linear(h)
#         return h

#     def __repr__(self):
#         return f"Batch_GNN"

#     def __str__(self):
#         return self.__repr__()