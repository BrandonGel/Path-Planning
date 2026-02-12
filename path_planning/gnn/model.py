import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import inspect

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
    else:
        raise ValueError(f"Invalid model type: {model_type}")

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'elu': nn.ELU(),
    'leaky_relu': nn.LeakyReLU(),
    'prelu': nn.PReLU(),
    'selu': nn.SELU(),
}


class GNN(torch.nn.Module):
    def __init__(self, in_channels=-1, hidden_channels=64, out_channels=1,num_blocks:int=3,num_start_layers:int=1,num_mlp_layers:int=2,activation_function:str='relu'):
        super().__init__()
        
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_start_layers = num_start_layers
        self.num_mlp_layers = num_mlp_layers

        self.start_linear = nn.ModuleList()
        self.gnn = nn.ModuleList()
        self.last_linear = nn.ModuleList()
        
        # Build the model structure
        self.build_start_linear()
        self.build_last_linear()
    
    def build_start_linear(self):
        # start linear layers
        self.start_linear = nn.ModuleList()
        if self.in_channels != -1:
            self.start_linear.append(torch.nn.Linear(self.in_channels, self.hidden_channels))
        else:
            self.start_linear.append(nn.LazyLinear(self.hidden_channels))
        for _ in range(1,self.num_start_layers):
            self.start_linear.append(self.activation_function)
            self.start_linear.append(nn.Linear(self.hidden_channels, self.hidden_channels))

    def build_gnn(self):
        pass # to be implemented
            
    def build_last_linear(self):
        # last linear layers
        self.last_linear = nn.ModuleList()
        if self.num_mlp_layers == 1:
            self.last_linear.append(nn.Linear(self.hidden_channels, self.out_channels))
        else:
            self.last_linear.append(nn.Linear(self.hidden_channels, self.hidden_channels))
            self.last_linear.append(self.activation_function)            
            for _ in range(1,self.num_mlp_layers-1):
                self.last_linear.append(nn.Linear(self.hidden_channels, self.hidden_channels))
                self.last_linear.append(self.activation_function)            
            self.last_linear.append(nn.Linear(self.hidden_channels, self.out_channels))

    def forward(self, x, edge_index,edge_attr=None):
        for linear in self.start_linear:
            x = linear(x)
        for gnn in self.gnn:
            # Check if it's an activation function (only takes x) or a GNN layer (takes x, edge_index, edge_attr)
            if gnn is self.activation_function or isinstance(gnn, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.SELU)):
                x = gnn(x)
            else:
                x = gnn(x, edge_index, edge_attr)
        for linear in self.last_linear:
            x = linear(x)
        x = 5*torch.tanh(x)
        return x

    def __repr__(self):
        return f"GNN"
    
    def __str__(self):
        return self.__repr__()

class GCN(GNN):
    def __init__(self, in_channels=-1, hidden_channels=64, out_channels=1,num_blocks:int=3,num_start_layers:int=1,num_mlp_layers:int=2,activation_function:str='relu'):
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,num_blocks=num_blocks,num_start_layers=num_start_layers,num_mlp_layers=num_mlp_layers,activation_function=activation_function)
        self.build_gnn()
        
    def build_gnn(self):
        # GNN layers
        self.gnn = nn.ModuleList()
        self.gnn.append(GCNConv(self.hidden_channels, self.hidden_channels,normalize=False))
        for _ in range(1,self.num_blocks):
            self.gnn.append(GCNConv(self.hidden_channels,self.hidden_channels,normalize=False))
            self.gnn.append(self.activation_function)
            
    def __repr__(self):
        return f"GCNConv"

class GAT(GNN):
    def __init__(self, in_channels=-1, hidden_channels=64, out_channels=1,num_heads=4,edge_dim=None,concat:bool=True,residual:bool=False,dropout:float=0.0,num_blocks:int=3,num_start_layers:int=1,num_mlp_layers:int=2,activation_function:str='relu'):
        # Set GAT-specific attributes before calling super().__init__() 
        # because build_last_linear() (called in super) needs them
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.concat = concat
        self.residual = residual
        self.dropout = dropout
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,num_blocks=num_blocks,num_start_layers=num_start_layers,num_mlp_layers=num_mlp_layers,activation_function=activation_function)
        # Build GNN layers (last_linear already built in super().__init__() via build_last_linear())
        self.build_gnn()

    def build_gnn(self):
        # GNN layers
        self.gnn = nn.ModuleList()
        self.gnn.append(GATConv(self.hidden_channels, self.hidden_channels,heads=self.num_heads,concat=self.concat,edge_dim=self.edge_dim,residual=self.residual,dropout=self.dropout))
        for _ in range(1,self.num_blocks):
            self.gnn.append(GATConv(self.hidden_channels*self.num_heads if self.concat else self.hidden_channels, self.hidden_channels,heads=self.num_heads,concat=self.concat,edge_dim=self.edge_dim,residual=self.residual,dropout=self.dropout))
            self.gnn.append(self.activation_function)
            
    def build_last_linear(self):
        # last linear layers
        self.last_linear = nn.ModuleList()
        if self.num_mlp_layers == 1:
            self.last_linear.append(nn.Linear(self.hidden_channels*self.num_heads if self.concat else self.hidden_channels, self.out_channels))
        else:
            self.last_linear.append(nn.Linear(self.hidden_channels*self.num_heads if self.concat else self.hidden_channels, self.hidden_channels))
            self.last_linear.append(self.activation_function)            
            for _ in range(1,self.num_mlp_layers-1):
                self.last_linear.append(nn.Linear(self.hidden_channels, self.hidden_channels))
                self.last_linear.append(self.activation_function)            
            self.last_linear.append(nn.Linear(self.hidden_channels, self.out_channels))

    def __repr__(self):
        return f"GATConv"