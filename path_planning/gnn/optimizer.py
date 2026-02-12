import torch
import inspect
from typing import List

OPTIMIZER_TYPES = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
    'adagrad': torch.optim.Adagrad,
    'adamax': torch.optim.Adamax,
}

def get_optimizer(optimizer_type:str,model_weights:List[torch.Tensor],**kwargs):
    if optimizer_type not in OPTIMIZER_TYPES:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")
    # Filter kwargs to only include parameters accepted by GAT
    optimizer_params = set(inspect.signature(OPTIMIZER_TYPES[optimizer_type].__init__).parameters.keys())
    optimizer_params.discard('self')  # Remove 'self' from the set
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in optimizer_params}
    return OPTIMIZER_TYPES[optimizer_type](model_weights,**filtered_kwargs)