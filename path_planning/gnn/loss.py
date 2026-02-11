import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from typing import List, Union, Callable, Dict, Any


class LossFunction:
    def __init__(self, loss_type:List[str], loss_fcn_weights:List[float],loss_args:Dict[str,Any]={}):
        if len(loss_type) < len(loss_fcn_weights) :
            raise ValueError("loss_type, loss_fcn_weights, and loss_args must have the same length")
        loss_args = {k: v if v is not None else None for k,v in loss_args.items()}
        self.loss_fcn_weights = {lt.lower(): weight for lt,weight in zip(loss_type,loss_fcn_weights)}
        self.loss_args = {lt.lower(): args for lt,args in zip(loss_type,loss_args)}
        self.loss_fcns = get_lost_fcn(loss_type=loss_type)
    def __call__(self, y_logits, y_true,edge_index=None):
        total_loss = 0.0
        for loss_type,loss_fcn in self.loss_fcns.items():
            weight = self.loss_fcn_weights[loss_type]
            if loss_type not in self.loss_args or self.loss_args[loss_type] is None or len(self.loss_args[loss_type]) == 0:
                loss = loss_fcn(y_logits, y_true)
            else:
                args = self.loss_args[loss_type]
                if loss_type in 'laplacian'and edge_index is not None:
                    loss = loss_fcn(y_logits,edge_index,**args)
                elif loss_type == 'graphsage_unsupervised' and edge_index is not None:
                    loss = loss_fcn(y_logits,y_true,edge_index,**args)
                else:
                    loss = loss_fcn(y_logits, y_true,**args)
            loss = loss * weight
            total_loss += loss
        return total_loss


def bce_loss(y_logits, y_true,weight=None,pos_weight=False):
    if pos_weight:
        pos_weight = (y_true>0).sum()/(y_true==0).sum()
    else:
        pos_weight = None
    return F.binary_cross_entropy_with_logits(y_logits, y_true,weight=weight,pos_weight=pos_weight)

def mse_loss(y_logits, y_true):
    y_pred = torch.sigmoid(y_logits)
    return F.mse_loss(y_pred, y_true)

def kld_loss(y_logits, y_true,reduction='batchmean'):
    y_pred_log = torch.log(torch.sigmoid(y_logits))
    return F.kl_div(y_pred_log, y_true, reduction=reduction,log_target=False)

def focal_loss(y_logits, y_true,alpha=0.25, gamma=2, reduction='mean'):
    return sigmoid_focal_loss(y_logits, y_true,alpha=alpha, gamma=gamma, reduction=reduction)

def dice_loss(y_logits, y_true,smooth=1):
    y_pred = torch.sigmoid(y_logits)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2 * intersection + smooth) / (union + smooth)  
    return 1 - dice

def laplacian_loss(y_logits, edge_index,edge_types=[('node','to','node')],p = 2):
    y_pred = torch.sigmoid(y_logits)
    loss =0
    for edge_type in edge_types:
        edge_index = edge_index[edge_type]
        y_pred_diff = y_pred[edge_index[0]]-y_pred[edge_index[1]]
        loss += torch.norm(y_pred_diff, p=p,dim=-1,keepdim=False).sum()
    loss /= len(y_pred)
    return loss

def graphsage_unsupervised_loss(y_logits,y_true, edge_index,edge_types=[('node','to','node')],p = 2):
    loss = 0
    for edge_type in edge_types:
        edge_index = edge_index[edge_type]
        link_logits = (y_logits[edge_index[0]]*y_logits[edge_index[1]]).sum(dim=-1)
        loss += bce_loss(link_logits, y_true)
    return loss/len(edge_types)

loss_fcn_types={
    'bce': bce_loss,
    'mse': mse_loss,
    'kld': kld_loss,
    'focal': focal_loss,
    'dice': dice_loss,
    'laplacian': laplacian_loss,
    'graphsage_unsupervised': graphsage_unsupervised_loss,
}

def get_lost_fcn(loss_type:Union[str,List[str]]=['bce']) -> Dict[str,Callable]:
    if isinstance(loss_type, str):
        if loss_type not in loss_fcn_types:
            raise ValueError(f"Invalid loss type: {loss_type}")
        loss_fcn = {loss_type: loss_fcn_types[loss_type]}
        return loss_fcn
    elif isinstance(loss_type, list):
        loss_fcn = {lt: loss_fcn_types[lt] for lt in loss_type}
        return loss_fcn
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
        assert False
