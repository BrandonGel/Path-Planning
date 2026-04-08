import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from typing import List, Union, Callable, Dict, Any, Tuple


class LossFunction:
    def __init__(self, loss_config:Dict[str,Any]):
        self.loss_fcn_weights = {loss_name.lower(): loss_config[loss_name]['weight'] for loss_name in loss_config.keys()}
        self.loss_args = {loss_name.lower(): loss_config[loss_name]['args'] for loss_name in loss_config.keys()}
        self.loss_fcns = get_lost_fcn(loss_type=loss_config)
    def __call__(self, y_logits, y_true,edge_index=None,edge_attr=None,edge_weight=None,batch_dict=None,additional_args:Dict[str,Any]={}):
        total_loss = 0.0
        for loss_type,loss_fcn in self.loss_fcns.items():
            weight = self.loss_fcn_weights[loss_type]
            # Threshold losses operate purely on the provided `threshold` tensor.
            if loss_type in ('threshold_collapse', 'threshold_chasing_score'):
                if 'threshold' not in additional_args:
                    raise ValueError(
                        f"Threshold not found in additional_args for loss type: {loss_type}")

                thr = additional_args['threshold']
                args = self.loss_args.get(loss_type)
                if args is None or len(args) == 0:
                    loss = loss_fcn(thr)
                else:
                    # Allow passing extra parameters if they exist in the config.
                    loss = loss_fcn(thr, **(args | additional_args))
            elif loss_type not in self.loss_args or self.loss_args[loss_type] is None or len(self.loss_args[loss_type]) == 0:
                loss = loss_fcn(y_logits, y_true, batch_dict)
            else:
                args = self.loss_args[loss_type] | additional_args
                if loss_type == 'laplacian' and edge_index is not None:
                    loss = loss_fcn(y_logits, edge_index, edge_weight, batch_dict, **args)
                elif loss_type == 'graphsage_unsupervised' and edge_index is not None:
                    loss = loss_fcn(y_logits, y_true, edge_index, batch_dict, **args)
                else:
                    loss = loss_fcn(y_logits, y_true, batch_dict, **args)
            loss = loss * weight
            total_loss += loss
        return total_loss


def bce_loss(y_logits, y_true,batch_dict=None,weight=None,pos_weight=False,alpha=1,threshold=0,threshold_scale=1):
    if pos_weight:
        pos_weight = (y_true>0).sum()/(y_true==0).sum()
    else:
        pos_weight = None
    if isinstance(threshold, torch.Tensor):
        y_logits = threshold_scale*(y_logits-threshold)
    return F.binary_cross_entropy_with_logits(y_logits, y_true,weight=weight,pos_weight=pos_weight)

def mse_loss(y_logits, y_true,batch_dict=None,threshold_scale=1,threshold=0):
    if isinstance(threshold, torch.Tensor):
        y_logits = threshold_scale*(y_logits-threshold)
    y_pred = torch.sigmoid(y_logits)
    return F.mse_loss(y_pred, y_true)

def kld_loss(y_logits, y_true,batch_dict=None,reduction='batchmean',threshold_scale=1,threshold=0):
    if isinstance(threshold, torch.Tensor):
        y_logits = threshold_scale*(y_logits-threshold)
    y_pred_log = torch.log(torch.sigmoid(y_logits))
    return F.kl_div(y_pred_log, y_true, reduction=reduction,log_target=False)

def focal_loss(y_logits, y_true,batch_dict=None,alpha=0.25, gamma=2, reduction='mean',threshold_scale=1,threshold=0):
    if isinstance(threshold, torch.Tensor):
        y_logits = threshold_scale*(y_logits-threshold)
    return sigmoid_focal_loss(y_logits, y_true,alpha=alpha, gamma=gamma, reduction=reduction)

def soft_focal_loss(y_logits, y_true,batch_dict=None,alpha=0.25, gamma=2, reduction='mean',threshold_scale=1,threshold=0):
    if isinstance(threshold, torch.Tensor):
        y_logits = threshold_scale*(y_logits-threshold)
    ce_loss = F.binary_cross_entropy_with_logits(y_logits, y_true, reduction="none")
    y_pred = torch.sigmoid(y_logits)
    loss = alpha*torch.abs(y_pred-y_true)**gamma*ce_loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def dice_loss(y_logits, y_true,batch_dict=None,smooth=1,threshold_scale=1,threshold=0):
    if isinstance(threshold, torch.Tensor):
        y_logits = threshold_scale*(y_logits-threshold)
    y_pred = torch.sigmoid(y_logits)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2 * intersection + smooth) / (union + smooth)  
    return 1 - dice

def laplacian_loss(y_logits, edge_index,edge_weight=None,batch_dict=None,edge_types:List[Tuple[str,str,str]]=[('node','to','node')],use_edge_weight=False,threshold_scale=1,threshold=0):
    if isinstance(threshold, torch.Tensor):
        y_logits = threshold_scale*(y_logits-threshold)
    y_pred = torch.sigmoid(y_logits)
    loss =0
    if isinstance(edge_types, tuple):
        edge_types = [edge_types]
    for edge_type in edge_types:
        e_index = edge_index[edge_type]
        y_pred_diff = y_pred[e_index[0]]-y_pred[e_index[1]]
        diff_norm = y_pred_diff.abs().squeeze(-1)**2
        e_weight = 1
        if use_edge_weight and edge_weight is not None and edge_type in edge_weight:
            e_weight = edge_weight[edge_type].squeeze(-1).reshape(-1)
            loss += torch.dot(diff_norm, e_weight)  
        else:
            loss += diff_norm.sum()
    loss /= len(y_pred)
    return loss

def graphsage_unsupervised_loss(y_logits,y_true, edge_index,batch_dict=None,edge_types:List[Tuple[str,str,str]]=[('node','to','node')],p = 2,threshold_scale=1,threshold=0):
    loss = 0
    if isinstance(edge_types, tuple):
        edge_types = [edge_types]
    for edge_type in edge_types:
        edge_index = edge_index[edge_type]
        if isinstance(threshold, torch.Tensor):
            y_logits = threshold_scale*(y_logits-threshold)
        link_logits = (y_logits[edge_index[0]]*y_logits[edge_index[1]]).sum(dim=-1,keepdim=True)
        link_true = (y_true[edge_index[0]]*y_true[edge_index[1]]).sum(dim=-1,keepdim=True)
        link_true[link_true>0] = 1
        loss += bce_loss(link_logits, link_true)
    return loss

def threshold_collapse_loss(threshold):
    return (threshold**2).mean()

def threshold_chasing_score_loss(threshold=0):
    return threshold.var()

loss_fcn_types={
    'bce': bce_loss,
    'mse': mse_loss,
    'kld': kld_loss,
    'focal': focal_loss,
    'soft_focal': soft_focal_loss,
    'dice': dice_loss,
    'laplacian': laplacian_loss,
    'graphsage_unsupervised': graphsage_unsupervised_loss,
    'threshold_collapse': threshold_collapse_loss,
    'threshold_chasing_score': threshold_chasing_score_loss,
}

def get_lost_fcn(loss_type:Union[str,List[str],dict]=['bce']) -> Dict[str,Callable]:
    if isinstance(loss_type, str):
        if loss_type not in loss_fcn_types:
            raise ValueError(f"Invalid loss type: {loss_type}")
        loss_fcn = {loss_type.lower(): loss_fcn_types[loss_type.lower()]}
        return loss_fcn
    elif isinstance(loss_type, list):
        loss_fcn = {lt.lower(): loss_fcn_types[lt.lower()] for lt in loss_type}
        return loss_fcn
    elif isinstance(loss_type, dict):
        loss_fcn = {lt.lower(): loss_fcn_types[lt.lower()] for lt in loss_type.keys()}
        return loss_fcn
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
