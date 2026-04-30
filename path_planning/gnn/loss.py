import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from typing import List, Union, Callable, Dict, Any, Tuple, Optional

_DEFAULT_EDGE_TYPES: Tuple[Tuple[str, str, str], ...] = (("node", "to", "node"),)
_THRESHOLD_LOSS_NAMES = frozenset({"threshold_collapse", "threshold_chasing_score"})


def get_logits(s: torch.Tensor, alpha=1.0, tau=0) -> torch.Tensor:
    return alpha * (s - tau)


def get_probability(s: torch.Tensor, alpha=1.0, tau=0) -> torch.Tensor:
    return torch.sigmoid(get_logits(s, alpha, tau))


class LossFunction:
    def __init__(self, loss_config: Dict[str, Any]):
        weights: Dict[str, Any] = {}
        args_map: Dict[str, Any] = {}
        for loss_name, cfg in loss_config.items():
            k = loss_name.lower()
            weights[k] = cfg["weight"]
            args_map[k] = cfg["args"]
        self.loss_fcn_weights = weights
        self.loss_args = args_map
        self.loss_fcns = get_lost_fcn(loss_type=loss_config)

    def __call__(
        self,
        y_logits,
        y_true,
        edge_index=None,
        edge_attr=None,
        edge_weight=None,
        batch_dict=None,
        additional_args: Optional[Dict[str, Any]] = None,
    ):
        extra = additional_args or {}
        total_loss: torch.Tensor | float = 0.0
        for loss_type, loss_fcn in self.loss_fcns.items():
            weight = self.loss_fcn_weights[loss_type]
            if loss_type in _THRESHOLD_LOSS_NAMES:
                if "threshold" not in extra:
                    raise ValueError(
                        f"Threshold not found in additional_args for loss type: {loss_type}"
                    )

                thr = extra["threshold"]
                args = self.loss_args.get(loss_type)
                if args is None or len(args) == 0:
                    loss = loss_fcn(thr)
                else:
                    loss = loss_fcn(thr, **(args | extra))
            elif (
                loss_type not in self.loss_args
                or self.loss_args[loss_type] is None
                or len(self.loss_args[loss_type]) == 0
            ):
                loss = loss_fcn(y_logits, y_true, batch_dict)
            else:
                args = self.loss_args[loss_type] | extra
                if loss_type == "laplacian" and edge_index is not None:
                    loss = loss_fcn(y_logits, edge_index, edge_weight, batch_dict, **args)
                elif loss_type == "graphsage_unsupervised" and edge_index is not None:
                    loss = loss_fcn(y_logits, y_true, edge_index, batch_dict, **args)
                else:
                    loss = loss_fcn(y_logits, y_true, batch_dict, **args)
            total_loss = total_loss + loss * weight
        return total_loss


def _normalize_edge_types(
    edge_types: Optional[Union[List[Tuple[str, str, str]], Tuple[str, str, str]]],
) -> List[Tuple[str, str, str]]:
    if edge_types is None:
        return list(_DEFAULT_EDGE_TYPES)
    if isinstance(edge_types, tuple):
        if len(edge_types) == 3 and all(isinstance(x, str) for x in edge_types):
            return [edge_types]  # type: ignore[list-item]
        return list(edge_types)  # type: ignore[arg-type]
    return list(edge_types)


def bce_loss(
    y_logits,
    y_true,
    batch_dict=None,
    weight=None,
    pos_weight=False,
):
    if pos_weight:
        n_pos = (y_true > 0).sum().to(dtype=y_true.dtype)
        n_neg = (y_true == 0).sum().to(dtype=y_true.dtype)
        pos_weight_t = n_pos / n_neg.clamp_min(1)
    else:
        pos_weight_t = None
    return F.binary_cross_entropy_with_logits(
        y_logits, y_true, weight=weight, pos_weight=pos_weight_t
    )


def mse_loss(y_logits, y_true, batch_dict=None):
    y_pred = torch.sigmoid(y_logits)
    return F.mse_loss(y_pred, y_true)


def kld_loss(
    y_logits,
    y_true,
    batch_dict=None,
    reduction="batchmean",
):
    y_pred = torch.sigmoid(y_logits)
    y_pred_log = torch.log(y_pred)
    return F.kl_div(y_pred_log, y_true, reduction=reduction, log_target=False)


def focal_loss(
    y_logits,
    y_true,
    batch_dict=None,
    alpha=0.25,
    gamma=2,
    reduction="mean",
):
    return sigmoid_focal_loss(
        y_logits, y_true, alpha=alpha, gamma=gamma, reduction=reduction
    )

def soft_focal_loss(
    y_logits,
    y_true,
    batch_dict=None,
    alpha=0.25,
    gamma=2,
    reduction="mean",
):
    ce_loss = F.binary_cross_entropy_with_logits(y_logits, y_true, reduction="none")
    y_pred = torch.sigmoid(y_logits)
    loss = alpha * torch.abs(y_pred - y_true).pow(gamma) * ce_loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Invalid reduction: {reduction}")

def ldam_loss(
    y_logits,
    y_true,
    batch_dict=None,
    reduction="mean",
    max_margin=0.5,
    probability_decision_threshold=0.5,
    threshold_alpha=1,
    threshold=0,
    n_pos_global: Optional[float] = None,
    n_neg_global: Optional[float] = None,
):
    """Label-Distribution-Aware Margin loss (binary logits).

    Per Cao et al., margins scale as n_j^(-1/4) and are normalized so
    max(m_pos, m_neg) equals ``max_margin`` 
    Positive (negative) examples use adjusted logits z - m_pos (z + m_neg).
    Samples strictly between the two probability thresholds get no margin
    (plain BCE on ``z``).

    If ``n_pos_global`` / ``n_neg_global`` are set (e.g. full-dataset class
    counts), margins use those counts instead of the current batch.
    """
    z = get_logits(y_logits, threshold_alpha, threshold)
    y_true = y_true.to(device=z.device, dtype=z.dtype)
    if y_true.shape != z.shape:
        y_true = y_true.reshape_as(z)
    y_pos = y_true > probability_decision_threshold
    y_neg = y_true < (1 - probability_decision_threshold)

    if n_pos_global is not None and n_neg_global is not None:
        n_pos = torch.tensor(
            float(n_pos_global), device=z.device, dtype=z.dtype
        ).clamp_min(1.0)
        n_neg = torch.tensor(
            float(n_neg_global), device=z.device, dtype=z.dtype
        ).clamp_min(1.0)
    else:
        n_pos = y_pos.sum().to(dtype=z.dtype).clamp_min(1.0)
        n_neg = y_neg.sum().to(dtype=z.dtype).clamp_min(1.0)

    r_pos = n_pos.pow(-0.25)
    r_neg = n_neg.pow(-0.25)
    denom = torch.maximum(r_pos, r_neg)
    m_pos = max_margin * r_pos / denom
    m_neg = max_margin * r_neg / denom

    z_adj = z.clone()
    z_adj = torch.where(y_pos, z - m_pos, z_adj)
    z_adj = torch.where(y_neg, z + m_neg, z_adj)

    return F.binary_cross_entropy_with_logits(z_adj, y_true, reduction=reduction)

def dice_loss(y_logits, y_true, batch_dict=None, smooth=1):
    y_pred = torch.sigmoid(y_logits)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice


def laplacian_loss(
    y_logits,
    edge_index,
    edge_weight=None,
    batch_dict=None,
    edge_types: Optional[Union[List[Tuple[str, str, str]], Tuple[str, str, str]]] = None,
    use_edge_weight=False,
):
    y_pred = torch.sigmoid(y_logits)
    loss = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
    for edge_type in _normalize_edge_types(edge_types):
        e_index = edge_index[edge_type]
        y_pred_diff = y_pred[e_index[0]] - y_pred[e_index[1]]
        diff_norm = y_pred_diff.abs().squeeze(-1).square()
        if use_edge_weight and edge_weight is not None and edge_type in edge_weight:
            e_weight = edge_weight[edge_type].squeeze(-1).reshape(-1)
            loss = loss + torch.dot(diff_norm, e_weight)
        else:
            loss = loss + diff_norm.sum()
    loss = loss / len(y_pred)
    return loss


def graphsage_unsupervised_loss(
    y_logits,
    y_true,
    edge_index,
    batch_dict=None,
    edge_types: Optional[Union[List[Tuple[str, str, str]], Tuple[str, str, str]]] = None,
    p=2,
):
    z = y_logits
    loss = torch.zeros((), device=z.device, dtype=z.dtype)
    for edge_type in _normalize_edge_types(edge_types):
        ei = edge_index[edge_type]
        link_logits = (z[ei[0]] * z[ei[1]]).sum(dim=-1, keepdim=True)
        link_true = (y_true[ei[0]] * y_true[ei[1]]).sum(dim=-1, keepdim=True)
        link_true = (link_true > 0).to(link_true.dtype)
        loss = loss + bce_loss(link_logits, link_true)
    return loss


def threshold_collapse_loss(threshold):
    return (threshold**2).mean()


def threshold_chasing_score_loss(threshold=0):
    return threshold.var()


loss_fcn_types = {
    "bce": bce_loss,
    "mse": mse_loss,
    "kld": kld_loss,
    "focal": focal_loss,
    "soft_focal": soft_focal_loss,
    "ldam": ldam_loss,
    "dice": dice_loss,
    "laplacian": laplacian_loss,
    "graphsage_unsupervised": graphsage_unsupervised_loss,
    "threshold_collapse": threshold_collapse_loss,
    "threshold_chasing_score": threshold_chasing_score_loss,
}


def get_lost_fcn(loss_type: Optional[Union[str, List[str], dict]] = None) -> Dict[str, Callable]:
    if loss_type is None:
        loss_type = ["bce"]
    if isinstance(loss_type, str):
        key = loss_type.lower()
        if key not in loss_fcn_types:
            raise ValueError(f"Invalid loss type: {loss_type}")
        return {key: loss_fcn_types[key]}
    if isinstance(loss_type, list):
        out: Dict[str, Callable] = {}
        for lt in loss_type:
            key = lt.lower()
            if key not in loss_fcn_types:
                raise ValueError(f"Invalid loss type: {lt}")
            out[key] = loss_fcn_types[key]
        return out
    if isinstance(loss_type, dict):
        out: Dict[str, Callable] = {}
        for lt in loss_type.keys():
            key = lt.lower()
            if key not in loss_fcn_types:
                raise ValueError(f"Invalid loss type: {lt}")
            out[key] = loss_fcn_types[key]
        return out
    raise ValueError(f"Invalid loss type: {loss_type}")
