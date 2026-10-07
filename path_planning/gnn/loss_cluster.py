"""Self-supervised losses for the cluster-GNN encoder (cluster.md §10-14).

The encoder outputs embeddings z (N, d); every loss reads its supervision
straight off the HeteroData batch:
  - graph_reconstruction: ('node','to','node') edges (+ negative sampling)
  - shortest_path: ('node','sp','node') sampled pairs and, optionally, the
    ('node','approx','node') start/goal-to-all Dijkstra pairs
  - cluster: soft K-means compactness on non-start/goal nodes, weighted by
    w_i = 1 + alpha*exp(-(d_i^B)^2 / 2 sigma^2) with d_i^B = batch['node'].y
    (the 'cluster' target space). Default weight 0 per milestone 3's staging.

Each entry of the config dict {name: {"weight": w, "args": {...}}} is
independently toggleable (§13); weight 0 skips computation entirely.
"""
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


def graph_reconstruction_loss(z: torch.Tensor, batch, num_neg_per_pos: int = 1):
    """L_graph (§10.1): BCE on sigma(z_i . z_j) over roadmap edges vs sampled
    non-edges. AddSelfLoops' (i,i) entries are dropped from the positives.
    Negatives are sampled over the whole (possibly batched) node set — cross-
    graph pairs are trivially unconnected, i.e. easy extra negatives."""
    edge_index = batch['node', 'to', 'node'].edge_index
    pos = edge_index[:, edge_index[0] != edge_index[1]]
    if pos.size(1) == 0:
        return z.sum() * 0.0
    neg = negative_sampling(pos, num_nodes=z.size(0),
                            num_neg_samples=pos.size(1) * num_neg_per_pos)
    pos_logits = (z[pos[0]] * z[pos[1]]).sum(dim=-1)
    neg_logits = (z[neg[0]] * z[neg[1]]).sum(dim=-1)
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return F.binary_cross_entropy_with_logits(logits, labels)


def shortest_path_loss(z: torch.Tensor, batch, use_start_goal_pairs: bool = True):
    """L_SP (§11): MSE between ||z_i - z_j|| and the (min-max normalized) graph
    shortest-path distance, over the sampled 'sp' pairs and optionally the
    task-aware start/goal-to-waypoint 'approx' pairs."""
    relations = [('node', 'sp', 'node')]
    if use_start_goal_pairs:
        relations.append(('node', 'approx', 'node'))
    losses = []
    for rel in relations:
        if rel not in batch.edge_types:
            continue
        store = batch[rel]
        ei, ea = store.edge_index, store.edge_attr
        mask = ei[0] != ei[1]  # 'approx' carries AddSelfLoops (i,i) entries
        ei, ea = ei[:, mask], ea[mask]
        if ei.size(1) == 0:
            continue
        d_hat = (z[ei[0]] - z[ei[1]]).norm(dim=-1)
        losses.append(F.mse_loss(d_hat, ea.view(-1)))
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def cluster_loss(z: torch.Tensor, batch, num_clusters: int = 64,
                 temperature: float = 1.0, alpha: float = 1.0, sigma: float = 1.0):
    """L_cluster (§12): soft K-means compactness, per graph in the batch.
    Start/goal nodes are excluded (§5 — protected singletons). Centroids are
    seeded from K random embeddings (detached), assignments
    q = softmax(-||z-mu||^2 / T), then mu recomputed as the q,w-weighted mean
    (differentiable) and compactness sum q_ik w_i ||z_i - mu_k||^2 taken.
    Never use alone — combine with a representation-preserving loss (§12)."""
    x = batch['node'].x
    dim = x.size(1) - 3  # [pos | start/goal | free | boundary]
    sg_mask = x[:, dim] > 0.5
    y = batch['node'].y.view(-1)
    w_all = 1.0 + alpha * torch.exp(-(y ** 2) / (2.0 * sigma ** 2))
    batch_vec = batch['node'].batch if hasattr(batch['node'], 'batch') and batch['node'].batch is not None \
        else torch.zeros(z.size(0), dtype=torch.long, device=z.device)

    losses = []
    for g in batch_vec.unique():
        m = (batch_vec == g) & ~sg_mask
        zg, wg = z[m], w_all[m]
        if zg.size(0) <= num_clusters:
            continue
        seed = torch.randperm(zg.size(0), device=z.device)[:num_clusters]
        mu = zg[seed].detach()
        d2 = torch.cdist(zg, mu).pow(2)
        q = torch.softmax(-d2 / temperature, dim=1)
        qw = q * wg.unsqueeze(1)
        mu = (qw.t() @ zg) / qw.sum(dim=0).unsqueeze(1).clamp_min(1e-12)
        d2 = torch.cdist(zg, mu).pow(2)
        losses.append((q * wg.unsqueeze(1) * d2).sum() / wg.sum().clamp_min(1e-12))
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


CLUSTER_LOSS_FCNS = {
    'graph_reconstruction': graph_reconstruction_loss,
    'shortest_path': shortest_path_loss,
    'cluster': cluster_loss,
}


class ClusterLossFunction:
    """Weighted sum of the configured self-supervised losses.

    loss_config: {name: {"weight": float, "args": {...} | None}}. Returns
    (total, {name: float}) so callers can log the components."""

    def __init__(self, loss_config: Dict[str, Any]):
        unknown = set(loss_config) - set(CLUSTER_LOSS_FCNS)
        if unknown:
            raise ValueError(f"Unknown cluster losses: {sorted(unknown)}")
        self.weights = {k: float(v.get('weight', 1.0)) for k, v in loss_config.items()}
        self.args = {k: dict(v.get('args') or {}) for k, v in loss_config.items()}

    def __call__(self, z: torch.Tensor, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = z.sum() * 0.0
        parts: Dict[str, float] = {}
        for name, weight in self.weights.items():
            if weight == 0.0:
                continue
            value = CLUSTER_LOSS_FCNS[name](z, batch, **self.args[name])
            total = total + weight * value
            parts[name] = float(value.detach())
        return total, parts
