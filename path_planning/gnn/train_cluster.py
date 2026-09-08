"""Training loop for the self-supervised cluster-GNN encoder (cluster.md
milestones 2-3). Mirrors the maintained scripts/train/run_train.py idioms
(get_model -> to_hetero -> lazy init -> compile -> resume -> optimizer; wandb
logging; epoch_{n}.pth checkpoints) but the objective is ClusterLossFunction
on embeddings — there is no y-score supervision."""
import os
from pathlib import Path
from time import time
from typing import List, Tuple

import numpy as np
import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero

from path_planning.gnn.dataloader_cluster import (
    GraphDataset,
    get_dummy_sample_data,
    get_graph_dataset_file_paths,
)
from path_planning.gnn.loss_cluster import ClusterLossFunction
from path_planning.gnn.model import get_model
from path_planning.gnn.optimizer import get_optimizer
from path_planning.utils.util import set_global_seed


def split_dataset_by_case(graph_dataset: GraphDataset, data_files: List[Tuple[Path, Path]],
                          batch_size: int = 8, test_size: float = 0.1,
                          random_state: int = 42, num_workers: int = 1):
    """Leakage-safe split (cluster.md §36): partition CASE directories, so a
    map's graph samples and its 4 rotation augmentations never straddle the
    train/val boundary. data_files order must match graph_dataset order
    (GraphDataset preserves it — pool.imap is ordered)."""
    # graph.npz path: case_N/sample_cluster/<rmt>/graph_i_aug/graph.npz
    cases = [Path(gf).parents[3] for gf, _ in data_files]
    unique_cases = sorted(set(cases))
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(unique_cases))
    n_test = int(round(test_size * len(unique_cases))) if len(unique_cases) > 1 else 0
    test_cases = {unique_cases[i] for i in perm[:n_test]}
    idx_train = [i for i, c in enumerate(cases) if c not in test_cases]
    idx_test = [i for i, c in enumerate(cases) if c in test_cases]
    print(f"Case split: {len(unique_cases) - len(test_cases)} train / {len(test_cases)} val cases "
          f"({len(idx_train)} / {len(idx_test)} samples)")

    train_loader = DataLoader(
        graph_dataset[idx_train],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(idx_train) >= batch_size,
    )
    test_loader = None
    if idx_test:
        test_loader = DataLoader(
            graph_dataset[idx_test],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    return idx_train, idx_test, train_loader, test_loader


def _forward_embeddings(model, batch):
    out = model(dict(batch.x_dict), batch.edge_index_dict, batch.edge_attr_dict, batch.batch_dict)
    return out['node']


def train_cluster_epoch(loader, model, optimizer, loss_fcn, device, run=None):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z = _forward_embeddings(model, batch)
        loss, parts = loss_fcn(z, batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += float(loss.detach())
        n += 1
        if run is not None:
            run.log({"batch/train_loss": float(loss.detach()),
                     **{f"batch/train_{k}": v for k, v in parts.items()}})
    return total / max(n, 1)


@torch.no_grad()
def test_cluster_epoch(loader, model, loss_fcn, device, run=None):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        z = _forward_embeddings(model, batch)
        loss, parts = loss_fcn(z, batch)
        total += float(loss.detach())
        n += 1
        if run is not None:
            run.log({"batch/test_loss": float(loss.detach()),
                     **{f"batch/test_{k}": v for k, v in parts.items()}})
    return total / max(n, 1)


@torch.no_grad()
def save_embedding_snapshot(loader, model, device, out_file, num_clusters: int = 32):
    """Debuggability (cluster.md §46): dump one batch's embeddings plus a quick
    sklearn K-means labeling so training progress can be inspected offline."""
    from sklearn.cluster import KMeans

    model.eval()
    batch = next(iter(loader)).to(device)
    z = _forward_embeddings(model, batch).cpu().numpy()
    x = batch['node'].x.cpu().numpy()
    y = batch['node'].y.view(-1).cpu().numpy()
    k = min(num_clusters, max(2, len(z) - 1))
    labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(z)
    np.savez_compressed(out_file, embeddings=z, kmeans_labels=labels,
                        node_features=x, boundary_distance=y)


def set_cluster_model(model_config: dict, train_config: dict, data_sample, device: torch.device):
    """run_train.set_model adapted for the encoder: metadata comes from
    get_dummy_sample_data() (to/approx/boundary only) so the supervision-only
    ('node','sp','node') relation never enters the to_hetero module tree."""
    loss_fcn = ClusterLossFunction(model_config['loss'])

    model_kwargs = dict(model_config['model'])
    homogeneous_model = get_model(model_type=model_kwargs['type'], **model_kwargs)
    model_in_channels = homogeneous_model.node_in_channels
    dim = data_sample['node'].x.shape[1] - 3
    metadata = get_dummy_sample_data(dim=dim).metadata()
    model = to_hetero(homogeneous_model, metadata, aggr=model_config['model']['to_hetero_aggr']).to(device)

    if model_in_channels == -1:
        with torch.no_grad():  # initialize lazy modules
            ds = data_sample.to(device)
            model(dict(ds.x_dict), ds.edge_index_dict, ds.edge_attr_dict, ds.batch_dict)

    compile_model = train_config['device']['compile']
    if compile_model:
        model = torch.compile(model, dynamic=train_config['device']['compile_dynamic'])
    model = model.to(device)

    resume_epoch = train_config['train']['resume_epoch']
    model_load_folder = train_config['train']['load_folder']
    if resume_epoch > 0 and model_load_folder is not None:
        model.load_state_dict(torch.load(os.path.join(model_load_folder, f"epoch_{resume_epoch}.pth")))

    optimizer = get_optimizer(optimizer_type=model_config['optimizer']['type'],
                              model_weights=model.parameters(), **model_config['optimizer'])

    model_type_str = model_config['model']['type']
    model_save_folder = Path(train_config['train']['save_folder']) / f"{model_type_str}{'_compile' if compile_model else ''}"
    os.makedirs(model_save_folder, exist_ok=True)
    return model, optimizer, loss_fcn, model_save_folder, model_type_str, resume_epoch


def run_cluster_train(train_config: dict, num_workers: int = None, use_cuda: bool = True,
                      online: bool = False):
    set_global_seed(train_config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    num_workers = num_workers if num_workers is not None else 1

    dataset_cfg = train_config['dataset']
    folder_path = [Path(p) for p in dataset_cfg['folder_path']]
    load_file = Path(dataset_cfg['load_file']) if dataset_cfg.get('load_file') else None
    save_file = Path(dataset_cfg['save_file']) if dataset_cfg.get('save_file') else None
    data_files = get_graph_dataset_file_paths(folder_path, dataset_cfg['config'])
    graph_dataset = GraphDataset(data_files, load_file=load_file, save_file=save_file,
                                 num_hops=-1, num_workers=num_workers)

    batch_size = train_config['train']['batch_size']
    idx_train, idx_test, train_loader, test_loader = split_dataset_by_case(
        graph_dataset, data_files, batch_size=batch_size,
        test_size=train_config['train']['test_size'],
        random_state=train_config['seed'], num_workers=num_workers)

    dummy_batch_size = min(4, len(idx_train))
    dummy_batch = next(iter(DataLoader(graph_dataset[idx_train[:dummy_batch_size]],
                                       batch_size=dummy_batch_size))).to(device)
    model, optimizer, loss_fcn, model_save_folder, model_type_str, resume_epoch = \
        set_cluster_model(train_config['encoder'], train_config, dummy_batch, device)

    num_epochs = max(0, resume_epoch) + train_config['train']['num_epochs']
    save_epoch = train_config['train']['save_epoch']
    verbose_epoch = train_config['train'].get('verbose_epoch', 10)
    snapshot_loader = test_loader if test_loader is not None else train_loader

    with wandb.init(mode="online" if online else "offline", dir=model_save_folder,
                    project="path-planning-cluster", name=f"cluster_{model_type_str}",
                    config=train_config) as run:
        model_version_folder = os.path.join(wandb.run.dir, "model")
        embeddings_folder = os.path.join(wandb.run.dir, "embeddings")
        os.makedirs(model_version_folder, exist_ok=True)
        os.makedirs(embeddings_folder, exist_ok=True)
        run.define_metric("epoch/train_loss", step_metric="epoch")
        run.define_metric("epoch/test_loss", step_metric="epoch")
        start_time = time()

        for epoch in range(resume_epoch + 1, num_epochs + 1):
            train_loss = train_cluster_epoch(train_loader, model, optimizer, loss_fcn, device, run)
            log = {"epoch": epoch, "epoch/train_loss": train_loss, "runtime": time() - start_time}
            if test_loader is not None:
                log["epoch/test_loss"] = test_cluster_epoch(test_loader, model, loss_fcn, device, run)
            run.log(log)
            if verbose_epoch and epoch % verbose_epoch == 0:
                msg = f"epoch {epoch}: train {train_loss:.4f}"
                if "epoch/test_loss" in log:
                    msg += f", val {log['epoch/test_loss']:.4f}"
                print(msg)
            if epoch % save_epoch == 0 or epoch == num_epochs:
                torch.save(model.state_dict(), os.path.join(model_version_folder, f"epoch_{epoch}.pth"))
                save_embedding_snapshot(snapshot_loader, model, device,
                                        os.path.join(embeddings_folder, f"epoch_{epoch}.npz"))
    return model, model_save_folder
