from sklearn.model_selection import train_test_split
from path_planning.gnn.dataloader import GraphDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch
import wandb
from time import time
from path_planning.gnn.loss import LossFunction
from typing import Dict
from path_planning.gnn.loss import get_probability

def split_dataset(graph_dataset:GraphDataset,batch_size=128,test_size=0.1,random_state=42,num_workers=1):

    idx_train,idx_test = train_test_split(range(len(graph_dataset)),test_size=test_size,random_state=random_state)
    train_dataset = graph_dataset[idx_train]
    test_dataset = graph_dataset[idx_test]

    print("Train set length: \t",len(train_dataset))
    print("Test set length: \t",len(test_dataset))

    # Keep the batch size constant so pooling output shapes stay stable.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    return idx_train,idx_test ,train_loader,test_loader

def get_dummy_sample(graph_dataset:GraphDataset,batch_size=1,num_workers=1,device='cuda:0'):
    dummy_loader = DataLoader(graph_dataset,batch_size=batch_size,num_workers=num_workers)
    dummy_batch = next(iter(dummy_loader)).to(device)
    return dummy_batch

def get_threshold_x_dict(x_dict:Dict[str,torch.Tensor],out:Dict[str,torch.Tensor]):
    x_dict['node'] = torch.concat(
        [
            x_dict['node'],
            torch.nn.functional.tanh(out['node']).reshape(-1, 1)],
        dim=1,
    )
    return x_dict

def threshold_graph_to_nodes(
    threshold: torch.Tensor, batch_dict: Dict[str, torch.Tensor],
    node_type: str = 'node'
) -> torch.Tensor:
    """Broadcast graph-level ``threshold`` (B, *) to per-node (N, *) using ``batch_dict['node']``."""
    idx = batch_dict[node_type]
    if threshold.dim() == 1:
        return threshold[idx].unsqueeze(-1)
    return threshold[idx]

def normalize_threshold(threshold: torch.Tensor,batch_dict: Dict[str, torch.Tensor], logits:torch.Tensor, node_type: str = 'node'):
    idx = batch_dict[node_type]
    logits_means = torch.stack([logits[node_type][idx==ii].mean(dim=0) for ii in torch.unique(idx)])
    new_threshold = threshold - logits_means
    return new_threshold
    

def train(train_loader,model,optimizer,loss_function:LossFunction,threshold_model=None,num_prune=0,device='cuda:0',run: wandb.Run =None):
    model.train()
    # if threshold_model is not None:
    #     threshold_model.eval()
    train_loss = []
    train_time = []
    for batch in train_loader:
        for _ in range(num_prune+1):
            st = time()
            optimizer.zero_grad()
            device_batch = batch.to(device)
            out = model(device_batch.x_dict, device_batch.edge_index_dict,device_batch.edge_attr_dict,device_batch.batch_dict)
            
            additional_args = {}
            loss = loss_function(out['node'],
                                device_batch['node'].y.reshape(-1,1),
                                device_batch.edge_index_dict,
                                device_batch.edge_attr_dict,
                                device_batch.edge_weight_dict,
                                device_batch.batch_dict,
                                additional_args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            et = time()
            train_time.append(et-st)
            train_loss.append(float(loss.item()))
            if run is not None:
                run.log({
                    "batch/train_loss": train_loss[-1],
                    "batch/train_time": train_time[-1],
                })
    return train_loss,train_time

def test(test_loader,model,loss_function:LossFunction,threshold_model=None,num_prune=0,device='cuda:0',run: wandb.Run =None):
    model.eval()
    # if threshold_model is not None:
    #     threshold_model.eval()
    test_loss = []
    test_time = []
    with torch.no_grad():
        for batch in test_loader:
            for _ in range(num_prune+1):
                st = time()
                device_batch = batch.to(device)
                out = model(device_batch.x_dict, device_batch.edge_index_dict,device_batch.edge_attr_dict,device_batch.batch_dict)

                additional_args = {}
                loss = loss_function(out['node'],
                                device_batch['node'].y.reshape(-1,1),
                                device_batch.edge_index_dict,
                                device_batch.edge_attr_dict,
                                device_batch.edge_weight_dict,
                                device_batch.batch_dict,
                                additional_args)
                et = time()
                test_time.append(et-st)
                test_loss.append(float(loss.item()))
                if run is not None:
                    run.log({
                        "batch/test_loss": test_loss[-1],
                        "batch/test_time": test_time[-1],
                    })
    return test_loss,test_time

def train_threshold(train_loader,threshold_model,threshold_optimizer,loss_function:LossFunction,model,num_prune=0,
                    device='cuda:0',run: wandb.Run =None):
    model.eval()
    threshold_model.train()
    train_loss = []
    train_time = []
    for batch in train_loader:
        for _ in range(num_prune+1):
            st = time()
            threshold_optimizer.zero_grad()
            device_batch = batch.to(device)
            with torch.no_grad():
                out = model(device_batch.x_dict, device_batch.edge_index_dict,device_batch.edge_attr_dict,device_batch.batch_dict)

            # Do not mutate `device_batch.x_dict` in-place, as the underlying
            # dataset objects may be reused by the DataLoader.
            x_dict = get_threshold_x_dict(device_batch.x_dict,out)
            threshold = threshold_model(x_dict, device_batch.edge_index_dict,
                                          device_batch.edge_attr_dict,
                                          device_batch.batch_dict)
            threshold = normalize_threshold(threshold, device_batch.batch_dict, out, 'node')
            additional_args = {
                'threshold': threshold_graph_to_nodes(threshold, device_batch.batch_dict),
            }

            loss = loss_function(
                out['node'],
                device_batch['node'].y.reshape(-1, 1),
                device_batch.edge_index_dict,
                device_batch.edge_attr_dict,
                device_batch.edge_weight_dict,
                device_batch.batch_dict,
                additional_args,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(threshold_model.parameters(), max_norm=5.0)
            threshold_optimizer.step()
            et = time()
            train_time.append(et-st)
            train_loss.append(float(loss.item()))
            if run is not None:
                run.log({
                    "batch/threshold_train_loss": train_loss[-1],
                    "batch/threshold_train_time": train_time[-1],
                })
    return train_loss,train_time

def test_threshold(test_loader,threshold_model,loss_function:LossFunction,model,num_prune=0,device='cuda:0',run: wandb.Run =None):
    model.eval()
    threshold_model.eval()
    test_loss = []
    test_time = []
    with torch.no_grad():
        for batch in test_loader:
            for _ in range(num_prune+1):
                st = time()
                device_batch = batch.to(device)
                out = model(device_batch.x_dict, device_batch.edge_index_dict,device_batch.edge_attr_dict,device_batch.batch_dict)

                x_dict = get_threshold_x_dict(device_batch.x_dict,out)
                threshold = threshold_model(x_dict, device_batch.edge_index_dict,
                                              device_batch.edge_attr_dict,
                                              device_batch.batch_dict)
                threshold = normalize_threshold(threshold, device_batch.batch_dict, out, 'node')
                additional_args = {
                    'threshold': threshold_graph_to_nodes(threshold, device_batch.batch_dict),
                }
                loss = loss_function(
                    out['node'],
                    device_batch['node'].y.reshape(-1, 1),
                    device_batch.edge_index_dict,
                    device_batch.edge_attr_dict,
                    device_batch.edge_weight_dict,
                    device_batch.batch_dict,
                    additional_args,
                )
                et = time()
                test_time.append(et-st)
                test_loss.append(float(loss.item()))
                if run is not None:
                    run.log({
                        "batch/threshold_test_loss": test_loss[-1],
                        "batch/threshold_test_time": test_time[-1],
                    })
    return test_loss,test_time