from sklearn.model_selection import train_test_split
from path_planning.gnn.dataloader import GraphDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch
import wandb
from time import time

def split_dataset(graph_dataset:GraphDataset,batch_size=128,test_size=0.1,random_state=42,num_workers=1):

    idx_train,idx_test = train_test_split(range(len(graph_dataset)),test_size=test_size,random_state=random_state)
    train_dataset = graph_dataset[idx_train]
    test_dataset = graph_dataset[idx_test]

    print("Train set length: \t",len(train_dataset))
    print("Test set length: \t",len(test_dataset))

    train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers)

    return idx_train,idx_test ,train_loader,test_loader



def train(train_loader,model,optimizer,device='cuda:0',loss_function=F.binary_cross_entropy_with_logits,run: wandb.Run =None):
    model.train()
    train_loss = []
    train_time = []
    for batch in train_loader:
        st = time()
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict,batch.edge_attr_dict)
        loss = loss_function(out['node'],
                               batch['node'].y.reshape(-1,1),
                               batch.edge_index_dict)
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

def test(test_loader,model,device='cuda:0',loss_function=F.binary_cross_entropy_with_logits,run: wandb.Run =None):
    model.eval()
    test_loss = []
    test_time = []
    with torch.no_grad():
        for batch in test_loader:
            st = time()
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict,batch.edge_attr_dict)
            loss = loss_function(out['node'],
                               batch['node'].y.reshape(-1,1),
                               batch.edge_index_dict)
            et = time()
            test_time.append(et-st)
            test_loss.append(float(loss.item()))
            if run is not None:
                run.log({
                    "batch/test_loss": test_loss[-1],
                    "batch/test_time": test_time[-1],
                })
    return test_loss,test_time