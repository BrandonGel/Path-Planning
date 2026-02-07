from sklearn.model_selection import train_test_split
from path_planning.gnn.dataloader import GraphDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

def split_dataset(graph_dataset:GraphDataset,batch_size=128,test_size=0.1,random_state=42):

    idx_train,idx_test = train_test_split(range(len(graph_dataset)),test_size=test_size,random_state=random_state)
    train_dataset = graph_dataset[idx_train]
    test_dataset = graph_dataset[idx_test]

    print("Train set length: \t",len(train_dataset))
    print("Test set length: \t",len(test_dataset))

    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size)

    return train_loader,test_loader



def train(train_loader,model,optimizer,device='cuda:0',loss_function=F.binary_cross_entropy_with_logits):
    model.train()
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = loss_function(out['node'],
                               batch['node'].y.reshape(-1,1))
        loss.backward()
        optimizer.step()

        total_examples += 1
        total_loss += float(loss.item())

    return total_loss / total_examples

def test(test_loader,model,optimizer,device='cuda:0',loss_function=F.binary_cross_entropy_with_logits):
    model.eval()
    total_examples = total_loss = 0
    for batch in test_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = loss_function(out['node'],
                               batch['node'].y.reshape(-1,1))
        loss.backward()
        optimizer.step()

        total_examples += 1
        total_loss += float(loss.item())

    return total_loss / total_examples