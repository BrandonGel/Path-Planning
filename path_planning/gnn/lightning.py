import pytorch_lightning as pl
from path_planning.gnn.optimizer import get_optimizer
from path_planning.gnn.loss import LossFunction
from path_planning.gnn.model import get_model
from path_planning.gnn.dataloader import get_graph_dataset_file_paths, GraphDataset
import os
import torch
from pathlib import Path
from torch_geometric.nn import to_hetero
from path_planning.gnn.train import split_dataset
import warnings
from multiprocessing import cpu_count
AVAIL_GPUS = min(1, torch.cuda.device_count())

warnings.filterwarnings("ignore", category=UserWarning, module='torch.fx.experimental.symbolic_shapes')
warnings.filterwarnings("ignore", message='.*_maybe_guard_rel.*')

class GraphGNNLightning(pl.LightningModule):
    def __init__(self, model,graph_dataset,train_config):
        super().__init__()
        self.train_config = train_config
        self.model = model
        self.loss_function = self.configure_loss_function()
        self.batch_size = train_config['train']['batch_size']

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        out = self.model(x_dict, edge_index_dict,edge_attr_dict)
        return out

    def configure_optimizers(self):
        optimizer_type = self.train_config['optimizer']['type']
        optimizer = get_optimizer(optimizer_type=optimizer_type,model_weights=self.model.parameters(),**self.train_config['optimizer'])
        return optimizer

    def configure_loss_function(self):
        loss_type = self.train_config['loss']['type']
        loss_fcn_weights = self.train_config['loss']['weights']
        loss_args = self.train_config['loss']['args']
        loss_fcn = LossFunction(loss_type=loss_type,loss_fcn_weights=loss_fcn_weights,loss_args=loss_args)
        return loss_fcn

    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch)
        pred_logits = out['node']
        true_labels = train_batch['node'].y.view(-1,1)
        edge_index = train_batch.edge_index_dict
        loss = self.loss_function(pred_logits, true_labels, edge_index)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch)
        pred_logits = out['node']
        true_labels = val_batch['node'].y.view(-1,1)
        edge_index = val_batch.edge_index_dict
        loss = self.loss_function(pred_logits, true_labels, edge_index)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def test_step(self, test_batch, batch_idx):
        out = self.forward(test_batch)
        pred_logits = out['node']
        true_labels = test_batch['node'].y.view(-1,1)
        edge_index = test_batch.edge_index_dict
        loss = self.loss_function(pred_logits, true_labels, edge_index)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
    
def train_graph(train_config,num_workers: int = None,use_cuda: bool = True):
    folder_path = [Path(p) for p in train_config['dataset']['folder_path']] if train_config['dataset']['folder_path'] is not None else None
    load_file = Path(train_config['dataset']['load_file']) if train_config['dataset']['load_file'] is not None else None
    save_file = Path(train_config['dataset']['save_file']) if train_config['dataset']['save_file'] is not None else None
    config = train_config['dataset']['config']
    
    num_workers = num_workers if num_workers is not None else cpu_count()
    data_files = get_graph_dataset_file_paths(folder_path,config)
    graph_dataset =GraphDataset(data_files,load_file=load_file,save_file=save_file,num_hops=-1,num_workers=num_workers)
    if save_file is not None:
        graph_dataset.save()

    batch_size = train_config['train']['batch_size']    
    test_size = train_config['train']['test_size']
    _,_ ,train_loader,test_loader =  split_dataset(graph_dataset,batch_size=batch_size,test_size=test_size)

    save_folder = train_config['train']['save_folder']
    load_folder = train_config['train']['load_folder']
    num_epochs = train_config['train']['num_epochs']
    resume_epoch = train_config['train']['resume_epoch']
    save_epoch = train_config['train']['save_epoch']
    os.makedirs(save_folder, exist_ok=True)
    gradient_clip_val = train_config['train']['gradient_clip_val']


    batch_size = train_config['train']['batch_size']    
    test_size = train_config['train']['test_size']
    _,_ ,train_loader,test_loader =  split_dataset(graph_dataset,batch_size=batch_size,test_size=test_size,num_workers=num_workers)

    # instanziate the trainer
    os.makedirs(save_folder, exist_ok=True)
    trainer = pl.Trainer(
        profiler="simple",
        default_root_dir=save_folder,
        precision=16, 
        accelerator="auto" if use_cuda == True else "cpu",
        devices=AVAIL_GPUS,
        max_epochs=num_epochs,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=False,
        log_every_n_steps=10,  # Lower value to ensure logging happens even with small batch counts
    )

    # instansiate the model
    model = get_model(model_type=train_config['model']['type'],**train_config['model'])
    model = to_hetero(model,graph_dataset[0].metadata(),aggr=train_config['model']['to_hetero_aggr'])
    graph_gnn = GraphGNNLightning(model,graph_dataset,train_config)
    with torch.no_grad():  # Initialize lazy modules.
        batch = next(iter(train_loader)).to(graph_gnn.device)
        graph_gnn(batch)
    if train_config['model']['compile']:
        graph_gnn = torch.compile(graph_gnn,dynamic=train_config['model']['compile_dynamic'])
    else:
        graph_gnn = graph_gnn.to(graph_gnn.device)

    if resume_epoch > 0:
        print(f"Resuming training from epoch {resume_epoch}")
        model_file = os.path.join(load_folder, f"{graph_gnn}_model_{resume_epoch}.pth")
        graph_gnn = GraphGNNLightning.load_from_checkpoint(model_file)
    else:
        trainer.fit(graph_gnn,train_loader,test_loader)
    test_results = trainer.test(graph_gnn,dataloaders=test_loader,verbose=False)
    return test_results

