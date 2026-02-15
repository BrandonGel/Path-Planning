import os
import warnings
import logging

# Suppress torch internal logging warnings BEFORE importing torch
os.environ['TORCH_LOGS'] = '-all'
os.environ['TORCHDYNAMO_VERBOSE'] = '0'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'

from pathlib import Path
import argparse
import wandb
import numpy as np
import torch
from time import time
from path_planning.gnn.dataloader import get_graph_dataset_file_paths, GraphDataset
from path_planning.gnn.train import split_dataset
from torch_geometric.nn import to_hetero
from path_planning.gnn.loss import LossFunction
from path_planning.gnn.train import train,test
from multiprocessing import cpu_count
from path_planning.gnn.model import get_model
from path_planning.gnn.optimizer import get_optimizer
from path_planning.utils.util import set_global_seed,set_train_config

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.fx.experimental.symbolic_shapes')
warnings.filterwarnings('ignore', message='.*_maybe_guard_rel.*')
# Suppress torch loggers
logging.getLogger('torch._dynamo').setLevel(logging.CRITICAL)
logging.getLogger('torch._inductor').setLevel(logging.CRITICAL)
logging.getLogger('torch.fx.experimental.symbolic_shapes').setLevel(logging.CRITICAL)
torch.set_float32_matmul_precision('medium')


def run_train(train_config: dict,num_workers: int = None, use_cuda: bool = True):    
    # Setting the seed & device
    set_global_seed(train_config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # Setting the dataset folder paths, load file, and save file
    folder_path = [Path(p) for p in train_config['dataset']['folder_path']] if train_config['dataset']['folder_path'] is not None else None
    load_file = Path(train_config['dataset']['load_file']) if train_config['dataset']['load_file'] is not None else None
    save_file = Path(train_config['dataset']['save_file']) if train_config['dataset']['save_file'] is not None else None

    # Getting the dataset config
    config = train_config['dataset']['config']
    num_workers = num_workers if num_workers is not None else cpu_count()

    # Getting the dataset file paths
    data_files = get_graph_dataset_file_paths(folder_path,config)
    graph_dataset =GraphDataset(data_files,load_file=load_file,save_file=save_file,num_hops=-1,num_workers=num_workers)
    if save_file is not None:
        graph_dataset.save()

    # Splitting the dataset into train and test
    batch_size = train_config['train']['batch_size']    
    test_size = train_config['train']['test_size']
    idx_train,idx_test ,train_loader,test_loader =  split_dataset(graph_dataset,batch_size=batch_size,test_size=test_size)

    # Setting the loss function
    loss_type = train_config['loss']['type']
    loss_fcn_weights = train_config['loss']['weights']
    loss_args = train_config['loss']['args']
    loss_fcn = LossFunction(loss_type=loss_type,loss_fcn_weights=loss_fcn_weights,loss_args=loss_args)


    # instanziate the model
    train_config['model']['in_channels'] = graph_dataset[0].x_dict['node'].shape[1] 
    model = get_model(model_type=train_config['model']['type'],**train_config['model'])
    model_in_channels = model.in_channels
    model = to_hetero(model,graph_dataset[0].metadata(),aggr=train_config['model']['to_hetero_aggr']).to(device)
    
    # Initialize lazy modules before creating optimizer
    if model_in_channels == -1:
        with torch.no_grad():  # Initialize lazy modules.
            graph_dataset[0].to(device).edge_attr_dict
            x_dict = graph_dataset[0].to(device).x_dict
            edge_index_dict = graph_dataset[0].to(device).edge_index_dict
            edge_attr_dict = graph_dataset[0].to(device).edge_attr_dict
            out = model(x_dict, edge_index_dict,edge_attr_dict)

    # Compile the model
    compile =train_config['model']['compile']
    compile_dynamic = train_config['model']['compile_dynamic']
    if compile:
        model = torch.compile(model, dynamic=compile_dynamic)
    model = model.to(device)
    
    # Resume the training if there is a loaded model
    resume_epoch = train_config['train']['resume_epoch']
    model_load_folder = train_config['train']['load_folder']
    if resume_epoch > 0 and model_load_folder is not None:
        model_load_file = f"epoch_{resume_epoch}.pth"        
        model_load_path = os.path.join(model_load_folder, model_load_file)        
        model.load_state_dict(torch.load(model_load_path))
    
    # Initialize the optimizer
    optimizer_type = train_config['optimizer']['type']
    optimizer = get_optimizer(optimizer_type=optimizer_type,model_weights=model.parameters(),**train_config['optimizer'])

    # Set the model save folder
    model_type_str = train_config['model']['type']
    model_save_folder = Path(train_config['train']['save_folder']) / f"{model_type_str}_{"compile" if args.compile else ""}"
    os.makedirs(model_save_folder, exist_ok=True)

    # Initialize the training
    num_epochs = max(0,resume_epoch)+train_config['train']['num_epochs']
    save_epoch = train_config['train']['save_epoch']
    verbose_epoch = train_config.get('train',{}).get('verbose_epoch',10)
    with wandb.init(dir = model_save_folder,project="path-planning", name=f"{model_type_str}_{"compile" if args.compile else ""}", config=train_config) as run:

        model_save_version_folder = os.path.join(wandb.run.dir, "model")
        os.makedirs(model_save_version_folder, exist_ok=True)
        
        run.define_metric("epoch/train_loss", step_metric="epoch")
        run.define_metric("epoch/test_loss", step_metric="epoch")
        run.define_metric("epoch/train_time", step_metric="epoch")
        run.define_metric("epoch/test_time", step_metric="epoch")
        run.define_metric("runtime", step_metric="epoch")
        start_time = time()
        for epoch in range(resume_epoch + 1, num_epochs + 1):
            st = time()
            train_loss,train_time = train(train_loader,model,optimizer,device=device,loss_function=loss_fcn,run=run)
            et = time()
            test_loss,test_time = test(test_loader,model,device=device,loss_function=loss_fcn,run=run)
            tt = time()
            epoch_train_time = et-st
            epoch_test_time = tt-et
            runtime = time() - start_time

            run.log({
                "epoch": epoch,
                "epoch/train_loss": np.mean(train_loss),
                "epoch/test_loss": np.mean(test_loss),
                "epoch/train_time": epoch_train_time,
                "epoch/test_time": epoch_test_time,
                "runtime": runtime,
            }, commit=False
            )
            if verbose_epoch > 0 and epoch % verbose_epoch == 0:
                print(
                    "Runtime: %.2f" % runtime,
                    "\tEpoch: %d" % epoch,
                    '\tTrain loss: %.4f' % np.mean(train_loss),
                    '\tTest loss: %.4f' % np.mean(test_loss),
                    '\tTrain time: %.4f' % epoch_train_time,
                    '\tTest time: %.4f' % epoch_test_time)
            if epoch % save_epoch == 0:
                model_save_file = f"epoch_{epoch}.pth"
                model_save_path = os.path.join(model_save_version_folder, model_save_file)
                torch.save(model.state_dict(), model_save_path)
                run.log_model(model_save_path, name=model_save_file)

        model_save_file = f"epoch_{num_epochs}.pth"
        model_save_path = os.path.join(model_save_version_folder, model_save_file)
        torch.save(model.state_dict(), model_save_path)
        run.log_model(model_save_path, name=model_save_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--folder_paths",type=str, nargs='+', help="folder paths of map and graph data")
    parser.add_argument("-l","--load_file",type=str, default=None, help="load file")
    parser.add_argument("-s","--save_file",type=str, default=None, help="save file")
    parser.add_argument("-cf","--config_file",type=str, default='config/train.yaml', help="config file")
    parser.add_argument("-ds","--use_discrete_space",type=bool, help="use discrete space")
    parser.add_argument("-rmt","--road_map_type",type=str, help="road map type")
    parser.add_argument("-ts","--target_space",type=str, help="target space")    
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("-cuda","--cuda",type=bool, default=True, help="use cuda")
    parser.add_argument("-train","--train_config",type=str, default='config/train.yaml', help="train config file")
    parser.add_argument("-compile","--compile",type=bool, default=True, help="compile model")
    parser.add_argument("-dynamic","--compile_dynamic",type=bool, default=True, help="compile dynamic mode")
    parser.add_argument("-seed","--seed",type=int, default=42, help="seed")
    args = parser.parse_args()
    

    # Setting the train config    
    folder_path = [
        # Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map32x32_resolution1.0/agents8_obst0.1'),
        # Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32.0x32.0_resolution1.0/agents1_obst0.1'),
        # Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32.0x32.0_resolution1.0/agents2_obst0.1'),
        # Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32.0x32.0_resolution1.0/agents4_obst0.1'),
        # Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map32.0x32.0_resolution1.0/agents8_obst0.1'),
        # Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map32.0x32.0_resolution1.0/agents16_obst0.1'),
        # Path('/home/bho36/Documents/code/Path-Planning/benchmark/train/map16.0x16.0_resolution1.0/agents4_obst0.1'),
        Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32x32_resolution1.0/agents1_obst0.1'),
        Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32x32_resolution1.0/agents2_obst0.1'),
        Path('/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map32x32_resolution1.0/agents4_obst0.1'),
    ]
    import yaml
    with open(args.config_file, 'r') as f:
        train_config = yaml.load(f,Loader=yaml.FullLoader)
    train_config['dataset']['folder_path'] = folder_path
    train_config = set_train_config(train_config,args)

    wandb.login()
    run_train(train_config,num_workers=args.num_workers, use_cuda=args.cuda)