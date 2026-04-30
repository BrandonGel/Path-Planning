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
from path_planning.gnn.train import train,test,train_threshold,test_threshold,get_dummy_sample
from multiprocessing import cpu_count
from path_planning.gnn.model import get_model
from path_planning.gnn.optimizer import get_optimizer
from path_planning.utils.util import set_global_seed,set_train_config
import yaml
from torch_geometric.data import HeteroData

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.fx.experimental.symbolic_shapes')
warnings.filterwarnings('ignore', message='.*_maybe_guard_rel.*')
# Suppress torch loggers
logging.getLogger('torch._dynamo').setLevel(logging.CRITICAL)
logging.getLogger('torch._inductor').setLevel(logging.CRITICAL)
logging.getLogger('torch.fx.experimental.symbolic_shapes').setLevel(logging.CRITICAL)
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.dynamic_shapes = True

def set_model(model_config:dict,train_config: dict, data_sample: HeteroData, device: torch.device,model_name_suffix:str=""):
    # Setting the loss function
    loss_config = model_config['loss']
    loss_fcn = LossFunction(loss_config=loss_config)


    # instantiate the model
    model_kwargs = dict(model_config['model'])
    homogeneous_model = get_model(model_type=model_kwargs['type'], **model_kwargs)
    model_in_channels = homogeneous_model.node_in_channels
    model = to_hetero(homogeneous_model,data_sample.metadata(),aggr=model_config['model']['to_hetero_aggr']).to(device)
    
    # Initialize lazy modules before creating optimizer
    if model_in_channels == -1:
        with torch.no_grad():  # Initialize lazy modules.
            data_sample = data_sample.to(device)
            # Avoid mutating `data_sample.x_dict` in-place: the same object may
            # be reused (e.g., lazy init + later dataset access).
            x_dict = dict(data_sample.x_dict)
            if 'threshold' in model_name_suffix:
                # Keep the augmentation tensor on the same device as existing features.
                x_dict['node'] = torch.concat([
                    x_dict['node'],
                    torch.zeros(x_dict['node'].shape[0], 1, device=x_dict['node'].device),
                ], dim=1)
            edge_index_dict = data_sample.edge_index_dict
            edge_attr_dict = data_sample.edge_attr_dict
            batch_dict = data_sample.batch_dict
            out = model(x_dict, edge_index_dict,edge_attr_dict,batch_dict)

    # Compile the model
    compile =train_config['device']['compile']
    compile_dynamic = train_config['device']['compile_dynamic']
    if compile:
        model = torch.compile(model, dynamic=compile_dynamic)
    model = model.to(device)
    
    # Resume the training if there is a loaded model
    resume_epoch = train_config['train']['resume_epoch']
    model_load_folder = train_config['train']['load_folder']
    if resume_epoch > 0 and model_load_folder is not None:
        model_load_file = f"epoch_{resume_epoch}.pth"  if model_name_suffix == "" else f"epoch_{resume_epoch}_{model_name_suffix}.pth"       
        model_load_path = os.path.join(model_load_folder, model_load_file)        
        model.load_state_dict(torch.load(model_load_path))
    
    # Initialize the optimizer
    optimizer_type = model_config['optimizer']['type']
    optimizer = get_optimizer(optimizer_type=optimizer_type,model_weights=model.parameters(),**model_config['optimizer'])

    # Set the model save folder
    model_type_str = model_config['model']['type']
    model_save_folder = Path(train_config['train']['save_folder']) / f"{model_type_str}{"_compile" if compile else ""}"
    os.makedirs(model_save_folder, exist_ok=True)

    return model, optimizer, loss_fcn,model_save_folder, model_type_str,resume_epoch

def run_train(train_config: dict,num_workers: int = None, use_cuda: bool = True, online: bool = False):    
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
    num_prune = train_config['train']['num_prune']

    use_threshold_gnn = train_config['threshold']['use']
    dummy_batch_size = 4
    dummy_batch = get_dummy_sample(graph_dataset[0:dummy_batch_size],batch_size=dummy_batch_size,num_workers=num_workers,device=device)
    model, optimizer, loss_fcn,model_save_folder, model_type_str,resume_epoch = set_model(train_config['evaluator'],train_config,dummy_batch,device)
    threshold_model,threshold_optimizer, threshold_loss_fcn = None,None,None
    if use_threshold_gnn:
        threshold_model, threshold_optimizer, threshold_loss_fcn, _,_,_ = set_model(train_config['threshold'],train_config,dummy_batch,device,model_name_suffix="threshold")
    

    # Initialize the training
    num_epochs = max(0,resume_epoch)+train_config['train']['num_epochs']
    save_epoch = train_config['train']['save_epoch']
    verbose_epoch = train_config.get('train',{}).get('verbose_epoch',10)
    with wandb.init(mode="offline" if not online else "online",dir = model_save_folder,project="path-planning", name=f"{model_type_str}_{"compile" if args.compile else ""}", config=train_config) as run:

        model_save_version_folder = os.path.join(wandb.run.dir, "model")
        os.makedirs(model_save_version_folder, exist_ok=True)
        
        run.define_metric("epoch/train_loss", step_metric="epoch")
        run.define_metric("epoch/test_loss", step_metric="epoch")
        run.define_metric("epoch/train_time", step_metric="epoch")
        run.define_metric("epoch/test_time", step_metric="epoch")
        if use_threshold_gnn:
            run.define_metric("epoch/threshold_train_loss", step_metric="epoch")
            run.define_metric("epoch/threshold_test_loss", step_metric="epoch")
            run.define_metric("epoch/threshold_train_time", step_metric="epoch")
            run.define_metric("epoch/threshold_test_time", step_metric="epoch")
        run.define_metric("runtime", step_metric="epoch")
        start_time = time()
        for epoch in range(resume_epoch + 1, num_epochs + 1):
            st = time()
            train_loss,train_time = train(train_loader,model,optimizer,threshold_model=threshold_model,loss_function=loss_fcn,device=device,run=run,num_prune=num_prune)
            et = time()
            test_loss,test_time = test(test_loader,model,threshold_model=threshold_model,loss_function=loss_fcn,device=device,run=run,num_prune=num_prune)
            tt = time()
            epoch_train_time = et-st
            epoch_test_time = tt-et

            if threshold_model is not None:
                st = time()
                threshold_train_loss,threshold_train_time = train_threshold(
                    train_loader,
                    threshold_model,
                    threshold_optimizer,
                    loss_function=threshold_loss_fcn,
                    model=model,
                    device=device,
                    run=run,
                    num_prune=num_prune,
                )
                et = time()
                threshold_test_loss,threshold_test_time = test_threshold(
                    test_loader,
                    threshold_model,
                    loss_function=threshold_loss_fcn,
                    model=model,
                    device=device,
                    run=run,
                    num_prune=num_prune,
                )
                tt = time()
                epoch_threshold_train_time = et-st
                epoch_threshold_test_time = tt-et


            runtime = time() - start_time

            run_dict = {
                "epoch": epoch,
                "epoch/train_loss": np.mean(train_loss),
                "epoch/test_loss": np.mean(test_loss),
                "epoch/train_time": epoch_train_time,
                "epoch/test_time": epoch_test_time,
                "runtime": runtime,
            }
            if use_threshold_gnn:
                run_dict.update({
                    "epoch/threshold_train_loss": np.mean(threshold_train_loss),
                    "epoch/threshold_test_loss": np.mean(threshold_test_loss),
                    "epoch/threshold_train_time": epoch_threshold_train_time,
                    "epoch/threshold_test_time": epoch_threshold_test_time,
                })
            run.log(run_dict, commit=False)
            if verbose_epoch > 0 and epoch % verbose_epoch == 0:
                if use_threshold_gnn:
                    print(f"Runtime: {runtime:.2f} \t Epoch: {epoch} \t Train loss: {np.mean(train_loss):.4f} \t Test loss: {np.mean(test_loss):.4f} \t Train time: {epoch_train_time:.4f} \t Test time: {epoch_test_time:.4f} \t Threshold train loss: {np.mean(threshold_train_loss):.4f} \t Threshold test loss: {np.mean(threshold_test_loss):.4f} \t Threshold train time: {epoch_threshold_train_time:.4f} \t Threshold test time: {epoch_threshold_test_time:.4f}")
                else:
                    print(f"Runtime: {runtime:.2f} \t Epoch: {epoch} \t Train loss: {np.mean(train_loss):.4f} \t Test loss: {np.mean(test_loss):.4f} \t Train time: {epoch_train_time:.4f} \t Test time: {epoch_test_time:.4f}")
            if epoch % save_epoch == 0:
                model_save_file = f"epoch_{epoch}.pth"
                model_save_path = os.path.join(model_save_version_folder, model_save_file)
                torch.save(model.state_dict(), model_save_path)
                run.log_model(model_save_path, name=model_save_file)

                if use_threshold_gnn:
                    threshold_model_save_file = f"epoch_{epoch}_threshold.pth"
                    threshold_model_save_path = os.path.join(model_save_version_folder, threshold_model_save_file)
                    torch.save(threshold_model.state_dict(), threshold_model_save_path)
                    run.log_model(threshold_model_save_path, name=threshold_model_save_file)

        model_save_file = f"epoch_{num_epochs}.pth"
        model_save_path = os.path.join(model_save_version_folder, model_save_file)
        torch.save(model.state_dict(), model_save_path)
        run.log_model(model_save_path, name=model_save_file)
        if use_threshold_gnn:
            threshold_model_save_file = f"epoch_{num_epochs}_threshold.pth"
            threshold_model_save_path = os.path.join(model_save_version_folder, threshold_model_save_file)
            torch.save(threshold_model.state_dict(), threshold_model_save_path)
            run.log_model(threshold_model_save_path, name=threshold_model_save_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--folder_paths",type=str, nargs='+', 
        default=[
        '/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map64.0x64.0_resolution1.0/agents1_obst0.025/radius0.0',
        '/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map64.0x64.0_resolution1.0/agents2_obst0.025/radius0.0',
        '/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/map64.0x64.0_resolution1.0/agents4_obst0.025/radius0.0',
        ])
    parser.add_argument("-l","--load_file",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/data.pt', help="load file")
    parser.add_argument("-s","--save_file",type=str, default='/home/bho36/Dropbox/Team_Path_Planning/brandon_graph_data/train/data.pt', help="save file")
    parser.add_argument("-ds","--use_discrete_space",dest="use_discrete_space",action="store_true", help="use discrete space")
    parser.add_argument("-rmt","--road_map_type",type=str, nargs='+', default=['prm','cdt'], help="road map type")
    parser.add_argument("-ts","--target_space",type=str, help="target space")    
    parser.add_argument("-w","--num_workers",type=int, default=None, help="number of parallel workers for cases (default: auto-detect CPU cores)")
    parser.add_argument("-cuda","--cuda",dest="cuda",action="store_true", help="use cuda")
    parser.add_argument("--no-cuda",dest="cuda",action="store_false", help="disable cuda")
    parser.add_argument("-train","--train_config",type=str, default='config/train.yaml', help="train config file")
    parser.add_argument("-compile","--compile",dest="compile",action="store_true", help="compile model")
    parser.add_argument("--no-compile",dest="compile",action="store_false", help="disable compile")
    parser.add_argument("-dynamic","--compile_dynamic",dest="compile_dynamic",action="store_true", help="compile dynamic mode")
    parser.add_argument("--no-compile-dynamic",dest="compile_dynamic",action="store_false", help="disable compile dynamic mode")
    parser.add_argument("-seed","--seed",type=int, default=42, help="seed")
    parser.add_argument("-online","--online",action="store_true", help="use wandb")
    parser.set_defaults(cuda=True, compile=True, compile_dynamic=True)
    args = parser.parse_args()
    
    with open(args.train_config, 'r') as f:
        train_config = yaml.load(f,Loader=yaml.FullLoader)
    train_config = set_train_config(train_config,args)

    wandb.login()
    run_train(train_config,num_workers=args.num_workers, use_cuda=args.cuda, online=args.online)