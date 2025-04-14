import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dlshowermodel.data.ClusterGraphDataset import ClusterGraphDataset
from dlshowermodel.models.TransformerGATv2 import TransformerGATv2Model
from dlshowermodel.train import train_model
from dlshowermodel.loss.loss_functions import get_loss_function
from dlshowermodel.utils import get_lr_scheduler
import wandb

# Main function to run the experiment
def run_experiment( dataset_params, train_params, model_config ):
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    from dlshowermodel.data.larmatchhit_hdf5_reader import LArMatchHitHDF5Dataset
    
    print("Loading LArMatchHitHDF5Dataset...")
    
    train_cachefile = None
    valid_cachefile = None    
    if 'load_training_data_from_cachefile' in dataset_params:
        train_cachefile = dataset_params['load_training_data_from_cachefile']
    if 'load_validation_data_from_cachefile' in dataset_params:
        valid_cachefile = dataset_params['load_validation_data_from_cachefile']

    train_file_paths = None
    valid_file_paths = None
    if 'train_file_paths' in dataset_params and train_cachefile is None:
        train_file_paths = dataset_params['train_file_paths']
    if 'valid_file_paths' in dataset_params and valid_cachefile is None:
        valid_file_paths = dataset_params['valid_file_paths']
        
    
    lar_dataset_train = LArMatchHitHDF5Dataset(
        file_paths=train_file_paths,
        file_has_training_labels=True,
        load_from_cachefile=train_cachefile,
        apply_max_filter=dataset_params['apply_max_filter'],
        max_num_spacepoints=dataset_params['max_num_spacepoints']
    )
    lar_dataset_valid = LArMatchHitHDF5Dataset(
        file_paths=valid_file_paths,
        file_has_training_labels=True,
        load_from_cachefile=valid_cachefile,
        apply_max_filter=dataset_params['apply_max_filter'],
        max_num_spacepoints=dataset_params['max_num_spacepoints']
    )

    
    print(f"Training Dataset size: {len(lar_dataset_train)}")
    print(f"Validation Dataset size: {len(lar_dataset_valid)}")    

    if 'train_num_workers' in dataset_params:
        train_num_workers = dataset_params['train_num_workers']
    else:
        train_num_workers = 0 # does not use spawned process
    if 'valid_num_workers' in dataset_params:
        valid_num_workers = dataset_params['valid_num_workers']
    else:
        valid_num_workers = 0 # does not use spawned process
    
    # Create graph dataset
    print("Creating Graph Dataset...")
    graphdata_device = device
    if train_num_workers>0:
        # cannot use cuda for clsutergraphdataset
        graphdata_device = torch.device('cpu')
    print('graphdata_device: ',graphdata_device)

    train_dataset = ClusterGraphDataset(lar_dataset_train, 
        k_neighbors=dataset_params['k_neighbors'], 
        device=graphdata_device)
    val_dataset = ClusterGraphDataset(lar_dataset_valid, 
        k_neighbors=dataset_params['k_neighbors'], 
        device=graphdata_device)
    test_dataset = ClusterGraphDataset(lar_dataset_valid, 
        k_neighbors=dataset_params['k_neighbors'], 
        device=graphdata_device)

    # use info about dataset to set niters per training dataset epoch
    nevents_train = len(train_dataset)
    niters_per_epoch = max( int(nevents_train/train_params['batch_size']), 1 )

    train_params['nevents_train_dataset']  = nevents_train
    train_params['niters_per_train_epoch'] = niters_per_epoch
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_params['batch_size'], 
        shuffle=True, 
        collate_fn=ClusterGraphDataset.collate_fn,
        num_workers=train_num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_params['batch_size'], 
        shuffle=True, 
        collate_fn=ClusterGraphDataset.collate_fn,
        num_workers=valid_num_workers
    )
    valid_iter = iter(val_loader)
    
    # Initialize model
    print("Initializing model...")
    model = TransformerGATv2Model.load_from_config(model_config).to(device)

    print(model)
    ntrainable = 0
    for par in model.parameters():
        ntrainable += par.numel()
    print("Number of parameters: ",ntrainable)
    #sys.exit(0)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), 
        lr=train_params['burn_in_lr'], 
        weight_decay=train_params['weight_decay'])

    if train_params['reload_optimizer_state']:
        print("Re-loading Optimizer State Checkpoint")
        saved_dict = torch.load( model_config['TransformerGATv2']['checkpoint_file'] )
        optim_state_dict = saved_dict['optimizer']
        optimizer.load_state_dict( optim_state_dict )

    # Get LR scheduler if defined
    if 'lr_scheduler' in train_params:
        lr_scheduler_name = train_params['lr_scheduler']['name']
        lr_scheduler_cfg  = train_params['lr_scheduler']['params']
        lr_scheduler_cfg['iters_per_epoch'] = niters_per_epoch # this means I should use floating point epoch
        lr_scheduler = get_lr_scheduler(lr_scheduler_name,lr_scheduler_cfg)
    else:
        lr_scheduler = None
    
    # Get Loss
    loss_name = train_params['Loss']
    loss_fn = get_loss_function( loss_name )
    
    print("Starting wandb logger")
    log_config = {"train_params":train_params,
                "dataset_params":dataset_params,
                "model_config":model_config}
    if train_params['log_to_wandb']:
        wandb_writer = wandb.init(
                project='dlshowerreco-gatv2-settransformer',
                config=log_config)
    else:
        wandb_writer = None

    # Train model
    print("Training model...")
    model = train_model(
        train_params,
        model, 
        train_loader, 
        val_loader, 
        valid_iter,
        loss_fn,  # Use our custom loss
        optimizer, 
        device,
        train_params['batch_size'],
        lr = train_params['lr'],
        burn_in_epochs=train_params['burn_in_epochs'],
        burn_in_lr=train_params['burn_in_lr'],
        num_epochs=train_params['epochs'], 
        patience=train_params['patience'],
        logger=wandb_writer,
        lr_scheduler=lr_scheduler
    )
    
    wandb_writer.finish()

    return model

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    file_paths = ["dataprep/test_bnbnue_corsika_full_notruth.h5"]
    #file_paths = ["dataprep/test_bnbnue_corsika_e1_notruth.h5"]

    dlshower_dir='/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/dlshowermodel'    
    
    dataset_params = dict(
        k_neighbors=16,
        apply_max_filter=False, 
        max_num_spacepoints=10000,
        train_file_paths=None,
        valid_file_paths=None,
        train_num_workers=24,
        valid_num_workers=6,
        load_training_data_from_cachefile=dlshower_dir+"/dataprep/dlshowermodel_training_cache_file.txt",
        load_validation_data_from_cachefile=dlshower_dir+"/dataprep/dlshowermodel_validation_cache_file.txt"
    )

    train_params = dict(
        batch_size=32,
        lr=1.0e-3, 
        weight_decay=5e-4, 
        epochs=100, 
        patience=1000000,
        burn_in_epochs=1,
        burn_in_lr=0.2e-4,
        niters_per_eval=10,
        log_to_wandb=True,
        starting_iter_num=0,
        epochs_per_checkpoint=1,
        eval_nvalid_batches=1,
        use_early_stopping=False,
        reload_optimizer_state=False,
        Loss={"name":"WeightedFocalLoss",
              "params":{
                  "gamma":2.0
              }
        },
        lr_scheduler={"name":"CosineAnnealingWithWarmup",
                "params":{
                    "epoch_period":20,
                    "warmup_epochs":0.1,
                    "lr_warmup":0.5e-4,
                    "lr_min":1.0e-4,
                    "lr_max":3.0e-3,
                    "epoch_offset":0.0,
                    "iter_offset":0.0,
                    "iters_per_epoch":5205,
                    "linear_ramp_epochs":0.4
                }
        }
    )

    model_config = TransformerGATv2Model.dump_example_config()

    """
    TransformerGATv2:
        ResGATv2:
            dropout: 0.5
            edgelayer_hidden_dim: 128
            gnn_hidden_dim: 48
            node_pos_embedding_dim: 48
            norm_type: graph
            num_gat_heads: 4
            pca_feature_dim: 21
        SetTransformer:
            cluster_token_dim: 64
            num_cluster_hidden_heads: 4
            num_cluster_out_tokens: 4
            spacepoint_feature_dim: 48
        load_from_checkpoint: False
        checkpoint_file: "your_checkpoint_file.pt"
    """
    resgatv2_cfg = model_config['TransformerGATv2']['ResGATv2']

    # modifying config for debugging runs
    checkpoint_dir=dlshower_dir+'/checkpoints'
    model_config['TransformerGATv2']['load_from_checkpoint'] = True
    #model_config['TransformerGATv2']['checkpoint_file'] = 'ubshower_gnn_bestmodel_f1_classic_salad.pt'
    #model_config['TransformerGATv2']['checkpoint_file'] = checkpoint_dir+'/classic_salad_61/ubshower_gnn_checkpoint_epoch20_iter114510.pt'
    #model_config['TransformerGATv2']['checkpoint_file'] = checkpoint_dir+'/olive-fog-85/ubshower_gnn_checkpoint_epoch16_iter202996.pt'    
    model_config['TransformerGATv2']['checkpoint_file'] = checkpoint_dir+"/flowing-blaze-86/ubshower_gnn_checkpoint_epoch33_iter291499.pt"
    train_params['epochs_per_checkpoint'] = 1
    train_params['starting_iter_num'] = 291499
    train_params['log_to_wandb'] = True
    train_params['lr_scheduler']['params']['warmup_epochs'] = 5
    train_params['lr_scheduler']['params']['lr_warmup'] = 5.0e-4
    train_params['lr_scheduler']['params']['epoch_period'] = 100
    train_params['lr_scheduler']['params']['lr_max'] = 1.0e-3
    train_params['lr_scheduler']['params']['lr_min'] = 0.5e-3
    train_params['lr_scheduler']['params']['linear_ramp_epochs'] = 10
    train_params['lr_scheduler']['params']['iters_per_epoch'] = 5205
    train_params['lr_scheduler']['params']['iter_offset']  = 291499
    train_params['lr_scheduler']['params']['epoch_offset'] = 0    



    model = run_experiment(
        dataset_params,
        train_params,
        model_config
    )
