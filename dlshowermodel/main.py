import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dlshowermodel.data.ClusterGraphDataset import ClusterGraphDataset
from dlshowermodel.models.TransformerGATv2 import TransformerGATv2Model
from dlshowermodel.train import train_model
from dlshowermodel.loss.loss_functions import get_loss_function
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
        train_cachefule = dataset_params['load_training_data_from_cachefile']
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

    if 'train_num_workers' in train_params:
        train_num_workers = train_params['train_num_workers']
    else:
        train_num_workers = 1
    
    # Create graph dataset
    print("Creating Graph Dataset...")
    train_graphdata_device = device
    if train_num_workers>0:
        # cannot use cuda for clsutergraphdataset
        train_graphdata_device = torch.device('cpu')
    print('train_graphdata_device: ',train_graphdata_device)

    train_dataset = ClusterGraphDataset(lar_dataset_train, 
        k_neighbors=dataset_params['k_neighbors'], 
        device=train_graphdata_device)
    val_dataset = ClusterGraphDataset(lar_dataset_valid, 
        k_neighbors=dataset_params['k_neighbors'], 
        device=train_graphdata_device)
    test_dataset = ClusterGraphDataset(lar_dataset_valid, 
        k_neighbors=dataset_params['k_neighbors'], 
        device=train_graphdata_device)

    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")


    
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
        collate_fn=ClusterGraphDataset.collate_fn
    )
    valid_iter = iter(val_loader)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_params['batch_size'],
        shuffle=False, 
        collate_fn=ClusterGraphDataset.collate_fn
    )
    
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
    
    # Get Loss
    loss_name = train_params['Loss']
    loss_fn = get_loss_function( loss_name )
    # if loss_name == "WeightedBCELoss":
    #     loss_fn = WeightedBCELoss()
    # def weighted_bce_loss(pred, target):
    #     # Calculate standard BCE loss
    #     bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    #     #print('pred=',pred.shape,"   target=",target.shape)
        
    #     # Apply lower weight to examples with target = 0.5 (unknown)
    #     weights = torch.ones_like(target,requires_grad=False)
    #     with torch.no_grad():
    #         pos_mask = (target == 1.0)
    #         neg_mask = (target == 0.0)
    #         npos = pos_mask.sum().to(torch.float)
    #         nneg = neg_mask.sum().to(torch.float)
    #         if npos>0:
    #             weights[pos_mask] = 1.0/npos
    #         if nneg>0:
    #             weights[neg_mask] = 1.0/nneg

        
    #     # Apply weights and take mean
    #     weighted_loss = 0.5*(bce_loss * weights).sum()
        
    #     return weighted_loss

    
    print("Starting wandb logger")
    log_config = {"train_params":train_params,
                "dataset_params":dataset_params,
                "model_config":model_config}
    wandb_writer = wandb.init(
            project='dlshowerreco-gatv2-settransformer',
            config=log_config)

    # Train model
    print("Training model...")
    model = train_model(
        train_params,
        model, 
        train_loader, 
        val_loader, 
        valid_iter,
        test_loader,
        loss_fn,  # Use our custom loss
        optimizer, 
        device,
        train_params['batch_size'],
        lr = train_params['lr'],
        burn_in_epochs=train_params['burn_in_epochs'],
        burn_in_lr=train_params['burn_in_lr'],
        num_epochs=train_params['epochs'], 
        patience=train_params['patience'],
        logger=wandb_writer
    )
    
    wandb_writer.finish()

    return model

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    file_paths = ["dataprep/test_bnbnue_corsika_full_notruth.h5"]
    #file_paths = ["dataprep/test_bnbnue_corsika_e1_notruth.h5"]
    
    dataset_params = dict(
        k_neighbors=16,
        apply_max_filter=False, 
        max_num_spacepoints=10000,
        train_file_paths=file_paths,
        valid_file_paths=file_paths,
        train_num_workers=4
        #load_training_data_from_cachefile="dataprep/dlshowermodel_training_cache_file.txt",
        #load_validation_data_from_cachefile="dataprep/dlshowermodel_validation_cache_file.txt"
    )

    train_params = dict(
        batch_size=16,
        lr=1.0e-3, 
        weight_decay=5e-4, 
        epochs=5000, 
        patience=1000000,
        burn_in_epochs=1,
        burn_in_lr=0.5e-4,
        niters_per_eval=10,
        log_to_wandb=True,
        starting_iter_num=0,
        epochs_per_checkpoint=1,
        eval_nvalid_batches=1,
        use_early_stopping=False,
        Loss={"name":"WeightedFocalLoss",
              "params":{
                  "gamma":2.0
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
    """
    resgatv2_cfg = model_config['TransformerGATv2']['ResGATv2']


    model = run_experiment(
        dataset_params,
        train_params,
        model_config
    )
