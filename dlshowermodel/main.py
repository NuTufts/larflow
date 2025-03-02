import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dlshowermodel.data.ClusterGraphDataset import ClusterGraphDataset
from dlshowermodel.models.TransformerGATv2 import TransformerGATv2Model
from dlshowermodel.train import train_model
import wandb

# Main function to run the experiment
def run_experiment(file_paths, batch_size=16, 
                  k_neighbors=5, 
                  cluster_hidden_dim=64, 
                  gnn_hidden_dim=64, 
                  num_heads=4,  
                  num_gcnn_layers=2,
                  num_out_tokens=8,
                  dropout=0.1,
                  lr=0.001, weight_decay=5e-4, epochs=100, patience=10,
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  load_from_cachefile=None, apply_max_filter=False, max_num_spacepoints=10000,
                  unknown_edge_weight=0.1):  # Weight for edges with unknown labels
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    from dlshowermodel.data.larmatchhit_hdf5_reader import LArMatchHitHDF5Dataset
    
    print("Loading LArMatchHitHDF5Dataset...")
    lar_dataset = LArMatchHitHDF5Dataset(
        file_paths=file_paths,
        file_has_training_labels=True,
        load_from_cachefile=load_from_cachefile,
        apply_max_filter=apply_max_filter,
        max_num_spacepoints=max_num_spacepoints
    )
    
    print(f"Dataset size: {len(lar_dataset)}")
    
    # Create graph dataset
    print("Creating Graph Dataset...")
    train_dataset = ClusterGraphDataset(lar_dataset, k_neighbors=k_neighbors, device=device)
    val_dataset = ClusterGraphDataset(lar_dataset, k_neighbors=k_neighbors, device=device)
    test_dataset = ClusterGraphDataset(lar_dataset, k_neighbors=k_neighbors, device=device)

    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=ClusterGraphDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=ClusterGraphDataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=ClusterGraphDataset.collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    model = TransformerGATv2Model(
        cluster_feature_dim=48,        # Fixed dimension from your dataset
        pca_feature_dim=21,            # Fixed dimension from your dataset
        pos_embedding_dim=48,          # Dimension of position embedding
        cluster_hidden_dim=cluster_hidden_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        num_out_tokens=num_out_tokens,
        num_heads=num_heads,
        num_gat_heads=num_heads,
        num_gcnn_layers=num_gcnn_layers,
        dropout=dropout,
        pos_min_freq=0.0001,          # Minimum frequency for position embedding
        pos_max_freq=1.0,              # Maximum frequency for position embedding
        pos_scale=100.0                 # Scale factor to make embedding sensitive to cm-scale changes
    ).to(device)

    print(model)
    ntrainable = 0
    for par in model.parameters():
        ntrainable += par.numel()
    print("Number of parameters: ",ntrainable)
    #sys.exit(0)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use weighted BCEWithLogitsLoss to handle unknown labels (0.5)
    # We'll use a custom loss function that gives less weight to edges with unknown labels
    def weighted_bce_loss(pred, target):
        # Calculate standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        #print('pred=',pred.shape,"   target=",target.shape)
        
        # Apply lower weight to examples with target = 0.5 (unknown)
        weights = torch.ones_like(target,requires_grad=False)
        with torch.no_grad():
            pos_mask = (target == 1.0)
            neg_mask = (target == 0.0)
            npos = pos_mask.sum().to(torch.float)
            nneg = neg_mask.sum().to(torch.float)
            if npos>0:
                weights[pos_mask] = 1.0/npos
            if nneg>0:
                weights[neg_mask] = 1.0/nneg

        
        # Apply weights and take mean
        weighted_loss = (bce_loss * weights).sum()
        
        return weighted_loss
    
    print("Starting wandb logger")
    wandb_writer = wandb.init(
            project='dlshowerreco-gatv2-settransformer')

    # Train model
    print("Training model...")
    model, train_metrics, val_metrics, test_metrics = train_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader,
        weighted_bce_loss,  # Use our custom loss
        optimizer, 
        device, 
        num_epochs=epochs, 
        patience=patience,
        logger=wandb_writer
    )
    
    wandb_writer.finish()

    return model, train_metrics, val_metrics, test_metrics

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    file_paths = ["dataprep/test_bnbnue_corsika_full_notruth.h5"]
    #file_paths = ["dataprep/test_bnbnue_corsika_e1_notruth.h5"]
    
    model, train_metrics, val_metrics, test_metrics = run_experiment(
        file_paths=file_paths,
        batch_size=4,
        k_neighbors=16,
        dropout=0.0,
        lr=1.0e-4,
        weight_decay=5e-4,
        num_heads=8,
        num_out_tokens=8,
        num_gcnn_layers=2,
        epochs=1000,
        patience=100000
    )