import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dlshowermodel.data.ClusterGraphDataset import ClusterGraphDataset
from dlshowermodel.models.TransformerGATv2 import TransformerGATv2Model
from dlshowermodel.train import train_model

# Main function to run the experiment
def run_experiment(file_paths, batch_size=16, k_neighbors=5, 
                  hidden_dim=64, gnn_hidden_dim=64, 
                  num_heads=4, num_layers=2, dropout=0.1,
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
        hidden_dim=hidden_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        x_range=(-50, 300),            # Range of x coordinates in cm
        y_range=(-120, 120),           # Range of y coordinates in cm
        z_range=(0, 1040),             # Range of z coordinates in cm
        pos_min_freq=0.0001,           # Minimum frequency for position embedding
        pos_max_freq=1.0,              # Maximum frequency for position embedding
        pos_scale=10.0                 # Scale factor to make embedding sensitive to cm-scale changes
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use weighted BCEWithLogitsLoss to handle unknown labels (0.5)
    # We'll use a custom loss function that gives less weight to edges with unknown labels
    def weighted_bce_loss(pred, target):
        # Calculate standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply lower weight to examples with target = 0.5 (unknown)
        weights = torch.ones_like(target)
        unknown_mask = (target == 0.5)
        weights[unknown_mask] = unknown_edge_weight
        
        # Apply weights and take mean
        weighted_loss = (bce_loss * weights).mean()
        
        return weighted_loss
    
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
        patience=patience
    )
    
    return model, train_metrics, val_metrics, test_metrics

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    file_paths = ["test_traindata_fullfile.h5"]
    
    model, train_metrics, val_metrics, test_metrics = run_experiment(
        file_paths=file_paths,
        batch_size=1,
        k_neighbors=5,
        hidden_dim=64,
        gnn_hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        lr=0.001,
        weight_decay=5e-4,
        epochs=100,
        patience=100
    )