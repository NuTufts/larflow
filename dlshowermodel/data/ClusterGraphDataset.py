import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import KNNGraph


# Create a PyTorch Geometric compatible dataset wrapper for your LArMatchHitHDF5Dataset
class ClusterGraphDataset(Dataset):
    def __init__(self, lar_dataset, k_neighbors=5, device=torch.device('cpu')):
        """
        Wrapper for the LArMatchHitHDF5Dataset to create PyTorch Geometric Data objects.
        
        Args:
            lar_dataset: The LArMatchHitHDF5Dataset instance
            k_neighbors: Number of neighbors for KNN edge construction
            device: Device to put the data on
        """
        self.lar_dataset = lar_dataset
        self.k_neighbors = k_neighbors
        self.device = device
        
        # Create a KNN transform to be applied to each graph
        self.knn_transform = KNNGraph(k=k_neighbors, loop=False, force_undirected=True)
    
    def __len__(self):
        return len(self.lar_dataset)
    
    def __getitem__(self, idx):
        # Get raw data from the LArMatchHitHDF5Dataset
        entry_data = self.lar_dataset[idx]
        
        # Extract relevant cluster features
        # cluster_sampled_feat: (NC, 16, 48) - 16 feature vectors per cluster, each 48-dim
        cluster_features = torch.tensor(entry_data['cluster_sampled_feat'], dtype=torch.float)
        
        # PCA features: (NC, 21) - PCA-based features
        pca_features = torch.tensor(entry_data['cluster_feat_pca'], dtype=torch.float)

        # pixelsum features: (NC,3)
        pixsum_features = torch.tensor(entry_data['cluster_feat_planepixelsum'], dtype=torch.float)
        
        # Create a PyTorch Geometric Data object
        data = Data()
        
        # Store raw features
        data.cluster_features = cluster_features  # Shape: [num_nodes, 16, 48]
        data.pca_features = pca_features  # Shape: [num_nodes, 21]
        data.pixsum_features = pixsum_features # Shape: [num_nodes, 3]
        
        # Store additional metadata
        data.idx = idx
        data.num_nodes = cluster_features.shape[0]
        
        # Add centroid positions for visualization and KNN if available
        if 'cluster_feat_centroid' in entry_data:
            data.pos = torch.tensor(entry_data['cluster_feat_centroid'], dtype=torch.float)
        else:
            # Use PCA features as positions for KNN
            data.pos = pca_features
        
        # Skip KNN if too few nodes
        if data.num_nodes <= 1:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_label = torch.zeros(0, dtype=torch.float)
            return data
        
        # Always create edges using KNN
        data = self.knn_transform(data)
        
        # If truth edges are available, use them to label the KNN edges
        #print(entry_data['idx'])
        if 'showercluster_edge_list' in entry_data \
            and entry_data['showercluster_edge_list'].size>0 \
            and entry_data['showercluster_edge_list'].shape[0] > 0:
            truth_edge_list = entry_data['showercluster_edge_list']
            
            # Create a set of truth edges for efficient lookup
            truth_edges = set()
            for i in range(truth_edge_list.shape[0]):
                #print( type(entry_data['showercluster_edge_list']) )
                #print( entry_data['showercluster_edge_list'] )
                src, dst = truth_edge_list[i, 0], truth_edge_list[i, 1]
                truth_edges.add((src, dst))
                truth_edges.add((dst, src))  # Add both directions for undirected graph
            
            # Label KNN edges: 1 if in truth edges, 0 otherwise
            edge_labels = []
            for i in range(data.edge_index.size(1)):
                src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
                if (src, dst) in truth_edges:
                    edge_labels.append(1.0)
                else:
                    edge_labels.append(0.0)
            
            data.edge_label = torch.tensor(edge_labels, dtype=torch.float)
        else:
            # If no truth edges available, all KNN edges are labeled as unknown (0.5)
            # This allows the model to learn from these examples with less certainty
            data.edge_label = torch.ones(data.edge_index.size(1), dtype=torch.float) * 0.0

        data.cluster_labels = torch.from_numpy(np.squeeze(entry_data['cluster_labels'].astype(np.int64))).to(self.device)
        data.cluster_points = torch.from_numpy(np.squeeze(entry_data['shower_points'])).to(self.device)
        
        return data
    
    def _create_knn_edges(self, data):
        """Create edges using k-nearest neighbors with PyTorch Geometric's KNNGraph."""
        # Use centroids for KNN if available, otherwise use pca features
        if hasattr(data, 'pos'):
            # Already has position data, KNNGraph will use it automatically
            pass
        else:
            # Use PCA features as positions for KNN
            data.pos = data.pca_features
        
        num_nodes = data.num_nodes
        
        # Skip if too few nodes
        if num_nodes <= 1:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            return data
        
        # Use PyTorch Geometric's KNNGraph transform
        k = min(self.k_neighbors, num_nodes - 1)  # Ensure k is valid
        transform = KNNGraph(k=k, loop=False, force_undirected=True)
        data = transform(data)
        
        # Note: We don't set edge_label here anymore, since that will be handled
        # based on the truth edge list in the __getitem__ method
        
        return data

    # Custom collate function for batching PyG Data objects
    def collate_fn(batch):
        """Custom collate function for PyG Data objects."""
        return Batch.from_data_list(batch)

