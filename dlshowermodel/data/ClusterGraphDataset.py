import torch
import numpy as np
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import KNNGraph
from dlshowermodel.utils.sinusoidal_embeddings import SinusoidalPositionEmbedding

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

        # Sinusoidal Embeddings for the sampled feature positions 
        self.pos_embed_dims = 48
        self.sampled_pos_embedding_fn = SinusoidalPositionEmbedding(
            embedding_dim=self.pos_embed_dims,
            x_range=[-500.0,500.0],
            y_range=[-500.0,500.0],
            z_range=[-500.0,500.0],
            min_freq=1.0,
            max_freq=1000.0,
            scale_factor=1.0
        )
    
    def __len__(self):
        return len(self.lar_dataset)
    
    def __getitem__(self, idx):
        # Get raw data from the LArMatchHitHDF5Dataset
        entry_data = self.lar_dataset[idx]
        
        # Extract relevant cluster features
        # cluster_sampled_feat: (NC, 16, 48) - 16 feature vectors per cluster, each 48-dim
        cluster_features = torch.tensor(entry_data['cluster_sampled_feat'], dtype=torch.float)
        cluster_sampled_pos = torch.tensor(entry_data['cluster_sampled_pos'],dtype=torch.float) #(NC,16,3)
        cluster_centroids = torch.tensor(np.expand_dims(entry_data['cluster_feat_centroid'],1),dtype=torch.float) #(NC,3) --> #(NC,1,3)
        #print("cluster_centroids")
        #print(cluster_centroids[:3,:])
        #print("sampled_pos")
        #print(cluster_sampled_pos[:3,:3,:])

        sampled_pos_centered = cluster_sampled_pos-cluster_centroids # should be (NC,16,3)
        num_clusters = sampled_pos_centered.shape[0]
        #print("centered_sampled_pos")
        #print(sampled_pos_centered[:3,:3,:])
        sampled_pos_embed = self.sampled_pos_embedding_fn( torch.flatten(sampled_pos_centered, 0, -2) ).view(num_clusters,16,self.pos_embed_dims)
        
        # PCA features: (NC, 21) - PCA-based features
        # pca feats: [0-8] dir of first 3 pca components, [9-11] explained variance, [12-17] bounds along pca axes, [18-20] lens across pca axes
        pca_features = torch.tensor(entry_data['cluster_feat_pca'], dtype=torch.float)
        #print(pca_features[:,18:])
        #print(pca_features[:,12:18])
        # normalize feats with length dimensions
        pca_features[:,18:] /= 100.0 # divide length by 100 cm
        pca_features[:,12:18] /= 100.0 # divide length by 100 cm

        # pixelsum features: (NC,3)
        pixsum_features = torch.tensor(entry_data['cluster_feat_planepixelsum'], dtype=torch.float)
        pixsum_features = torch.sqrt( pixsum_features + 1.0 )/300.0
        
        # Create a PyTorch Geometric Data object
        data = Data()
        
        # Store raw features
        data.sampled_pos_embed = sampled_pos_embed # Shape: [num_nodes, 16, 48]
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
        
        # Make edge labels
        data.edge_label = torch.zeros(data.edge_index.size(1), dtype=torch.float)

        # If truth edges are available, use them to label the KNN edges
        #print(entry_data['idx'])
        if 'showercluster_edge_list' in entry_data \
            and entry_data['showercluster_edge_list'].size>0 \
            and entry_data['showercluster_edge_list'].shape[0] > 0:
            truth_edge_list = entry_data['showercluster_edge_list']
            #print(truth_edge_list.shape)
            #print(truth_edge_list)
            
            # Create a set of truth edges for efficient lookup
            truth_edges = {}
            for i in range(truth_edge_list.shape[0]):
                #print( type(entry_data['showercluster_edge_list']) )
                #print( entry_data['showercluster_edge_list'] )
                src, dst = truth_edge_list[i, 0], truth_edge_list[i, 1]
                if src!=dst:
                    truth_edges[(src,dst)] = False
                    truth_edges[(dst,src)] = False
            
            # Label KNN edges: 1 if in truth edges, 0 otherwise
            for i in range(data.edge_index.size(1)):
                src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
                if (src, dst) in truth_edges:
                    data.edge_label[i] = 1.0
                    truth_edges[(src,dst)] = True
                    truth_edges[(dst,src)] = True

            # make a list of true edges not in the kNN tree!
            missing_truth_edges = []
            for (src,dst) in truth_edges:
                if truth_edges[(src,dst)]==False:
                    missing_truth_edges.append( [src,dst] )
            #print("number of missing truth edges: ",len(missing_truth_edges)/2," out of ",len(truth_edges)/2)
            #for missing_truth_edge in missing_truth_edges:
            #    print("  ",missing_truth_edge)
            if len(missing_truth_edges)>0:
                data.missing_truth_edges = torch.tensor( missing_truth_edges, dtype=torch.long )
            else:
                data.missing_truth_edges = torch.zeros( (1,2), dtype=torch.float )
            
        data.cluster_labels = torch.from_numpy(np.squeeze(entry_data['cluster_labels'].astype(np.int64))).to(self.device)
        data.cluster_points = torch.from_numpy(np.squeeze(entry_data['shower_points'])).to(self.device)

        # print("data.edge_label: ",data.edge_label.shape)
        # print("data.edge_index: ",data.edge_index.shape)
        # print("data.cluster_labels: ",data.cluster_labels.shape)
        # print("data.cluster_points: ",data.cluster_points.shape)
        # print("data.missing_truth_edges: ",data.missing_truth_edges.shape)
        
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

