import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATv2Conv
from dlshowermodel.utils.sinusoidal_embeddings import SinusoidalPositionEmbedding
from dlshowermodel.models.ClusterTransformer import ClusterTransformer

# Define the complete model combining Transformer, Position Embedding, and GATv2
class TransformerGATv2Model(nn.Module):
    def __init__(self, cluster_feature_dim=48, pca_feature_dim=21, pos_embedding_dim=48, 
                 hidden_dim=64, gnn_hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
                 x_range=(-50, 300), y_range=(-120, 120), z_range=(0, 1040),
                 pos_min_freq=0.0001, pos_max_freq=1.0, pos_scale=10.0):
        super(TransformerGATv2Model, self).__init__()
        
        # Transformer for processing cluster features
        self.transformer = ClusterTransformer(
            input_dim=cluster_feature_dim, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Positional embedding for spatial coordinates
        self.position_embedding = SinusoidalPositionEmbedding(
            embedding_dim=pos_embedding_dim,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            min_freq=pos_min_freq,
            max_freq=pos_max_freq,
            scale_factor=pos_scale
        )
        
        # PCA features projection
        self.pca_projection = nn.Linear(pca_feature_dim, hidden_dim)
        
        # Combine transformer output, position embedding, and PCA features
        combined_dim = hidden_dim + pos_embedding_dim + hidden_dim
        self.combined_projection = nn.Linear(combined_dim, gnn_hidden_dim)
        
        # GATv2Conv layers for edge prediction
        self.conv1 = GATv2Conv(gnn_hidden_dim, gnn_hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATv2Conv(gnn_hidden_dim * 4, gnn_hidden_dim * 2, heads=1, dropout=dropout)
        
        # Edge prediction layers
        self.edge_pred = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 4, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, 1)
        )
    
    def get_node_embeddings(self, data):
        # Process cluster features with transformer
        # data.cluster_features shape: [num_nodes, 16, 48]
        transformed_features = self.transformer(data.cluster_features)  # Shape: [num_nodes, hidden_dim]
        
        # Get position embeddings
        # data.pos shape: [num_nodes, 3]
        pos_embeddings = self.position_embedding(data.pos)  # Shape: [num_nodes, pos_embedding_dim]
        
        # Process PCA features
        pca_emb = self.pca_projection(data.pca_features)  # Shape: [num_nodes, hidden_dim]
        
        # Combine transformer output, position embeddings, and PCA features
        combined = torch.cat([transformed_features, pos_embeddings, pca_emb], dim=1)
        node_features = self.combined_projection(combined)  # Shape: [num_nodes, gnn_hidden_dim]
        
        return node_features
    
    def forward(self, data):
        # Get node embeddings
        x = self.get_node_embeddings(data)
        
        # Store node features in data for later use
        data.x = x
        
        # Apply GATv2Conv layers
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, data.edge_index)  # Shape: [num_nodes, gnn_hidden_dim * 2]
        
        # Get node pairs for edge prediction
        src, dst = data.edge_index
        
        # Get node features for each node in the edge
        src_feat = x[src]  # Shape: [num_edges, gnn_hidden_dim * 2]
        dst_feat = x[dst]  # Shape: [num_edges, gnn_hidden_dim * 2]
        
        # Combine node features for edge prediction
        edge_feat = torch.cat([src_feat, dst_feat], dim=1)  # Shape: [num_edges, gnn_hidden_dim * 4]
        
        # Predict edge probabilities
        edge_pred = self.edge_pred(edge_feat).squeeze(-1)  # Shape: [num_edges]
        
        return edge_pred
