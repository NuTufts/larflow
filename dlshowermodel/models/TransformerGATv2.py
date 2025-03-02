import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATv2Conv
from dlshowermodel.utils.sinusoidal_embeddings import SinusoidalPositionEmbedding
from dlshowermodel.models.SetTransformer import SetTransformer

# Define the complete model combining Transformer, Position Embedding, and GATv2
class TransformerGATv2Model(nn.Module):
    def __init__(self, cluster_feature_dim=48, 
                pca_feature_dim=21, 
                pos_embedding_dim=48, 
                cluster_hidden_dim=64, 
                gnn_hidden_dim=48, 
                num_out_tokens=4,
                num_heads=4,  
                num_gcnn_layers=2, 
                num_gat_heads=4,
                dropout=0.1,
                cat_pos_embed=False,
                x_range=(-520, 520), y_range=(-520, 520), z_range=(-520, 520),
                pos_origin=(0.0,0.0,1036.0/2.0),
                pos_min_freq=1.0, pos_max_freq=1000.0, pos_scale=1.0):
        super(TransformerGATv2Model, self).__init__()
        
        # Transformer for processing cluster features
        self.transformer = SetTransformer(
            cluster_feature_dim, # dim_input
            num_out_tokens, # num_outputs
            cluster_hidden_dim,
            num_heads=num_heads,
            ln=True)
        
        # Positional embeddings for the spatial features

        # Positional embedding for spatial coordinates for graph clusters
        self.position_embedding = SinusoidalPositionEmbedding(
            embedding_dim=pos_embedding_dim,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            min_freq=pos_min_freq,
            max_freq=pos_max_freq,
            scale_factor=pos_scale
        )

        self.pos_origin = torch.tensor(pos_origin,dtype=torch.float,requires_grad=False)
        
        # larmatch vector projection into cluster transformer input space
        self.larmatch_projection = nn.Linear(cluster_feature_dim,cluster_feature_dim)
        
        # Combine transformer output + PCA + charge feats
        dim_charge_feats = 3
        combined_dim = cluster_hidden_dim*num_out_tokens + pca_feature_dim + dim_charge_feats
        self.cat_pos_embed = cat_pos_embed
        if self.cat_pos_embed:
            combined_dim += pos_embedding_dim
        self.combined_projection = nn.Linear(combined_dim, pos_embedding_dim)
        
        # GATv2Conv layers for edge prediction
        self.num_gcnn_layers = num_gcnn_layers
        for ilayer in range(num_gcnn_layers):
            layername = f'gatv2conv_layer{ilayer}'
            ninput_dims = gnn_hidden_dim
            noutput_dims = gnn_hidden_dim
            nheads = num_gat_heads
            # mods for first layer
            if ilayer>0:
                ninput_dims *= num_gat_heads
            if ilayer==0:
                ninput_dims = pos_embedding_dim
            # mods for last layer
            if ilayer==num_gcnn_layers-1:
                noutput_dims *= 2
                nheads = 1
            conv = GATv2Conv(ninput_dims, noutput_dims, heads=nheads, dropout=dropout)
            setattr(self,layername,conv)
        
        # Edge prediction layers
        self.edge_pred = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 4, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, 1)
        )
    
    def get_node_embeddings(self, data):
        # Process cluster features with transformer

        # project larmatch feature vector into embedding space
        # in: [num_nodes,16,48]
        # out: [num_nodes,16,48]
        N = data.cluster_features.shape[0]
        lmprojection = self.larmatch_projection( data.cluster_features ) 

        # add sampled pos embedding
        #lmprojection = lmprojection + data.sampled_pos_embed
        #print("lmprojection out: ",lmprojection.shape)

        # data.cluster_features shape: [num_nodes, 16, 48]
        transformed_features = self.transformer(lmprojection).view(N,-1)  # Shape: [num_nodes, hidden_dim]
        #print("transformed_features: ",transformed_features.shape)

        # Get position embeddings
        # data.pos shape: [num_nodes, 3]
        # first move node positions relative to origin
        node_pos = data.pos-self.pos_origin.to(data.pos.device)

        # make the position embeddings
        pos_embeddings = self.position_embedding(node_pos)  # Shape: [num_nodes, pos_embedding_dim]
        #print("pos_embeddings: ",pos_embeddings.shape)
        
        # Combine transformer output, position embeddings, and PCA features
        if not self.cat_pos_embed:
            combined = torch.cat([transformed_features, data.pca_features, data.pixsum_features], dim=1)
            node_features = self.combined_projection(combined) + pos_embeddings  # Shape: [num_nodes, gnn_hidden_dim]
        else:
            combined = torch.cat([transformed_features, data.pca_features, data.pixsum_features, pos_embeddings], dim=1)
            node_features = self.combined_projection(combined)  # Shape: [num_nodes, gnn_hidden_dim]
        
        
        return node_features
    
    def forward(self, data):
        # Get graph node embeddings [will go through transformer]
        x = self.get_node_embeddings(data)
        
        # Store node features in data for later use
        data.x = x
        
        # Apply GATv2Conv layers
        for ilayer in range(self.num_gcnn_layers):
            conv = getattr(self,f'gatv2conv_layer{ilayer}')
            x = conv(x, data.edge_index)
            if ilayer<self.num_gcnn_layers-1:
                x = F.elu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
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
