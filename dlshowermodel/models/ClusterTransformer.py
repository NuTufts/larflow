import torch
import torch.nn as nn


# Define the Transformer for processing the 16 feature vectors for each node
# The goal is that this part of the network tells what information about the
# cluster is useful to promote to the graph
# Should be a back and forth based on node vector?
class ClusterTransformer(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(ClusterTransformer, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection to create node embeddings
        self.output_projection = nn.Linear(input_dim * 16, hidden_dim)
        print(self.output_projection)
        
    def forward(self, x):
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # Shape: [batch_size, 16, hidden_dim]
        print("transformer out: ",x.shape)
        # Flatten and project to create node embeddings
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Shape: [batch_size, 16 * hidden_dim]
        print("x.view: ",x.shape)
        x = self.output_projection(x)  # Shape: [batch_size, hidden_dim]
        
        return x