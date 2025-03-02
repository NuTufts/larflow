import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ResGATv2Block(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.5, norm_type='graph'):
        super(ResGATv2Block, self).__init__()
        
        assert out_channels%heads==0, f"out_channels({out_channels}) must be divisble by heads({heads})"

        # GATv2 layer
        self.gatv2 = GATv2Conv(
            in_channels=in_channels, 
            out_channels=out_channels // heads,  # Divide by heads to maintain dimension
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # Projection layer (if dimensions don't match)
        self.use_projection = (in_channels != out_channels)
        if self.use_projection:
            self.projection = nn.Linear(in_channels, out_channels)
        
        # Normalization layer
        self.norm_type = norm_type
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(out_channels)
        elif norm_type == 'graph':
            from torch_geometric.nn import GraphNorm
            self.norm = GraphNorm(out_channels)
        
    def forward(self, x, edge_index, batch=None):
        # Store original input for residual connection
        identity = x
        
        # Apply GATv2 layer
        out = self.gatv2(x, edge_index)
        
        # Apply normalization BEFORE the residual addition
        if self.norm_type == 'graph' and batch is not None:
            out = self.norm(out, batch)
        else:
            out = self.norm(out)
        
        # Apply projection if necessary
        if self.use_projection:
            identity = self.projection(identity)
        
        # Add residual connection
        out = out + identity
        
        # Apply activation after residual connection
        out = F.elu(out)
        
        return out


# Example usage in a full GNN model
class ResidualGATv2Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(ResidualGATv2Net, self).__init__()
        
        self.num_layers = num_layers
        
        # Initial convolution to get to hidden dimension
        self.conv_in = GATv2Conv(in_channels, hidden_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(
                ResidualGATv2Block(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=4,
                    norm_type='layer'  # You can choose 'batch', 'layer', or 'graph'
                )
            )
        
        # Output layer
        self.out_layer = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch=None):
        # Initial conv
        x = self.conv_in(x, edge_index)
        x = F.relu(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x, edge_index, batch)
        
        # Global pooling
        # (Depending on your task, you might want to add pooling here)
        
        # Output projection
        x = self.out_layer(x)
        
        return x