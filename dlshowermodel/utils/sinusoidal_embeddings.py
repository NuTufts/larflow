import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim, 
                x_range=(-500, 500), y_range=(-500, 500), z_range=(-500, 500), 
                min_freq=1.0, max_freq=1000.0, 
                scale_factor=1.0):
        """
        Sinusoidal position embedding for 3D spatial coordinates.
        
        Args:
            embedding_dim: Dimension of the output embedding (must be divisible by 6)
            x_range: Range of x coordinates (min, max) in cm
            y_range: Range of y coordinates (min, max) in cm
            z_range: Range of z coordinates (min, max) in cm
            min_freq: Minimum frequency for the sinusoidal functions
            max_freq: Maximum frequency for the sinusoidal functions
            scale_factor: Scale factor to control the sensitivity to small distances
        """
        super(SinusoidalPositionEmbedding, self).__init__()
        
        assert embedding_dim % 6 == 0, "embedding_dim must be divisible by 6 (2 functions * 3 coordinates)"
        
        self.embedding_dim = embedding_dim
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.scale_factor = scale_factor
        
        # Number of frequency bands per coordinate
        self.num_bands = embedding_dim // 6
        
        # Create frequency bands with logarithmic scaling
        self.frequencies = math.pi*torch.exp(
            torch.linspace(
                math.log(min_freq), 
                math.log(max_freq), 
                self.num_bands
            )
        ) * scale_factor
        
    def normalize_coordinate(self, coord, coord_range):
        """Normalize coordinate to [-1, 1] range"""
        min_val, max_val = coord_range
        return 2 * (coord - min_val) / (max_val - min_val) - 1
    
    def forward(self, positions):
        """
        Create sinusoidal embeddings for 3D positions.
        
        Args:
            positions: Tensor of shape [batch_size, 3] containing (x, y, z) coordinates
            
        Returns:
            Tensor of shape [batch_size, embedding_dim] containing position embeddings
        """
        batch_size = positions.shape[0]
        
        # Normalize coordinates to [-1, 1]
        x = self.normalize_coordinate(positions[:, 0], self.x_range)
        y = self.normalize_coordinate(positions[:, 1], self.y_range)
        z = self.normalize_coordinate(positions[:, 2], self.z_range)
        
        # Reshape for broadcasting
        x = x.view(batch_size, 1)  # [batch_size, 1]
        y = y.view(batch_size, 1)  # [batch_size, 1]
        z = z.view(batch_size, 1)  # [batch_size, 1]
        
        # Create frequency products for each coordinate
        # Shape: [batch_size, num_bands]
        x_freqs = x * self.frequencies.to(positions.device)
        y_freqs = y * self.frequencies.to(positions.device)
        z_freqs = z * self.frequencies.to(positions.device)
        
        # Apply sine and cosine to get embeddings
        # Each results in shape [batch_size, num_bands]
        x_sin = torch.sin(x_freqs)
        x_cos = torch.cos(x_freqs)
        y_sin = torch.sin(y_freqs)
        y_cos = torch.cos(y_freqs)
        z_sin = torch.sin(z_freqs)
        z_cos = torch.cos(z_freqs)
        
        # Concatenate all embeddings
        # Final shape: [batch_size, embedding_dim]
        embeddings = torch.cat([x_sin, x_cos, y_sin, y_cos, z_sin, z_cos], dim=1)
        
        return embeddings

# Example usage
if __name__ == "__main__":
    # Create a position embedding for 3D coordinates
    pos_embedding = SinusoidalPositionEmbedding(
        embedding_dim=48,  # Must be divisible by 6
        x_range=(-500, 500),
        y_range=(-500, 500),
        z_range=(-500, 500),
        min_freq=1.0,
        max_freq=1000.0,
        scale_factor=1.0
    )
    print("Frequencies:")
    print(pos_embedding.frequencies)
    
    # Example positions [batch_size, 3]
    positions = torch.tensor([
        [0.0, 0.0, 0.0],    # Center position
        [0.0, 0.0, 0.01],    # 1cm shift in x
        [0.0, 0.0, 0.1],    # 1cm shift in x
        [0.0, 0.0, 1.0],    # 1cm shift in x
        [0.0, 0.0, 10.0],    # 1cm shift in y
        [0.0, 0.0, 100.0],    # 1cm shift in y
        [0.0, 0.0, 500.0]    # 1cm shift in y
    ])
    
    # Get embeddings
    embeddings = pos_embedding(positions)
    print(f"Embedding shape: {embeddings.shape}")
    print("sin(z): ")
    print(embeddings[:,-16:-8])
    print("cos(z): ")
    print(embeddings[:,-8:])
    
    # Compute pairwise distances in the embedding space
    distances = torch.cdist(embeddings, embeddings)
    print("Pairwise distances in embedding space:")
    print(distances)