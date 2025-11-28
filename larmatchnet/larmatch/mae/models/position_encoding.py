"""
Position Encoding Modules for 3D Spacepoints

Provides various position encoding schemes for encoding (x, y, z) coordinates
of spacepoints into high-dimensional representations.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEncoding3D(nn.Module):
    """
    3D sinusoidal position encoding following the Transformer paper (Vaswani et al.).

    For each spatial dimension (x, y, z), we generate sinusoidal encodings
    and concatenate them to form the final position encoding.

    Args:
        d_model: Total dimension of the position encoding (must be divisible by 6)
        max_spatial_extent: Maximum coordinate value for normalization (in cm)
        temperature: Temperature for frequency scaling (higher = lower frequencies)
    """

    def __init__(self, d_model, max_spatial_extent=1000.0, temperature=10000.0):
        super().__init__()

        if d_model % 6 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 6 for 3D sinusoidal encoding")

        self.d_model = d_model
        self.d_per_dim = d_model // 3  # Features per spatial dimension
        self.max_spatial_extent = max_spatial_extent
        self.temperature = temperature

        # Precompute frequency bands
        # Each dimension gets d_model/3 features, half sin and half cos
        half_d = self.d_per_dim // 2
        freq_bands = torch.exp(
            torch.arange(half_d, dtype=torch.float32) *
            (-math.log(temperature) / half_d)
        )
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, positions):
        """
        Generate sinusoidal position encodings for 3D coordinates.

        Args:
            positions: Tensor of shape (N, 3) containing (x, y, z) coordinates

        Returns:
            Tensor of shape (N, d_model) containing position encodings
        """
        # Normalize positions to [0, 1] range
        positions_norm = positions / self.max_spatial_extent

        encodings = []
        for dim in range(3):
            # Get coordinate for this dimension
            coord = positions_norm[:, dim:dim+1]  # (N, 1)

            # Scale by frequency bands
            scaled = coord * self.freq_bands.unsqueeze(0) * math.pi  # (N, half_d)

            # Compute sin and cos
            sin_enc = torch.sin(scaled)
            cos_enc = torch.cos(scaled)

            # Interleave sin and cos
            dim_encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # (N, half_d, 2)
            dim_encoding = dim_encoding.reshape(positions.shape[0], -1)  # (N, d_per_dim)

            encodings.append(dim_encoding)

        # Concatenate encodings from all dimensions
        return torch.cat(encodings, dim=-1)  # (N, d_model)


class LearnableFourierEncoding(nn.Module):
    """
    Learnable Fourier feature encoding for 3D positions.

    Uses randomly initialized but learnable frequency matrices to project
    positions into a high-dimensional space, followed by sin/cos activations.

    Args:
        d_model: Output dimension of the encoding
        sigma: Standard deviation for initializing frequency matrix
    """

    def __init__(self, d_model, sigma=10.0):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) must be even for Fourier encoding")

        self.d_model = d_model
        half_d = d_model // 2

        # Learnable frequency matrix B: (3, d_model/2)
        # Initialized from normal distribution
        self.B = nn.Parameter(torch.randn(3, half_d) * sigma)

    def forward(self, positions):
        """
        Generate learnable Fourier position encodings.

        Args:
            positions: Tensor of shape (N, 3) containing (x, y, z) coordinates

        Returns:
            Tensor of shape (N, d_model) containing position encodings
        """
        # Project positions: (N, 3) @ (3, half_d) = (N, half_d)
        projected = 2 * math.pi * torch.matmul(positions, self.B)

        # Apply sin and cos
        sin_enc = torch.sin(projected)
        cos_enc = torch.cos(projected)

        # Concatenate
        return torch.cat([sin_enc, cos_enc], dim=-1)  # (N, d_model)


class RotaryPositionEncoding3D(nn.Module):
    """
    Rotary Position Encoding (RoPE) adapted for 3D coordinates.

    RoPE encodes positions through rotation matrices applied to query/key vectors
    in attention. This module generates the rotation matrices for 3D positions.

    Args:
        d_head: Dimension of each attention head (must be divisible by 6)
        max_spatial_extent: Maximum coordinate value for normalization
        base: Base for frequency computation
    """

    def __init__(self, d_head, max_spatial_extent=1000.0, base=10000.0):
        super().__init__()

        if d_head % 6 != 0:
            raise ValueError(f"d_head ({d_head}) must be divisible by 6 for 3D RoPE")

        self.d_head = d_head
        self.d_per_dim = d_head // 3
        self.max_spatial_extent = max_spatial_extent

        # Compute inverse frequencies for each dimension pair
        half_d = self.d_per_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_d, dtype=torch.float32) / half_d))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        """
        Compute rotation matrices for RoPE.

        Args:
            positions: Tensor of shape (N, 3) containing (x, y, z) coordinates

        Returns:
            cos_pos, sin_pos: Tensors of shape (N, d_head) for applying rotation
        """
        # Normalize positions
        positions_norm = positions / self.max_spatial_extent

        cos_list = []
        sin_list = []

        for dim in range(3):
            coord = positions_norm[:, dim:dim+1]  # (N, 1)

            # Compute angles
            angles = coord * self.inv_freq.unsqueeze(0)  # (N, half_d)

            # Repeat for pairs
            angles = angles.repeat(1, 2)  # (N, d_per_dim)

            cos_list.append(torch.cos(angles))
            sin_list.append(torch.sin(angles))

        cos_pos = torch.cat(cos_list, dim=-1)  # (N, d_head)
        sin_pos = torch.cat(sin_list, dim=-1)  # (N, d_head)

        return cos_pos, sin_pos

    @staticmethod
    def apply_rotary_embedding(x, cos_pos, sin_pos):
        """
        Apply rotary position embedding to input tensor.

        Args:
            x: Input tensor of shape (..., d_head)
            cos_pos: Cosine components of shape (N, d_head)
            sin_pos: Sine components of shape (N, d_head)

        Returns:
            Rotated tensor of same shape as x
        """
        # Split into pairs for rotation
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        cos_pos_1 = cos_pos[..., ::2]
        cos_pos_2 = cos_pos[..., 1::2]
        sin_pos_1 = sin_pos[..., ::2]
        sin_pos_2 = sin_pos[..., 1::2]

        # Apply rotation
        rotated_1 = x1 * cos_pos_1 - x2 * sin_pos_1
        rotated_2 = x1 * sin_pos_1 + x2 * cos_pos_2

        # Interleave back
        rotated = torch.stack([rotated_1, rotated_2], dim=-1)
        return rotated.flatten(-2)


class CombinedPositionEncoding(nn.Module):
    """
    Combines position encoding with image features.

    This module takes image features from the backbone and adds/concatenates
    position encodings to create the final spacepoint representation.

    Args:
        image_feat_dim: Dimension of image features from backbone
        pos_encoding_dim: Dimension of position encoding
        output_dim: Desired output dimension
        combination_mode: How to combine features ('add', 'concat', 'gate')
    """

    def __init__(self, image_feat_dim, pos_encoding_dim, output_dim,
                 combination_mode='add', pos_encoding_type='sinusoidal',
                 max_spatial_extent=1000.0):
        super().__init__()

        self.combination_mode = combination_mode
        self.image_feat_dim = image_feat_dim
        self.pos_encoding_dim = pos_encoding_dim
        self.output_dim = output_dim

        # Position encoding module
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionEncoding3D(
                pos_encoding_dim, max_spatial_extent=max_spatial_extent
            )
        elif pos_encoding_type == 'learnable_fourier':
            self.pos_encoder = LearnableFourierEncoding(pos_encoding_dim)
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")

        # Projection layers based on combination mode
        if combination_mode == 'add':
            # Project both to same dimension, then add
            self.image_proj = nn.Linear(image_feat_dim, output_dim)
            self.pos_proj = nn.Linear(pos_encoding_dim, output_dim)
        elif combination_mode == 'concat':
            # Concatenate and project
            self.proj = nn.Linear(image_feat_dim + pos_encoding_dim, output_dim)
        elif combination_mode == 'gate':
            # Use position to gate image features
            self.image_proj = nn.Linear(image_feat_dim, output_dim)
            self.gate_net = nn.Sequential(
                nn.Linear(pos_encoding_dim, output_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown combination_mode: {combination_mode}")

    def forward(self, image_features, positions):
        """
        Combine image features with position encodings.

        Args:
            image_features: Tensor of shape (N, image_feat_dim)
            positions: Tensor of shape (N, 3) containing (x, y, z) coordinates

        Returns:
            Tensor of shape (N, output_dim)
        """
        pos_encoding = self.pos_encoder(positions)

        if self.combination_mode == 'add':
            return self.image_proj(image_features) + self.pos_proj(pos_encoding)
        elif self.combination_mode == 'concat':
            combined = torch.cat([image_features, pos_encoding], dim=-1)
            return self.proj(combined)
        elif self.combination_mode == 'gate':
            gate = self.gate_net(pos_encoding)
            return self.image_proj(image_features) * gate
