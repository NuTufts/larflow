"""
Transformer Encoder for MAE

Implements transformer encoder blocks with options for standard and
efficient attention mechanisms to handle large numbers of spacepoints.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_flash_attention: Use PyTorch's scaled_dot_product_attention if available
    """

    def __init__(self, d_model, n_heads, dropout=0.0, use_flash_attention=True):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model (given {d_model}) must be divisible by n_heads (given {n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flash_attention = use_flash_attention

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, attention_mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (B, N, d_model)
            attention_mask: Optional mask of shape (B, N, N) or (B, 1, N, N)

        Returns:
            Output tensor of shape (B, N, d_model)
        """
        B, N, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, N, 3*d_model)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's efficient attention
            attn_mask = attention_mask
            if attention_mask is not None and attention_mask.dim() == 3:
                attn_mask = attention_mask.unsqueeze(1)  # (B, 1, N, N)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # Manual attention computation
            scale = 1.0 / math.sqrt(self.d_head)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, N, N)

            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.out_proj(out)

        return out


class LinearAttention(nn.Module):
    """
    Linear attention with O(N) complexity instead of O(N^2).

    Uses kernel feature maps to approximate softmax attention.
    Useful for handling large numbers of spacepoints.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        feature_map: Type of feature map ('elu', 'relu', 'softmax')
    """

    def __init__(self, d_model, n_heads, dropout=0.0, feature_map='elu'):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.feature_map = feature_map

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _feature_map(self, x):
        """Apply feature map to approximate softmax."""
        if self.feature_map == 'elu':
            return F.elu(x) + 1
        elif self.feature_map == 'relu':
            return F.relu(x)
        elif self.feature_map == 'softmax':
            return F.softmax(x, dim=-1)
        else:
            return F.elu(x) + 1

    def forward(self, x, attention_mask=None):
        """
        Forward pass with linear attention.

        Args:
            x: Input tensor of shape (B, N, d_model)
            attention_mask: Not used in linear attention (included for API compatibility)

        Returns:
            Output tensor of shape (B, N, d_model)
        """
        B, N, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply feature map
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Linear attention: O(N) complexity
        # Instead of Q @ K^T @ V, compute K^T @ V first, then Q @ (K^T @ V)
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)  # (B, n_heads, d_head, d_head)
        qkv = torch.einsum('bhnd,bhdm->bhnm', q, kv)  # (B, n_heads, N, d_head)

        # Normalize
        k_sum = k.sum(dim=2, keepdim=True)  # (B, n_heads, 1, d_head)
        normalizer = torch.einsum('bhnd,bhkd->bhnk', q, k_sum).clamp(min=1e-6)  # (B, n_heads, N, 1)
        out = qkv / normalizer

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.dropout(self.out_proj(out))

        return out


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension (default: 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()

        d_ff = d_ff or 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with pre-norm architecture.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        attention_type: Type of attention ('standard', 'linear')
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0,
                 attention_type='standard'):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if attention_type == 'standard':
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        elif attention_type == 'linear':
            self.attention = LinearAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Forward pass through encoder block.

        Args:
            x: Input tensor of shape (B, N, d_model)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (B, N, d_model)
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = residual + self.dropout(x)

        # Pre-norm feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder blocks.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder blocks
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        attention_type: Type of attention for all layers
    """

    def __init__(self, d_model, n_heads, n_layers, d_ff=None, dropout=0.0,
                 attention_type='standard'):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_type=attention_type
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None, return_all_layers=False):
        """
        Forward pass through encoder stack.

        Args:
            x: Input tensor of shape (B, N, d_model)
            attention_mask: Optional attention mask
            return_all_layers: If True, return outputs from all layers

        Returns:
            If return_all_layers: List of tensors from each layer
            Otherwise: Final output tensor of shape (B, N, d_model)
        """
        all_outputs = []

        for layer in self.layers:
            x = layer(x, attention_mask)
            if return_all_layers:
                all_outputs.append(x)

        x = self.norm(x)

        if return_all_layers:
            all_outputs[-1] = x  # Replace last with normalized version
            return all_outputs

        return x


class WindowedAttention(nn.Module):
    """
    Windowed/local attention for handling very large sequences.

    Divides the sequence into windows and computes attention within each window.
    Optionally includes global tokens that attend to all positions.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of local attention window
        n_global_tokens: Number of global tokens (attend to all positions)
        dropout: Dropout probability
    """

    def __init__(self, d_model, n_heads, window_size=256, n_global_tokens=0,
                 dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.n_global_tokens = n_global_tokens

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x, attention_mask=None):
        """
        Forward pass with windowed attention.

        Args:
            x: Input tensor of shape (B, N, d_model)
            attention_mask: Not fully supported for windowed attention

        Returns:
            Output tensor of shape (B, N, d_model)
        """
        B, N, D = x.shape

        # Handle global tokens separately if present
        if self.n_global_tokens > 0:
            global_tokens = x[:, :self.n_global_tokens]
            local_tokens = x[:, self.n_global_tokens:]
            N_local = N - self.n_global_tokens
        else:
            local_tokens = x
            N_local = N

        # Pad sequence to be divisible by window size
        pad_len = (self.window_size - N_local % self.window_size) % self.window_size
        if pad_len > 0:
            local_tokens = F.pad(local_tokens, (0, 0, 0, pad_len))

        N_padded = local_tokens.shape[1]
        n_windows = N_padded // self.window_size

        # Reshape into windows
        local_tokens = local_tokens.reshape(B * n_windows, self.window_size, D)

        # Apply attention within each window
        local_out = self.attention(local_tokens)

        # Reshape back
        local_out = local_out.reshape(B, N_padded, D)

        # Remove padding
        if pad_len > 0:
            local_out = local_out[:, :N_local]

        # Handle global tokens
        if self.n_global_tokens > 0:
            # Global tokens attend to all positions
            global_context = torch.cat([global_tokens, local_out], dim=1)
            global_out = self.attention(global_context)[:, :self.n_global_tokens]

            out = torch.cat([global_out, local_out], dim=1)
        else:
            out = local_out

        return out
