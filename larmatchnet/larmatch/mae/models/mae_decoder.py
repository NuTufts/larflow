"""
MAE Decoder Module

A shallow transformer decoder for reconstructing masked pixel values
from the encoder representations.
"""

import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoderBlock


class MAEDecoder(nn.Module):
    """
    Shallow decoder for MAE reconstruction.

    Following the MAE paper, this decoder is significantly shallower than
    the encoder (typically 2-4 layers vs 12+ for encoder).

    The decoder takes:
    - Encoded representations from unmasked tokens
    - Learnable mask tokens at masked positions
    - Position embeddings for all tokens

    And outputs predictions for the masked pixel values.

    Args:
        encoder_dim: Dimension of encoder output
        decoder_dim: Hidden dimension for decoder (typically smaller than encoder)
        n_heads: Number of attention heads
        n_layers: Number of decoder layers (typically 2-4)
        output_dim: Dimension of output (e.g., 3 for pixel values in 3 planes)
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        attention_type: Type of attention ('standard', 'linear')
    """

    def __init__(self, encoder_dim, decoder_dim=256, n_heads=8, n_layers=2,
                 output_dim=3, d_ff=None, dropout=0.0, attention_type='standard'):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim

        # Project encoder output to decoder dimension
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder transformer blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=decoder_dim,
                n_heads=n_heads,
                d_ff=d_ff or 4 * decoder_dim,
                dropout=dropout,
                attention_type=attention_type
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(decoder_dim)

        # Prediction head for pixel values
        self.pred_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize prediction head
        for module in self.pred_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize encoder to decoder projection
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        nn.init.zeros_(self.encoder_to_decoder.bias)

    def forward(self, encoder_output, mask_indices, pos_embeddings,
                return_all_tokens=False):
        """
        Forward pass through MAE decoder.

        Args:
            encoder_output: Encoded unmasked tokens, shape (B, N_unmasked, encoder_dim)
            mask_indices: Boolean mask indicating masked positions, shape (B, N_total)
            pos_embeddings: Position embeddings for all tokens, shape (B, N_total, decoder_dim)
            return_all_tokens: If True, return predictions for all tokens

        Returns:
            predictions: Predicted pixel values for masked tokens, shape (B, N_masked, output_dim)
                        or (B, N_total, output_dim) if return_all_tokens=True
        """
        B, N_unmasked, _ = encoder_output.shape
        N_total = mask_indices.shape[1]
        N_masked = mask_indices.sum(dim=1).max().item()

        device = encoder_output.device

        # Project encoder output to decoder dimension
        unmasked_tokens = self.encoder_to_decoder(encoder_output)

        # Expand mask token for all masked positions
        mask_tokens = self.mask_token.expand(B, N_masked, -1)

        # Combine unmasked and masked tokens in correct positions
        # This requires careful indexing based on mask_indices
        full_sequence = self._assemble_tokens(
            unmasked_tokens, mask_tokens, mask_indices, pos_embeddings
        )

        # Pass through decoder layers
        x = full_sequence
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Predict pixel values
        predictions = self.pred_head(x)

        if return_all_tokens:
            return predictions

        # Extract only predictions for masked positions
        masked_predictions = self._extract_masked(predictions, mask_indices)

        return masked_predictions

    def _assemble_tokens(self, unmasked_tokens, mask_tokens, mask_indices, pos_embeddings):
        """
        Assemble the full sequence by placing unmasked and mask tokens in correct positions.

        Args:
            unmasked_tokens: Shape (B, N_unmasked, decoder_dim)
            mask_tokens: Shape (B, N_masked, decoder_dim)
            mask_indices: Boolean mask, shape (B, N_total)
            pos_embeddings: Shape (B, N_total, decoder_dim)

        Returns:
            full_sequence: Shape (B, N_total, decoder_dim)
        """
        B, N_total = mask_indices.shape
        device = unmasked_tokens.device

        # Initialize full sequence
        full_sequence = torch.zeros(B, N_total, unmasked_tokens.shape[-1], device=device)

        for b in range(B):
            # Get indices for this batch
            masked_idx = mask_indices[b].nonzero(as_tuple=True)[0]
            unmasked_idx = (~mask_indices[b]).nonzero(as_tuple=True)[0]

            # Place tokens
            full_sequence[b, unmasked_idx] = unmasked_tokens[b, :len(unmasked_idx)]
            full_sequence[b, masked_idx] = mask_tokens[b, :len(masked_idx)]

        # Add position embeddings
        full_sequence = full_sequence + pos_embeddings

        return full_sequence

    def _extract_masked(self, predictions, mask_indices):
        """
        Extract predictions for masked positions only.

        Args:
            predictions: Shape (B, N_total, output_dim)
            mask_indices: Boolean mask, shape (B, N_total)

        Returns:
            masked_predictions: Shape (B, N_masked_max, output_dim)
        """
        B, N_total, output_dim = predictions.shape
        N_masked_max = mask_indices.sum(dim=1).max().item()

        masked_predictions = torch.zeros(B, N_masked_max, output_dim, device=predictions.device)

        for b in range(B):
            masked_idx = mask_indices[b].nonzero(as_tuple=True)[0]
            n_masked = len(masked_idx)
            masked_predictions[b, :n_masked] = predictions[b, masked_idx]

        return masked_predictions


class MAEDecoderV2(nn.Module):
    """
    Alternative MAE decoder that processes all tokens together more efficiently.

    Instead of reassembling the sequence, this decoder:
    1. Projects encoded tokens
    2. Adds position information to mask tokens
    3. Runs cross-attention between mask tokens and encoded tokens
    4. Predicts pixel values for masked positions

    This can be more memory efficient for very large sequences.

    Args:
        encoder_dim: Dimension of encoder output
        decoder_dim: Hidden dimension for decoder
        n_heads: Number of attention heads
        n_layers: Number of decoder layers
        output_dim: Dimension of output
        dropout: Dropout probability
    """

    def __init__(self, encoder_dim, decoder_dim=256, n_heads=8, n_layers=2,
                 output_dim=3, dropout=0.0):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # Project encoder output
        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Cross-attention layers (queries from mask tokens, keys/values from encoded)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(decoder_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(decoder_dim)

        # Output projection
        self.pred_head = nn.Linear(decoder_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder_proj.weight)
        nn.init.zeros_(self.encoder_proj.bias)
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

    def forward(self, encoder_output, n_masked, mask_pos_embeddings):
        """
        Forward pass through decoder.

        Args:
            encoder_output: Encoded unmasked tokens, shape (B, N_unmasked, encoder_dim)
            n_masked: Number of masked tokens to predict
            mask_pos_embeddings: Position embeddings for masked tokens, shape (B, n_masked, decoder_dim)

        Returns:
            predictions: Shape (B, n_masked, output_dim)
        """
        B = encoder_output.shape[0]

        # Project encoder output
        memory = self.encoder_proj(encoder_output)

        # Initialize query tokens (mask tokens + position embeddings)
        queries = self.mask_token.expand(B, n_masked, -1) + mask_pos_embeddings

        # Cross-attention layers
        for layer in self.cross_attention_layers:
            queries = layer(queries, memory)

        queries = self.norm(queries)

        # Predict
        predictions = self.pred_head(queries)

        return predictions


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for MAE decoder.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, queries, memory):
        """
        Forward pass.

        Args:
            queries: Query tokens, shape (B, N_q, d_model)
            memory: Key/value tokens from encoder, shape (B, N_kv, d_model)

        Returns:
            Updated queries, shape (B, N_q, d_model)
        """
        # Cross-attention
        residual = queries
        queries = self.norm1(queries)
        memory = self.norm_memory(memory)
        attn_out, _ = self.cross_attn(queries, memory, memory)
        queries = residual + attn_out

        # Feed-forward
        residual = queries
        queries = self.norm2(queries)
        queries = residual + self.ff(queries)

        return queries
