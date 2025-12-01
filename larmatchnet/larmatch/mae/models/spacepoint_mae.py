"""
Spacepoint MAE - Main Model

The main Masked Auto-Encoder model for learning spacepoint representations.
Combines the image backbone, position encoding, transformer encoder, and
MAE decoder for self-supervised pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .position_encoding import SinusoidalPositionEncoding3D, CombinedPositionEncoding
from .transformer_encoder import TransformerEncoder
from .mae_decoder import MAEDecoder, MAEDecoderV2
from .mae_backbone import MAEBackbone, create_backbone


class SpacepointMAE(nn.Module):
    """
    Masked Auto-Encoder for Spacepoint Representations.

    This model implements the MAE pretraining approach for learning robust
    encodings of energy deposition spacepoints in LArTPC data.

    Architecture:
    1. Image backbone (UNet) extracts features from wireplane images
    2. Spacepoint features are extracted for each 3D point
    3. Position encoding is added to spacepoint features
    4. Random masking is applied
    5. Unmasked tokens go through transformer encoder
    6. Decoder reconstructs masked pixel values
    7. Optional: contrastive loss, auxiliary tasks

    Args:
        backbone_config: Configuration for image backbone
        d_model: Dimension of transformer model
        encoder_layers: Number of transformer encoder layers
        encoder_heads: Number of attention heads in encoder
        decoder_layers: Number of decoder layers
        decoder_heads: Number of attention heads in decoder
        decoder_dim: Dimension of decoder (typically smaller than d_model)
        output_dim: Dimension of reconstruction target (3 for pixel values)
        mask_ratio: Fraction of tokens to mask
        dropout: Dropout probability
        attention_type: Type of attention ('standard', 'linear')
        pos_encoding_type: Type of position encoding ('sinusoidal', 'learnable_fourier')
        max_spatial_extent: Maximum coordinate value for position encoding
    """

    def __init__(
        self,
        backbone_config: dict = None,
        d_model: int = 240, # must be a multiple of 6
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        decoder_dim: int = 132,
        output_dim: int = 3,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
        attention_type: str = 'standard',
        pos_encoding_type: str = 'sinusoidal',
        max_spatial_extent: float = 1000.0,
        use_auxiliary_heads: bool = True,
        num_ssnet_classes: int = 5,
        num_keypoint_classes: int = 6,
    ):
        super().__init__()

        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.output_dim = output_dim
        self.use_auxiliary_heads = use_auxiliary_heads

        # Image feature backbone
        backbone_config = backbone_config or {}
        self.backbone = create_backbone(backbone_config)
        backbone_dim = self.backbone.get_output_dim()

        # Project backbone features to model dimension
        self.input_proj = nn.Linear(backbone_dim, d_model)

        # Position encoding
        self.pos_encoder = SinusoidalPositionEncoding3D(
            d_model=d_model,
            max_spatial_extent=max_spatial_extent
        )

        # CLS token for global representation (optional)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=encoder_heads,
            n_layers=encoder_layers,
            dropout=dropout,
            attention_type=attention_type
        )

        # MAE decoder
        self.decoder = MAEDecoder(
            encoder_dim=d_model,
            decoder_dim=decoder_dim,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            output_dim=output_dim,
            dropout=dropout,
            attention_type=attention_type
        )

        # Decoder position encoding (same dimension as decoder)
        self.decoder_pos_encoder = SinusoidalPositionEncoding3D(
            d_model=decoder_dim,
            max_spatial_extent=max_spatial_extent
        )

        # Auxiliary heads for supervised losses
        if use_auxiliary_heads:
            # Ghost vs True classification
            self.ghost_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 2)
            )

            # SSNet particle classification
            self.ssnet_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, num_ssnet_classes)
            )

            # Keypoint scores regression
            self.keypoint_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, num_keypoint_classes)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        if self.use_auxiliary_heads:
            for head in [self.ghost_head, self.ssnet_head, self.keypoint_head]:
                for module in head.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)

    def random_masking(self, x: torch.Tensor, positions: torch.Tensor,
                      mask_ratio: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on input tokens.

        Args:
            x: Input tokens, shape (B, N, d_model)
            positions: 3D positions, shape (B, N, 3)
            mask_ratio: Override default mask ratio

        Returns:
            x_unmasked: Unmasked tokens, shape (B, N_unmasked, d_model)
            mask: Boolean mask, True for masked positions, shape (B, N)
            ids_restore: Indices to restore original order, shape (B, N)
            positions_unmasked: Positions of unmasked tokens, shape (B, N_unmasked, 3)
        """
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        B, N, D = x.shape
        n_keep = int(N * (1 - mask_ratio))

        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first n_keep tokens
        ids_keep = ids_shuffle[:, :n_keep]

        # Gather unmasked tokens
        x_unmasked = torch.gather(x, dim=1,
                                  index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        positions_unmasked = torch.gather(positions, dim=1,
                                         index=ids_keep.unsqueeze(-1).expand(-1, -1, 3))

        # Create mask (True = masked)
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return x_unmasked, mask, ids_restore, positions_unmasked

    def forward_encoder(self, spacepoint_features: torch.Tensor,
                       positions: torch.Tensor,
                       mask_ratio: float = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder with masking.

        Args:
            spacepoint_features: Features from backbone, shape (B, N, backbone_dim)
            positions: 3D positions, shape (B, N, 3)
            mask_ratio: Override default mask ratio

        Returns:
            Dictionary containing:
                - encoded: Encoded representations, shape (B, N_unmasked, d_model)
                - mask: Boolean mask, shape (B, N)
                - ids_restore: Indices for restoration, shape (B, N)
                - positions_unmasked: Unmasked positions, shape (B, N_unmasked, 3)
                - positions_masked: Masked positions, shape (B, N_masked, 3)
        """
        B, N, _ = spacepoint_features.shape

        # Project to model dimension
        x = self.input_proj(spacepoint_features)

        # Add position encoding
        pos_enc = self.pos_encoder(positions.reshape(-1, 3)).reshape(B, N, -1)
        x = x + pos_enc

        # Apply masking
        x_unmasked, mask, ids_restore, positions_unmasked = self.random_masking(
            x, positions, mask_ratio
        )

        # Get masked positions for decoder
        n_masked = mask.sum(dim=1).max().item()
        masked_indices = mask.nonzero(as_tuple=False)
        positions_masked = torch.zeros(B, n_masked, 3, device=positions.device)
        for b in range(B):
            b_mask = masked_indices[masked_indices[:, 0] == b, 1]
            positions_masked[b, :len(b_mask)] = positions[b, b_mask]

        # Encode unmasked tokens
        encoded = self.encoder(x_unmasked)

        return {
            'encoded': encoded,
            'mask': mask,
            'ids_restore': ids_restore,
            'positions_unmasked': positions_unmasked,
            'positions_masked': positions_masked,
        }

    def forward_decoder(self, encoder_output: torch.Tensor,
                       mask: torch.Tensor,
                       positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder for reconstruction.

        Args:
            encoder_output: Encoded unmasked tokens, shape (B, N_unmasked, d_model)
            mask: Boolean mask, shape (B, N)
            positions: All positions, shape (B, N, 3)

        Returns:
            predictions: Predicted pixel values for masked positions
        """
        B, N = mask.shape

        # Get decoder position embeddings for all positions
        pos_embeddings = self.decoder_pos_encoder(positions.reshape(-1, 3)).reshape(B, N, -1)

        # Project to decoder dimension
        decoder_proj = self.decoder.encoder_to_decoder
        pos_embeddings_proj = decoder_proj(
            torch.zeros(B, N, self.d_model, device=encoder_output.device)
        ) + pos_embeddings[:, :, :self.decoder.decoder_dim]

        # Adjust position embeddings dimension if needed
        if pos_embeddings.shape[-1] != self.decoder.decoder_dim:
            # Create projection layer dynamically or just slice
            pos_embeddings = pos_embeddings[:, :, :self.decoder.decoder_dim]

        # Decode
        predictions = self.decoder(
            encoder_output,
            mask,
            pos_embeddings
        )

        return predictions

    def forward_auxiliary(self, encoded: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through auxiliary heads.

        Args:
            encoded: Encoded representations, shape (B, N, d_model)

        Returns:
            Dictionary with predictions from each auxiliary head
        """
        outputs = {}

        if self.use_auxiliary_heads:
            outputs['ghost_logits'] = self.ghost_head(encoded)
            outputs['ssnet_logits'] = self.ssnet_head(encoded)
            outputs['keypoint_scores'] = self.keypoint_head(encoded)

        return outputs

    def forward(self, input_wireplane_sparsetensors: List,
               query_v: List[torch.Tensor],
               positions: torch.Tensor,
               mask_ratio: float = None,
               return_encoder_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through MAE.

        Args:
            input_wireplane_sparsetensors: List of 3 SparseTensors for wireplane images
            query_v: List of 3 query coordinate tensors
            positions: 3D spacepoint positions, shape (B, N, 3)
            mask_ratio: Override default mask ratio
            return_encoder_features: If True, include encoder features in output

        Returns:
            Dictionary containing:
                - reconstruction: Predicted pixel values for masked positions
                - mask: Boolean mask
                - encoder_output: (optional) Encoded representations
                - auxiliary: (optional) Auxiliary head outputs
        """
        # Extract spacepoint features from images
        spacepoint_features, image_features = self.backbone(
            input_wireplane_sparsetensors, query_v
        )

        # Handle batch dimensions
        # Note: positions comes from collator with shape (B, N, 3)
        # spacepoint_features from backbone has shape (total_N, feature_dim)
        # We need to reshape spacepoint_features to match positions batch structure
        if spacepoint_features.dim() == 2 and positions.dim() == 3:
            # Reshape spacepoint_features to (B, N, feature_dim) to match positions
            B, N, _ = positions.shape
            spacepoint_features = spacepoint_features.view(B, N, -1)
        elif spacepoint_features.dim() == 2 and positions.dim() == 2:
            # Both need batch dimension
            spacepoint_features = spacepoint_features.unsqueeze(0)
            positions = positions.unsqueeze(0)

        # Encode
        encoder_results = self.forward_encoder(
            spacepoint_features, positions, mask_ratio
        )

        # Decode
        reconstruction = self.forward_decoder(
            encoder_results['encoded'],
            encoder_results['mask'],
            positions
        )

        outputs = {
            'reconstruction': reconstruction,
            'mask': encoder_results['mask'],
            'ids_restore': encoder_results['ids_restore'],
        }

        # Auxiliary predictions (on unmasked tokens)
        if self.use_auxiliary_heads:
            aux_outputs = self.forward_auxiliary(encoder_results['encoded'])
            outputs['auxiliary'] = aux_outputs

        if return_encoder_features:
            outputs['encoder_output'] = encoder_results['encoded']
            outputs['positions_unmasked'] = encoder_results['positions_unmasked']

        return outputs

    def get_encoder_output(self, input_wireplane_sparsetensors: List,
                          query_v: List[torch.Tensor],
                          positions: torch.Tensor) -> torch.Tensor:
        """
        Get encoder output without masking (for downstream tasks).

        Args:
            input_wireplane_sparsetensors: List of 3 SparseTensors
            query_v: List of 3 query coordinate tensors
            positions: 3D positions, shape (B, N, 3)

        Returns:
            Encoded representations, shape (B, N, d_model)
        """
        # Extract spacepoint features
        spacepoint_features, _ = self.backbone(
            input_wireplane_sparsetensors, query_v
        )

        # Handle batch dimensions (same logic as forward method)
        if spacepoint_features.dim() == 2 and positions.dim() == 3:
            B, N, _ = positions.shape
            spacepoint_features = spacepoint_features.view(B, N, -1)
        elif spacepoint_features.dim() == 2 and positions.dim() == 2:
            spacepoint_features = spacepoint_features.unsqueeze(0)
            positions = positions.unsqueeze(0)

        B, N, _ = spacepoint_features.shape

        # Project and add position encoding
        x = self.input_proj(spacepoint_features)
        pos_enc = self.pos_encoder(positions.reshape(-1, 3)).reshape(B, N, -1)
        x = x + pos_enc

        # Encode all tokens (no masking)
        encoded = self.encoder(x)

        return encoded


class SpacepointMAEForPretraining(SpacepointMAE):
    """
    SpacepointMAE with additional utilities for pretraining.

    Includes:
    - EMA teacher for co-distillation
    - Projection head for contrastive learning
    """

    def __init__(self, *args, use_ema_teacher: bool = False,
                 ema_decay: float = 0.999,
                 contrastive_dim: int = 128, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay

        # Projection head for contrastive learning
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, contrastive_dim)
        )

        # EMA teacher model
        if use_ema_teacher:
            self.teacher = None  # Initialized lazily

    @torch.no_grad()
    def _init_teacher(self):
        """Initialize teacher as copy of student."""
        import copy
        self.teacher = copy.deepcopy(self)
        for param in self.teacher.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher with EMA of student weights."""
        if self.teacher is None:
            self._init_teacher()
            return

        for student_param, teacher_param in zip(
            self.parameters(), self.teacher.parameters()
        ):
            teacher_param.data = (
                self.ema_decay * teacher_param.data +
                (1 - self.ema_decay) * student_param.data
            )

    def get_contrastive_features(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Get features for contrastive learning.

        Args:
            encoded: Encoded representations, shape (B, N, d_model)

        Returns:
            Projected features, shape (B, N, contrastive_dim)
        """
        return F.normalize(self.contrastive_proj(encoded), dim=-1)
