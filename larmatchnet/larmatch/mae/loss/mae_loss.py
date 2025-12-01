"""
Combined MAE Loss

Aggregates all loss components for MAE training with configurable weights.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .reconstruction_loss import ReconstructionLoss, MaskedReconstructionLoss
from .contrastive_loss import ContrastiveLoss, SupervisedContrastiveLoss, ParticlePoolingContrastiveLoss
from .auxiliary_losses import GhostClassificationLoss, SSNetLoss, KeypointLoss
from .distillation_loss import DistillationLoss


class MAELoss(nn.Module):
    """
    Combined loss for MAE pretraining.

    Combines:
    - Reconstruction loss (primary MAE objective)
    - Contrastive loss (same-particle similarity)
    - Auxiliary supervised losses (ghost, SSNet, keypoint)
    - Distillation loss (optional EMA teacher)

    Args:
        use_learnable_weights: If True, learn loss weights during training
        reconstruction_weight: Weight for reconstruction loss
        contrastive_weight: Weight for contrastive loss
        ghost_weight: Weight for ghost classification
        ssnet_weight: Weight for SSNet classification
        keypoint_weight: Weight for keypoint regression
        distillation_weight: Weight for distillation loss
        config: Optional config dict with all parameters
    """

    def __init__(
        self,
        use_learnable_weights: bool = False,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        ghost_weight: float = 0.5,
        ssnet_weight: float = 0.5,
        keypoint_weight: float = 0.3,
        distillation_weight: float = 0.1,
        reconstruction_type: str = 'mse',
        contrastive_type: str = 'supervised',
        num_ssnet_classes: int = 5,
        num_keypoint_types: int = 6,
        config: dict = None
    ):
        super().__init__()

        # Override with config if provided
        if config is not None:
            reconstruction_weight = config.get('RECONSTRUCTION_WEIGHT', reconstruction_weight)
            contrastive_weight = config.get('CONTRASTIVE_WEIGHT', contrastive_weight)
            ghost_weight = config.get('GHOST_WEIGHT', ghost_weight)
            ssnet_weight = config.get('SSNET_WEIGHT', ssnet_weight)
            keypoint_weight = config.get('KEYPOINT_WEIGHT', keypoint_weight)
            distillation_weight = config.get('DISTILLATION_WEIGHT', distillation_weight)
            reconstruction_type = config.get('RECONSTRUCTION_TYPE', reconstruction_type)
            contrastive_type = config.get('CONTRASTIVE_TYPE', contrastive_type)
            num_ssnet_classes = config.get('NUM_SSNET_CLASSES', num_ssnet_classes)
            num_keypoint_types = config.get('NUM_KEYPOINT_TYPES', num_keypoint_types)
            use_learnable_weights = config.get('USE_LEARNABLE_WEIGHTS', use_learnable_weights)

        self.use_learnable_weights = use_learnable_weights

        # Initialize loss modules
        self.reconstruction_loss = MaskedReconstructionLoss(
            loss_type=reconstruction_type,
            normalize_target=True
        )

        if contrastive_type == 'supervised':
            self.contrastive_loss = SupervisedContrastiveLoss()
        elif contrastive_type == 'particle_pooling':
            self.contrastive_loss = ParticlePoolingContrastiveLoss()
        else:
            self.contrastive_loss = ContrastiveLoss()

        self.ghost_loss = GhostClassificationLoss()
        self.ssnet_loss = SSNetLoss(num_classes=num_ssnet_classes)
        self.keypoint_loss = KeypointLoss(num_keypoint_types=num_keypoint_types)
        self.distillation_loss = DistillationLoss()

        # Loss weights
        if use_learnable_weights:
            # Learnable log-variance weights (Kendall et al., "Multi-Task Learning")
            self.log_var_recon = nn.Parameter(torch.zeros(1))
            self.log_var_contrast = nn.Parameter(torch.zeros(1))
            self.log_var_ghost = nn.Parameter(torch.zeros(1))
            self.log_var_ssnet = nn.Parameter(torch.zeros(1))
            self.log_var_kp = nn.Parameter(torch.zeros(1))
            self.log_var_distill = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('w_recon', torch.tensor(reconstruction_weight))
            self.register_buffer('w_contrast', torch.tensor(contrastive_weight))
            self.register_buffer('w_ghost', torch.tensor(ghost_weight))
            self.register_buffer('w_ssnet', torch.tensor(ssnet_weight))
            self.register_buffer('w_kp', torch.tensor(keypoint_weight))
            self.register_buffer('w_distill', torch.tensor(distillation_weight))

    def _get_weight(self, log_var: torch.Tensor, fixed_weight: torch.Tensor) -> torch.Tensor:
        """Get loss weight, either learned or fixed."""
        if self.use_learnable_weights:
            # Weight = 1 / (2 * sigma^2), regularized by log(sigma)
            return torch.exp(-log_var)
        else:
            return fixed_weight

    def _get_reg(self, log_var: torch.Tensor) -> torch.Tensor:
        """Get regularization term for learnable weights."""
        if self.use_learnable_weights:
            return 0.5 * log_var
        else:
            return torch.tensor(0.0, device=log_var.device if hasattr(log_var, 'device') else 'cpu')

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        encoder_features: torch.Tensor = None,
        instance_labels: torch.Tensor = None,
        is_true: torch.Tensor = None,
        auxiliary_outputs: Dict[str, torch.Tensor] = None,
        ssnet_labels: torch.Tensor = None,
        keypoint_targets: torch.Tensor = None,
        teacher_features: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined MAE loss.

        Args:
            predictions: Reconstruction predictions, shape (B, N, 3)
            targets: Reconstruction targets (pixel values), shape (B, N, 3)
            mask: Boolean mask for masked positions, shape (B, N)
            encoder_features: Encoded representations for contrastive loss
            instance_labels: Particle instance labels
            is_true: Boolean for ghost vs true
            auxiliary_outputs: Dict with 'ghost_logits', 'ssnet_logits', 'keypoint_scores'
            ssnet_labels: SSNet class labels
            keypoint_targets: Keypoint score targets
            teacher_features: Teacher model features for distillation

        Returns:
            Dictionary with total loss and individual loss components
        """
        device = predictions.device
        losses = {}

        # Reconstruction loss (primary)
        recon_loss = self.reconstruction_loss(predictions, targets, mask, is_true)
        w_recon = self._get_weight(
            self.log_var_recon if self.use_learnable_weights else None,
            self.w_recon
        )
        losses['reconstruction'] = recon_loss
        total_loss = w_recon * recon_loss
        if self.use_learnable_weights:
            total_loss = total_loss + self._get_reg(self.log_var_recon)

        # For contrastive and auxiliary losses, we need labels for unmasked positions only
        # since encoder_features only contains unmasked tokens
        # mask is True for masked positions, so ~mask gives unmasked positions
        unmasked_mask = ~mask  # True for unmasked positions

        # Extract labels for unmasked positions
        if instance_labels is not None:
            # Flatten and extract unmasked labels
            B, N_total = mask.shape
            N_unmasked = encoder_features.shape[1] if encoder_features is not None else unmasked_mask.sum(dim=1).max().item()
            unmasked_instance_labels = torch.zeros(B, N_unmasked, dtype=instance_labels.dtype, device=device)
            for b in range(B):
                unmasked_idx = unmasked_mask[b].nonzero(as_tuple=True)[0]
                n_unmasked = len(unmasked_idx)
                unmasked_instance_labels[b, :n_unmasked] = instance_labels[b, unmasked_idx]
        else:
            unmasked_instance_labels = None

        if is_true is not None:
            B, N_total = mask.shape
            N_unmasked = encoder_features.shape[1] if encoder_features is not None else unmasked_mask.sum(dim=1).max().item()
            unmasked_is_true = torch.zeros(B, N_unmasked, dtype=is_true.dtype, device=device)
            for b in range(B):
                unmasked_idx = unmasked_mask[b].nonzero(as_tuple=True)[0]
                n_unmasked = len(unmasked_idx)
                unmasked_is_true[b, :n_unmasked] = is_true[b, unmasked_idx]
        else:
            unmasked_is_true = None

        # Contrastive loss
        if encoder_features is not None and unmasked_instance_labels is not None:
            contrast_loss = self.contrastive_loss(
                encoder_features, unmasked_instance_labels, unmasked_is_true
            )
            w_contrast = self._get_weight(
                self.log_var_contrast if self.use_learnable_weights else None,
                self.w_contrast
            )
            losses['contrastive'] = contrast_loss
            total_loss = total_loss + w_contrast * contrast_loss
            if self.use_learnable_weights:
                total_loss = total_loss + self._get_reg(self.log_var_contrast)

        # Auxiliary losses
        if auxiliary_outputs is not None:
            # Ghost classification
            if 'ghost_logits' in auxiliary_outputs and unmasked_is_true is not None:
                ghost_loss = self.ghost_loss(
                    auxiliary_outputs['ghost_logits'],
                    unmasked_is_true.long()
                )
                w_ghost = self._get_weight(
                    self.log_var_ghost if self.use_learnable_weights else None,
                    self.w_ghost
                )
                losses['ghost'] = ghost_loss
                total_loss = total_loss + w_ghost * ghost_loss
                if self.use_learnable_weights:
                    total_loss = total_loss + self._get_reg(self.log_var_ghost)

            # SSNet classification
            if 'ssnet_logits' in auxiliary_outputs and ssnet_labels is not None:
                # Extract unmasked ssnet labels
                B, N_total = mask.shape
                N_unmasked = auxiliary_outputs['ssnet_logits'].shape[1]
                unmasked_ssnet_labels = torch.zeros(B, N_unmasked, dtype=ssnet_labels.dtype, device=device)
                for b in range(B):
                    unmasked_idx = unmasked_mask[b].nonzero(as_tuple=True)[0]
                    n_unmasked = len(unmasked_idx)
                    unmasked_ssnet_labels[b, :n_unmasked] = ssnet_labels[b, unmasked_idx]

                ssnet_loss = self.ssnet_loss(
                    auxiliary_outputs['ssnet_logits'],
                    unmasked_ssnet_labels
                )
                w_ssnet = self._get_weight(
                    self.log_var_ssnet if self.use_learnable_weights else None,
                    self.w_ssnet
                )
                losses['ssnet'] = ssnet_loss
                total_loss = total_loss + w_ssnet * ssnet_loss
                if self.use_learnable_weights:
                    total_loss = total_loss + self._get_reg(self.log_var_ssnet)

            # Keypoint regression
            if 'keypoint_scores' in auxiliary_outputs and keypoint_targets is not None:
                # Extract unmasked keypoint targets
                B, N_total = mask.shape
                N_unmasked = auxiliary_outputs['keypoint_scores'].shape[1]
                n_kp_types = keypoint_targets.shape[-1]
                unmasked_keypoint_targets = torch.zeros(B, N_unmasked, n_kp_types, dtype=keypoint_targets.dtype, device=device)
                for b in range(B):
                    unmasked_idx = unmasked_mask[b].nonzero(as_tuple=True)[0]
                    n_unmasked = len(unmasked_idx)
                    unmasked_keypoint_targets[b, :n_unmasked] = keypoint_targets[b, unmasked_idx]

                kp_loss = self.keypoint_loss(
                    auxiliary_outputs['keypoint_scores'],
                    unmasked_keypoint_targets
                )
                w_kp = self._get_weight(
                    self.log_var_kp if self.use_learnable_weights else None,
                    self.w_kp
                )
                losses['keypoint'] = kp_loss
                total_loss = total_loss + w_kp * kp_loss
                if self.use_learnable_weights:
                    total_loss = total_loss + self._get_reg(self.log_var_kp)

        # Distillation loss
        if encoder_features is not None and teacher_features is not None:
            distill_loss = self.distillation_loss(
                encoder_features, teacher_features
            )
            w_distill = self._get_weight(
                self.log_var_distill if self.use_learnable_weights else None,
                self.w_distill
            )
            losses['distillation'] = distill_loss
            total_loss = total_loss + w_distill * distill_loss
            if self.use_learnable_weights:
                total_loss = total_loss + self._get_reg(self.log_var_distill)

        losses['total'] = total_loss

        # Add weight information for logging
        if self.use_learnable_weights:
            losses['weight_recon'] = torch.exp(-self.log_var_recon)
            losses['weight_contrast'] = torch.exp(-self.log_var_contrast)
            losses['weight_ghost'] = torch.exp(-self.log_var_ghost)
            losses['weight_ssnet'] = torch.exp(-self.log_var_ssnet)
            losses['weight_kp'] = torch.exp(-self.log_var_kp)

        return losses


def create_mae_loss(config: dict) -> MAELoss:
    """
    Factory function to create MAE loss from config.

    Args:
        config: Configuration dictionary

    Returns:
        MAELoss instance
    """
    return MAELoss(config=config)
