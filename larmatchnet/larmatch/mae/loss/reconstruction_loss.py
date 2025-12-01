"""
Reconstruction Loss for MAE

Loss functions for reconstructing masked pixel values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for masked auto-encoding.

    Supports MSE, L1, and smooth L1 loss functions.
    Can optionally weight loss by ghost/true status.

    Args:
        loss_type: Type of loss ('mse', 'l1', 'smooth_l1', 'huber')
        reduction: Reduction method ('mean', 'sum', 'none')
        normalize_target: Whether to normalize target values
        per_plane_loss: Whether to compute loss per plane separately
    """

    def __init__(self, loss_type: str = 'mse',
                 reduction: str = 'mean',
                 normalize_target: bool = True,
                 per_plane_loss: bool = False):
        super().__init__()

        self.loss_type = loss_type
        self.reduction = reduction
        self.normalize_target = normalize_target
        self.per_plane_loss = per_plane_loss

        # Target normalization parameters (will be updated during training)
        self.register_buffer('target_mean', torch.zeros(3))
        self.register_buffer('target_std', torch.ones(3))

    def forward(self, predictions: torch.Tensor,
               targets: torch.Tensor,
               mask: torch.Tensor = None,
               is_true: torch.Tensor = None,
               true_weight: float = 1.0) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            predictions: Predicted values, shape (B, N_masked, 3) or (B, N, 3)
            targets: Target pixel values, shape (B, N_masked, 3) or (B, N, 3)
            mask: Boolean mask indicating which positions are masked
            is_true: Boolean indicating true (non-ghost) spacepoints
            true_weight: Weight multiplier for true spacepoint losses

        Returns:
            Scalar loss value
        """
        # Normalize targets if requested
        if self.normalize_target:
            # Ensure normalization params are on the same device as targets
            target_mean = self.target_mean.to(targets.device)
            target_std = self.target_std.to(targets.device)
            targets = (targets - target_mean) / (target_std + 1e-6)

        # Compute base loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(predictions, targets, reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Shape: (B, N, 3)

        # Apply weighting for true vs ghost
        if is_true is not None and true_weight != 1.0:
            weights = torch.ones_like(loss)
            is_true_expanded = is_true.unsqueeze(-1).expand_as(loss)
            weights[is_true_expanded] = true_weight
            loss = loss * weights

        # Reduce per-plane if requested
        if self.per_plane_loss:
            # Return loss per plane
            if self.reduction == 'mean':
                loss = loss.mean(dim=(0, 1))  # (3,)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=(0, 1))
            return loss

        # Standard reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def update_normalization(self, targets: torch.Tensor):
        """
        Update target normalization statistics.

        Args:
            targets: Batch of target values, shape (B, N, 3)
        """
        with torch.no_grad():
            # Compute statistics
            mean = targets.mean(dim=(0, 1))
            std = targets.std(dim=(0, 1))

            # Exponential moving average update
            alpha = 0.99
            self.target_mean = alpha * self.target_mean + (1 - alpha) * mean
            self.target_std = alpha * self.target_std + (1 - alpha) * std


class EnergyWeightedReconstructionLoss(nn.Module):
    """
    Reconstruction loss weighted by deposited energy.

    Higher energy depositions are more important for physics,
    so they receive higher weight in the loss.

    Args:
        loss_type: Base loss type
        energy_power: Power to raise energy weights to (higher = more emphasis on high energy)
        min_weight: Minimum weight to avoid zero weights
    """

    def __init__(self, loss_type: str = 'mse',
                 energy_power: float = 0.5,
                 min_weight: float = 0.1):
        super().__init__()

        self.base_loss = ReconstructionLoss(loss_type, reduction='none')
        self.energy_power = energy_power
        self.min_weight = min_weight

    def forward(self, predictions: torch.Tensor,
               targets: torch.Tensor,
               edep: torch.Tensor = None) -> torch.Tensor:
        """
        Compute energy-weighted reconstruction loss.

        Args:
            predictions: Predicted values
            targets: Target values
            edep: Energy deposition values, shape (B, N, 3) or (B, N)

        Returns:
            Scalar loss
        """
        # Compute unweighted loss
        loss = self.base_loss(predictions, targets)  # (B, N, 3)

        if edep is not None:
            # Compute energy weights
            if edep.dim() == 2:
                edep = edep.unsqueeze(-1).expand_as(loss)

            # Normalize and apply power
            edep_norm = edep / (edep.max() + 1e-6)
            weights = torch.pow(edep_norm + self.min_weight, self.energy_power)

            # Normalize weights
            weights = weights / weights.mean()

            loss = loss * weights

        return loss.mean()


class MaskedReconstructionLoss(nn.Module):
    """
    Reconstruction loss that only applies to masked positions.

    This wraps a base loss and handles the masking logic.

    Args:
        base_loss: Base reconstruction loss module
    """

    def __init__(self, loss_type: str = 'mse',
                 normalize_target: bool = True):
        super().__init__()
        self.base_loss = ReconstructionLoss(
            loss_type=loss_type,
            reduction='none',
            normalize_target=normalize_target
        )

    def forward(self, predictions: torch.Tensor,
               targets: torch.Tensor,
               mask: torch.Tensor,
               is_true: torch.Tensor = None) -> torch.Tensor:
        """
        Compute reconstruction loss on masked positions only.

        Args:
            predictions: Predictions, shape (B, N_masked, 3) if already masked,
                        or (B, N, 3) if full predictions
            targets: Full targets, shape (B, N, 3)
            mask: Boolean mask, True for masked positions, shape (B, N)
            is_true: True for non-ghost spacepoints, shape (B, N)

        Returns:
            Scalar loss
        """
        B = predictions.shape[0]
        N_total = mask.shape[1]
        N_pred = predictions.shape[1]
        D = predictions.shape[2]

        # Check if predictions are already for masked positions only
        N_masked = mask.sum(dim=1).max().item()

        if N_pred == N_masked and N_pred != N_total:
            # Predictions are already for masked positions only (shape: B, N_masked, D)
            # Extract masked targets to match
            masked_pred = predictions
            masked_target = torch.zeros(B, N_masked, D, device=targets.device, dtype=targets.dtype)
            masked_is_true = None
            if is_true is not None:
                masked_is_true = torch.zeros(B, N_masked, device=is_true.device, dtype=is_true.dtype)

            for b in range(B):
                masked_idx = mask[b].nonzero(as_tuple=True)[0]
                n_masked_b = len(masked_idx)
                masked_target[b, :n_masked_b] = targets[b, masked_idx]
                if is_true is not None:
                    masked_is_true[b, :n_masked_b] = is_true[b, masked_idx]

            if masked_pred.numel() == 0:
                return torch.tensor(0.0, device=predictions.device)

            # Compute loss on masked positions
            loss = self.base_loss(masked_pred, masked_target)

            # Weight by true/ghost if provided
            if masked_is_true is not None:
                weights = torch.ones_like(loss)
                weights[masked_is_true.unsqueeze(-1).expand_as(weights)] = 2.0
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()

        else:
            # Predictions are full, extract masked positions
            masked_pred = predictions[mask]  # (N_masked_total, D)
            masked_target = targets[mask]  # (N_masked_total, D)

            if masked_pred.numel() == 0:
                return torch.tensor(0.0, device=predictions.device)

            # Compute loss on masked positions
            loss = self.base_loss(
                masked_pred.unsqueeze(0),
                masked_target.unsqueeze(0)
            )

            # Weight by true/ghost if provided
            if is_true is not None:
                masked_is_true = is_true[mask]
                # Weight true spacepoints higher
                weights = torch.ones_like(loss)
                weights[masked_is_true.unsqueeze(-1).expand_as(weights)] = 2.0
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()

        return loss
