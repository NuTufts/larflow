"""
Auxiliary Supervised Losses for MAE

Additional supervised losses using available labels:
- Ghost classification (ghost vs true spacepoint)
- SSNet classification (particle type)
- Keypoint score regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GhostClassificationLoss(nn.Module):
    """
    Binary classification loss for ghost vs true spacepoints.

    Uses focal loss to handle class imbalance (many more ghosts than true).

    Args:
        gamma: Focal loss gamma parameter
        alpha: Weight for positive (true) class
        reduction: Loss reduction method
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: float = 0.75,
                 reduction: str = 'mean'):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
               targets: torch.Tensor,
               mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute ghost classification loss.

        Args:
            logits: Predicted logits, shape (B, N, 2) or (N, 2)
            targets: Binary labels (0=ghost, 1=true), shape (B, N) or (N,)
            mask: Optional mask for valid positions

        Returns:
            Scalar loss
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, C = logits.shape

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1).long()

        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]

        if len(targets_flat) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Focal loss
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weighting
        alpha_weight = torch.where(
            targets_flat == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device)
        )

        loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SSNetLoss(nn.Module):
    """
    Multi-class classification loss for particle type (SSNet labels).

    Classes typically: background, shower, track, michel, delta

    Args:
        num_classes: Number of particle classes
        class_weights: Optional per-class weights
        ignore_index: Label to ignore (e.g., unlabeled points)
        use_focal: Whether to use focal loss
        gamma: Focal loss gamma
    """

    def __init__(self, num_classes: int = 5,
                 class_weights: torch.Tensor = None,
                 ignore_index: int = -1,
                 use_focal: bool = True,
                 gamma: float = 2.0):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_focal = use_focal
        self.gamma = gamma

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor,
               targets: torch.Tensor,
               mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute SSNet classification loss.

        Args:
            logits: Predicted logits, shape (B, N, num_classes) or (N, num_classes)
            targets: Class labels, shape (B, N) or (N,)
            mask: Optional mask for valid positions

        Returns:
            Scalar loss
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, C = logits.shape

        # Flatten
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1).long()

        # Apply mask and ignore index
        valid = targets_flat != self.ignore_index
        if mask is not None:
            valid = valid & mask.reshape(-1)

        logits_flat = logits_flat[valid]
        targets_flat = targets_flat[valid]

        if len(targets_flat) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute cross entropy
        if self.use_focal:
            ce_loss = F.cross_entropy(
                logits_flat, targets_flat,
                weight=self.class_weights,
                reduction='none'
            )
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
            loss = (focal_weight * ce_loss).mean()
        else:
            loss = F.cross_entropy(
                logits_flat, targets_flat,
                weight=self.class_weights
            )

        return loss


class KeypointLoss(nn.Module):
    """
    Regression loss for keypoint proximity scores.

    Predicts how close each spacepoint is to various keypoint types
    (e.g., vertex, track end, shower start).

    Args:
        num_keypoint_types: Number of keypoint types
        loss_type: Loss type ('mse', 'l1', 'smooth_l1')
        use_weighting: Weight loss by keypoint score magnitude
    """

    def __init__(self, num_keypoint_types: int = 6,
                 loss_type: str = 'mse',
                 use_weighting: bool = True):
        super().__init__()

        self.num_keypoint_types = num_keypoint_types
        self.loss_type = loss_type
        self.use_weighting = use_weighting

    def forward(self, predictions: torch.Tensor,
               targets: torch.Tensor,
               mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute keypoint regression loss.

        Args:
            predictions: Predicted scores, shape (B, N, K) or (N, K)
            targets: Target scores, shape (B, N, K) or (N, K)
            mask: Optional mask for valid positions

        Returns:
            Scalar loss
        """
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, K = predictions.shape

        # Apply mask
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        else:
            predictions = predictions.reshape(-1, K)
            targets = targets.reshape(-1, K)

        if predictions.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Compute base loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(predictions, targets, reduction='none')
        else:
            loss = F.mse_loss(predictions, targets, reduction='none')

        # Weight by target magnitude (focus on near-keypoint regions)
        if self.use_weighting:
            weights = 1.0 + targets  # Higher weight for points near keypoints
            loss = loss * weights

        return loss.mean()


class InstanceSegmentationLoss(nn.Module):
    """
    Loss for instance segmentation using embedding-based approach.

    Encourages spacepoints from same particle to have similar embeddings
    and different particles to have different embeddings.

    Args:
        delta_v: Margin for variance (pull) loss
        delta_d: Margin for distance (push) loss
        alpha: Weight for variance loss
        beta: Weight for distance loss
        gamma: Weight for regularization loss
    """

    def __init__(self, delta_v: float = 0.5,
                 delta_d: float = 1.5,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 gamma: float = 0.001):
        super().__init__()

        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, embeddings: torch.Tensor,
               instance_labels: torch.Tensor,
               is_true: torch.Tensor = None) -> torch.Tensor:
        """
        Compute instance segmentation loss.

        Args:
            embeddings: Point embeddings, shape (B, N, D) or (N, D)
            instance_labels: Instance labels, shape (B, N) or (N,)
            is_true: Optional mask for true spacepoints

        Returns:
            Scalar loss
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            instance_labels = instance_labels.unsqueeze(0)
            if is_true is not None:
                is_true = is_true.unsqueeze(0)

        B = embeddings.shape[0]
        device = embeddings.device

        total_var_loss = 0.0
        total_dist_loss = 0.0
        total_reg_loss = 0.0
        n_valid = 0

        for b in range(B):
            emb = embeddings[b]
            labels = instance_labels[b]

            # Filter to true if provided
            if is_true is not None:
                valid = is_true[b]
                emb = emb[valid]
                labels = labels[valid]

            # Get unique instances
            unique_labels = torch.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]
            n_instances = len(unique_labels)

            if n_instances < 2:
                continue

            # Compute cluster means
            cluster_means = []
            for label in unique_labels:
                mask = labels == label
                mean = emb[mask].mean(dim=0)
                cluster_means.append(mean)

            cluster_means = torch.stack(cluster_means)  # (C, D)

            # Variance loss (pull): points close to their cluster mean
            var_loss = 0.0
            for i, label in enumerate(unique_labels):
                mask = labels == label
                cluster_emb = emb[mask]
                distances = torch.norm(cluster_emb - cluster_means[i], dim=1)
                var_loss += torch.clamp(distances - self.delta_v, min=0).pow(2).mean()
            var_loss /= n_instances

            # Distance loss (push): cluster means far apart
            dist_loss = 0.0
            n_pairs = 0
            for i in range(n_instances):
                for j in range(i + 1, n_instances):
                    dist = torch.norm(cluster_means[i] - cluster_means[j])
                    dist_loss += torch.clamp(2 * self.delta_d - dist, min=0).pow(2)
                    n_pairs += 1
            if n_pairs > 0:
                dist_loss /= n_pairs

            # Regularization loss
            reg_loss = cluster_means.norm(dim=1).mean()

            total_var_loss += var_loss
            total_dist_loss += dist_loss
            total_reg_loss += reg_loss
            n_valid += 1

        if n_valid > 0:
            loss = (
                self.alpha * total_var_loss / n_valid +
                self.beta * total_dist_loss / n_valid +
                self.gamma * total_reg_loss / n_valid
            )
            return loss
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
