"""
Distillation Loss for MAE

Implements co-distillation and knowledge distillation losses
using an EMA (Exponential Moving Average) teacher model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss using EMA teacher.

    Encourages the student model to produce similar representations
    to the EMA teacher model.

    Args:
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss relative to task loss
        loss_type: Type of distillation ('kl', 'mse', 'cosine')
    """

    def __init__(self, temperature: float = 4.0,
                 alpha: float = 0.5,
                 loss_type: str = 'cosine'):
        super().__init__()

        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type

    def forward(self, student_features: torch.Tensor,
               teacher_features: torch.Tensor,
               mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_features: Student model outputs, shape (B, N, D)
            teacher_features: Teacher model outputs, shape (B, N, D)
            mask: Optional mask for valid positions

        Returns:
            Scalar loss
        """
        if mask is not None:
            student_features = student_features[mask]
            teacher_features = teacher_features[mask]

        if student_features.numel() == 0:
            return torch.tensor(0.0, device=student_features.device, requires_grad=True)

        if self.loss_type == 'kl':
            # KL divergence between softened distributions
            student_log_probs = F.log_softmax(student_features / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_features / self.temperature, dim=-1)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            loss = loss * (self.temperature ** 2)  # Scale by T^2

        elif self.loss_type == 'mse':
            # MSE between features
            loss = F.mse_loss(student_features, teacher_features)

        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            student_norm = F.normalize(student_features, dim=-1)
            teacher_norm = F.normalize(teacher_features, dim=-1)
            similarity = (student_norm * teacher_norm).sum(dim=-1)
            loss = (1 - similarity).mean()

        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(student_features, teacher_features)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return self.alpha * loss


class CoDistillationLoss(nn.Module):
    """
    Co-distillation loss for mutual learning.

    Both models learn from each other rather than having a fixed teacher.
    Uses stop-gradient on one model's output.

    Args:
        temperature: Temperature for distributions
        symmetric: If True, both models serve as teacher for each other
    """

    def __init__(self, temperature: float = 4.0,
                 symmetric: bool = True):
        super().__init__()

        self.temperature = temperature
        self.symmetric = symmetric
        self.distill_loss = DistillationLoss(temperature, alpha=1.0, loss_type='cosine')

    def forward(self, features_1: torch.Tensor,
               features_2: torch.Tensor,
               mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute co-distillation loss.

        Args:
            features_1: First model's features
            features_2: Second model's features
            mask: Optional mask

        Returns:
            Scalar loss
        """
        # Model 1 learns from model 2 (stopped gradient)
        loss_1 = self.distill_loss(features_1, features_2.detach(), mask)

        if self.symmetric:
            # Model 2 learns from model 1
            loss_2 = self.distill_loss(features_2, features_1.detach(), mask)
            return (loss_1 + loss_2) / 2
        else:
            return loss_1


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss across augmented views.

    Encourages consistent representations across different augmentations
    or views of the same data.

    Args:
        loss_type: Type of matching ('mse', 'cosine', 'contrastive')
        temperature: Temperature for contrastive variant
    """

    def __init__(self, loss_type: str = 'cosine',
                 temperature: float = 0.1):
        super().__init__()

        self.loss_type = loss_type
        self.temperature = temperature

    def forward(self, features_view1: torch.Tensor,
               features_view2: torch.Tensor,
               alignment: torch.Tensor = None) -> torch.Tensor:
        """
        Compute feature matching loss between two views.

        Args:
            features_view1: Features from view 1, shape (B, N, D)
            features_view2: Features from view 2, shape (B, N, D) or (B, M, D)
            alignment: If views have different sizes, alignment indices

        Returns:
            Scalar loss
        """
        if alignment is not None:
            # Align features based on provided indices
            features_view2 = features_view2[:, alignment]

        if self.loss_type == 'mse':
            loss = F.mse_loss(features_view1, features_view2)

        elif self.loss_type == 'cosine':
            feat1_norm = F.normalize(features_view1, dim=-1)
            feat2_norm = F.normalize(features_view2, dim=-1)
            similarity = (feat1_norm * feat2_norm).sum(dim=-1)
            loss = (1 - similarity).mean()

        elif self.loss_type == 'contrastive':
            # Each point in view1 should match corresponding point in view2
            B, N, D = features_view1.shape
            feat1_norm = F.normalize(features_view1, dim=-1)
            feat2_norm = F.normalize(features_view2, dim=-1)

            # Flatten batch
            feat1_flat = feat1_norm.reshape(-1, D)
            feat2_flat = feat2_norm.reshape(-1, D)

            # Similarity matrix
            sim = torch.mm(feat1_flat, feat2_flat.t()) / self.temperature

            # Positive pairs are on diagonal
            labels = torch.arange(sim.shape[0], device=sim.device)
            loss = F.cross_entropy(sim, labels)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class EMATeacher:
    """
    EMA Teacher model wrapper.

    Maintains an exponential moving average of the student model weights.

    Args:
        student_model: The student model to track
        ema_decay: EMA decay rate (higher = slower update)
    """

    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999):
        import copy

        self.ema_decay = ema_decay
        self.teacher_model = copy.deepcopy(student_model)

        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """
        Update teacher weights with EMA of student.

        Args:
            student_model: Current student model
        """
        for student_param, teacher_param in zip(
            student_model.parameters(),
            self.teacher_model.parameters()
        ):
            teacher_param.data = (
                self.ema_decay * teacher_param.data +
                (1 - self.ema_decay) * student_param.data
            )

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Forward pass through teacher model."""
        return self.teacher_model(*args, **kwargs)

    def state_dict(self):
        """Get teacher state dict for checkpointing."""
        return self.teacher_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load teacher state dict."""
        self.teacher_model.load_state_dict(state_dict)
