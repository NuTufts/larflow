"""
Contrastive Loss for MAE

Implements contrastive learning losses to encourage similar representations
for spacepoints from the same particle and different representations for
spacepoints from different particles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) style contrastive loss.

    For each anchor spacepoint, treats other spacepoints from the same particle
    as positives and spacepoints from different particles as negatives.

    Args:
        temperature: Temperature parameter for softmax scaling
        use_hard_negatives: Whether to use hard negative mining
        max_positives: Maximum positive pairs per anchor
        max_negatives: Maximum negative pairs per anchor
    """

    def __init__(self, temperature: float = 0.1,
                 use_hard_negatives: bool = False,
                 max_positives: int = 10,
                 max_negatives: int = 100):
        super().__init__()

        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.max_positives = max_positives
        self.max_negatives = max_negatives

    def forward(self, features: torch.Tensor,
               instance_labels: torch.Tensor,
               is_true: torch.Tensor = None) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            features: Normalized feature vectors, shape (B, N, D) or (N, D)
            instance_labels: Particle instance labels, shape (B, N) or (N,)
            is_true: Boolean mask for true spacepoints (optional)

        Returns:
            Scalar loss value
        """
        # Handle batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)
            instance_labels = instance_labels.unsqueeze(0)
            if is_true is not None:
                is_true = is_true.unsqueeze(0)

        B, N, D = features.shape
        device = features.device

        total_loss = 0.0
        n_valid = 0

        for b in range(B):
            feat = features[b]  # (N, D)
            labels = instance_labels[b]  # (N,)

            # Filter to true spacepoints if provided
            if is_true is not None:
                valid_mask = is_true[b] & (labels > 0)
            else:
                valid_mask = labels > 0

            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if len(valid_indices) < 2:
                continue

            # Sample anchors if too many
            if len(valid_indices) > 1000:
                perm = torch.randperm(len(valid_indices), device=device)[:1000]
                valid_indices = valid_indices[perm]

            valid_feat = feat[valid_indices]  # (M, D)
            valid_labels = labels[valid_indices]  # (M,)

            # Compute all pairwise similarities
            similarity = torch.mm(valid_feat, valid_feat.t()) / self.temperature  # (M, M)

            # Create positive mask (same particle)
            positive_mask = valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)
            positive_mask.fill_diagonal_(False)  # Exclude self

            # For each anchor, compute loss
            M = len(valid_indices)
            for i in range(min(M, 100)):  # Limit anchors for efficiency
                pos_mask_i = positive_mask[i]
                n_pos = pos_mask_i.sum().item()

                if n_pos == 0:
                    continue

                # Get positive and negative similarities
                pos_sim = similarity[i][pos_mask_i]
                neg_sim = similarity[i][~pos_mask_i & (torch.arange(M, device=device) != i)]

                if len(neg_sim) == 0:
                    continue

                # Limit pairs
                if len(pos_sim) > self.max_positives:
                    pos_idx = torch.randperm(len(pos_sim))[:self.max_positives]
                    pos_sim = pos_sim[pos_idx]

                if len(neg_sim) > self.max_negatives:
                    if self.use_hard_negatives:
                        # Take hardest negatives (highest similarity)
                        neg_idx = torch.argsort(neg_sim, descending=True)[:self.max_negatives]
                    else:
                        neg_idx = torch.randperm(len(neg_sim))[:self.max_negatives]
                    neg_sim = neg_sim[neg_idx]

                # NT-Xent loss for this anchor
                # For each positive, it should be higher than all negatives
                for pos_s in pos_sim:
                    logits = torch.cat([pos_s.unsqueeze(0), neg_sim])
                    target = torch.zeros(1, dtype=torch.long, device=device)
                    loss_i = F.cross_entropy(logits.unsqueeze(0), target)
                    total_loss += loss_i
                    n_valid += 1

        if n_valid > 0:
            return total_loss / n_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).

    Uses particle instance labels as supervision for contrastive learning.
    More efficient than NT-Xent for many positive pairs.

    Reference: Khosla et al., "Supervised Contrastive Learning"

    Args:
        temperature: Temperature for scaling
        base_temperature: Base temperature for normalization
        contrast_mode: 'all' or 'one' (use all positives or sample one)
    """

    def __init__(self, temperature: float = 0.07,
                 base_temperature: float = 0.07,
                 contrast_mode: str = 'all'):
        super().__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, features: torch.Tensor,
               labels: torch.Tensor,
               mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: Feature vectors, shape (B, N, D)
            labels: Instance labels, shape (B, N)
            mask: Optional mask for valid positions, shape (B, N)

        Returns:
            Scalar loss
        """
        device = features.device

        if features.dim() == 2:
            features = features.unsqueeze(0)
            labels = labels.unsqueeze(0)

        B, N, D = features.shape

        # Normalize features
        features = F.normalize(features, dim=-1)

        total_loss = 0.0
        n_batches = 0

        for b in range(B):
            feat = features[b]
            lab = labels[b]

            # Apply mask if provided
            if mask is not None:
                valid = mask[b]
                feat = feat[valid]
                lab = lab[valid]

            # Filter to labeled points (label > 0)
            labeled = lab > 0
            if labeled.sum() < 2:
                continue

            feat = feat[labeled]
            lab = lab[labeled]
            n = feat.shape[0]

            # Compute similarity matrix
            sim_matrix = torch.mm(feat, feat.t()) / self.temperature

            # Create label mask (1 if same label, 0 otherwise)
            label_mask = lab.unsqueeze(0) == lab.unsqueeze(1)
            label_mask = label_mask.float()

            # Remove diagonal
            logits_mask = torch.ones_like(sim_matrix) - torch.eye(n, device=device)
            label_mask = label_mask * logits_mask

            # Compute log softmax over negatives
            exp_sim = torch.exp(sim_matrix) * logits_mask
            log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

            # Compute mean of log-likelihood over positive pairs
            mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / (label_mask.sum(dim=1) + 1e-6)

            # Loss: negative log-likelihood
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

            # Only average over anchors with positives
            has_positive = label_mask.sum(dim=1) > 0
            if has_positive.sum() > 0:
                loss = loss[has_positive].mean()
                total_loss += loss
                n_batches += 1

        if n_batches > 0:
            return total_loss / n_batches
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class ParticlePoolingContrastiveLoss(nn.Module):
    """
    Contrastive loss that pools spacepoint features to particle level.

    Instead of point-to-point comparison, pools features from each particle
    and compares particle-level representations.

    Args:
        temperature: Temperature for contrastive loss
        pooling: Pooling method ('mean', 'max', 'attention')
        projection_dim: Dimension of projection head
    """

    def __init__(self, temperature: float = 0.1,
                 pooling: str = 'mean',
                 feature_dim: int = 256,
                 projection_dim: int = 128):
        super().__init__()

        self.temperature = temperature
        self.pooling = pooling

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1),
                nn.Softmax(dim=0)
            )

    def pool_particles(self, features: torch.Tensor,
                      instance_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool spacepoint features to particle level.

        Args:
            features: Shape (N, D)
            instance_labels: Shape (N,)

        Returns:
            pooled_features: Shape (P, D) where P is number of particles
            particle_ids: Shape (P,)
        """
        unique_particles = torch.unique(instance_labels)
        unique_particles = unique_particles[unique_particles > 0]

        pooled = []
        particle_ids = []

        for pid in unique_particles:
            mask = instance_labels == pid
            particle_feat = features[mask]

            if self.pooling == 'mean':
                pooled_feat = particle_feat.mean(dim=0)
            elif self.pooling == 'max':
                pooled_feat = particle_feat.max(dim=0)[0]
            elif self.pooling == 'attention':
                weights = self.attention(particle_feat)
                pooled_feat = (weights * particle_feat).sum(dim=0)
            else:
                pooled_feat = particle_feat.mean(dim=0)

            pooled.append(pooled_feat)
            particle_ids.append(pid)

        if len(pooled) == 0:
            return None, None

        return torch.stack(pooled), torch.tensor(particle_ids, device=features.device)

    def forward(self, features: torch.Tensor,
               instance_labels: torch.Tensor,
               is_true: torch.Tensor = None) -> torch.Tensor:
        """
        Compute particle-level contrastive loss.

        Args:
            features: Shape (B, N, D) or (N, D)
            instance_labels: Shape (B, N) or (N,)
            is_true: Optional mask for true spacepoints

        Returns:
            Scalar loss
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)
            instance_labels = instance_labels.unsqueeze(0)
            if is_true is not None:
                is_true = is_true.unsqueeze(0)

        B = features.shape[0]
        device = features.device

        total_loss = 0.0
        n_valid = 0

        for b in range(B):
            feat = features[b]
            labels = instance_labels[b]

            # Filter to true if provided
            if is_true is not None:
                valid = is_true[b]
                feat = feat[valid]
                labels = labels[valid]

            # Pool to particle level
            pooled, pids = self.pool_particles(feat, labels)

            if pooled is None or len(pooled) < 2:
                continue

            # Project
            projected = self.projector(pooled)
            projected = F.normalize(projected, dim=-1)

            # InfoNCE loss: each particle against all others
            sim = torch.mm(projected, projected.t()) / self.temperature
            n_particles = sim.shape[0]

            # Diagonal is self-similarity, use as positive
            # All off-diagonal are negatives
            labels_cont = torch.arange(n_particles, device=device)
            loss = F.cross_entropy(sim, labels_cont)

            total_loss += loss
            n_valid += 1

        if n_valid > 0:
            return total_loss / n_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
