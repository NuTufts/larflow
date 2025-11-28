"""
Masking Strategies for MAE Training

Implements various masking strategies for the masked auto-encoding task.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class BaseMasking:
    """Base class for masking strategies."""

    def __init__(self, mask_ratio: float = 0.75):
        self.mask_ratio = mask_ratio

    def __call__(self, n_tokens: int, **kwargs) -> torch.Tensor:
        """
        Generate mask for tokens.

        Args:
            n_tokens: Number of tokens to mask

        Returns:
            Boolean tensor of shape (n_tokens,), True = masked
        """
        raise NotImplementedError


class RandomMasking(BaseMasking):
    """
    Random uniform masking - each token has equal probability of being masked.

    This is the standard MAE masking strategy.
    """

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__(mask_ratio)

    def __call__(self, n_tokens: int, device: str = 'cpu', **kwargs) -> torch.Tensor:
        """
        Generate random mask.

        Args:
            n_tokens: Number of tokens
            device: Device for tensor

        Returns:
            Boolean mask tensor
        """
        n_masked = int(n_tokens * self.mask_ratio)

        # Random permutation
        perm = torch.randperm(n_tokens, device=device)

        # First n_masked are masked
        mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        mask[perm[:n_masked]] = True

        return mask

    def batch_call(self, batch_size: int, n_tokens: int,
                  device: str = 'cpu') -> torch.Tensor:
        """
        Generate masks for a batch.

        Args:
            batch_size: Number of samples in batch
            n_tokens: Number of tokens per sample
            device: Device for tensor

        Returns:
            Boolean mask tensor of shape (batch_size, n_tokens)
        """
        masks = torch.stack([
            self(n_tokens, device=device)
            for _ in range(batch_size)
        ])
        return masks


class SpatialMasking(BaseMasking):
    """
    Spatial block masking - masks contiguous spatial regions.

    Useful for encouraging the model to learn longer-range dependencies.
    """

    def __init__(self, mask_ratio: float = 0.75, block_size: float = 50.0):
        """
        Args:
            mask_ratio: Fraction of tokens to mask
            block_size: Size of spatial blocks in detector units (cm)
        """
        super().__init__(mask_ratio)
        self.block_size = block_size

    def __call__(self, n_tokens: int, positions: torch.Tensor,
                device: str = 'cpu', **kwargs) -> torch.Tensor:
        """
        Generate spatial block mask.

        Args:
            n_tokens: Number of tokens
            positions: 3D positions, shape (n_tokens, 3)
            device: Device for tensor

        Returns:
            Boolean mask tensor
        """
        if positions is None:
            # Fall back to random masking
            return RandomMasking(self.mask_ratio)(n_tokens, device)

        positions = positions.to(device)
        n_masked_target = int(n_tokens * self.mask_ratio)

        # Compute block indices for each position
        block_indices = (positions / self.block_size).long()

        # Get unique blocks
        unique_blocks, inverse_indices = torch.unique(
            block_indices, dim=0, return_inverse=True
        )
        n_blocks = unique_blocks.shape[0]

        # Randomly select blocks to mask until we reach target
        mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        block_perm = torch.randperm(n_blocks, device=device)

        n_masked = 0
        for block_idx in block_perm:
            if n_masked >= n_masked_target:
                break
            block_mask = (inverse_indices == block_idx)
            mask = mask | block_mask
            n_masked = mask.sum().item()

        # Trim if we overshot
        if n_masked > n_masked_target:
            masked_indices = mask.nonzero(as_tuple=True)[0]
            to_unmask = masked_indices[torch.randperm(n_masked)[:n_masked - n_masked_target]]
            mask[to_unmask] = False

        return mask


class ParticleAwareMasking(BaseMasking):
    """
    Particle-aware masking - masks entire particles or parts of particles.

    Options:
    - mask_entire_particles: Mask all spacepoints from selected particles
    - partial_particle_mask: Mask fraction of spacepoints from each particle
    """

    def __init__(self, mask_ratio: float = 0.75,
                 mask_entire_particles: bool = False,
                 min_particle_fraction: float = 0.3):
        """
        Args:
            mask_ratio: Overall fraction of tokens to mask
            mask_entire_particles: If True, mask entire particles at once
            min_particle_fraction: Minimum fraction of particle to mask if partial
        """
        super().__init__(mask_ratio)
        self.mask_entire_particles = mask_entire_particles
        self.min_particle_fraction = min_particle_fraction

    def __call__(self, n_tokens: int, instance_labels: torch.Tensor = None,
                device: str = 'cpu', **kwargs) -> torch.Tensor:
        """
        Generate particle-aware mask.

        Args:
            n_tokens: Number of tokens
            instance_labels: Particle instance labels, shape (n_tokens,)
            device: Device for tensor

        Returns:
            Boolean mask tensor
        """
        if instance_labels is None:
            return RandomMasking(self.mask_ratio)(n_tokens, device)

        instance_labels = instance_labels.to(device)
        n_masked_target = int(n_tokens * self.mask_ratio)

        # Get unique particles
        unique_particles = torch.unique(instance_labels)
        # Remove background (typically label 0 or -1)
        unique_particles = unique_particles[unique_particles > 0]

        mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)

        if self.mask_entire_particles:
            # Randomly select particles to fully mask
            particle_perm = torch.randperm(len(unique_particles), device=device)
            n_masked = 0

            for idx in particle_perm:
                if n_masked >= n_masked_target:
                    break
                particle_id = unique_particles[idx]
                particle_mask = (instance_labels == particle_id)
                mask = mask | particle_mask
                n_masked = mask.sum().item()

            # Handle overshoot by randomly unmasking some points
            if n_masked > n_masked_target:
                masked_indices = mask.nonzero(as_tuple=True)[0]
                n_to_unmask = n_masked - n_masked_target
                to_unmask = masked_indices[torch.randperm(n_masked)[:n_to_unmask]]
                mask[to_unmask] = False
        else:
            # Partial masking per particle
            for particle_id in unique_particles:
                particle_indices = (instance_labels == particle_id).nonzero(as_tuple=True)[0]
                n_particle = len(particle_indices)

                # Random fraction to mask from this particle
                frac = np.random.uniform(self.min_particle_fraction, 1.0)
                n_to_mask = int(n_particle * frac * self.mask_ratio / 0.5)  # Scale by expected

                if n_to_mask > 0:
                    perm = torch.randperm(n_particle, device=device)
                    mask[particle_indices[perm[:n_to_mask]]] = True

            # Adjust to hit target
            n_masked = mask.sum().item()
            if n_masked < n_masked_target:
                # Need to mask more - random from unmasked
                unmasked = (~mask).nonzero(as_tuple=True)[0]
                n_to_add = min(n_masked_target - n_masked, len(unmasked))
                add_indices = unmasked[torch.randperm(len(unmasked))[:n_to_add]]
                mask[add_indices] = True
            elif n_masked > n_masked_target:
                # Need to unmask some
                masked = mask.nonzero(as_tuple=True)[0]
                n_to_remove = n_masked - n_masked_target
                remove_indices = masked[torch.randperm(len(masked))[:n_to_remove]]
                mask[remove_indices] = False

        return mask


class GhostAwareMasking(BaseMasking):
    """
    Ghost-aware masking - handles the ghost vs true spacepoint distinction.

    Options:
    - mask_only_true: Only mask true (non-ghost) spacepoints
    - balanced_ghost_true: Balance masking between ghost and true
    """

    def __init__(self, mask_ratio: float = 0.75,
                 mask_only_true: bool = False,
                 ghost_mask_ratio: float = 0.5):
        """
        Args:
            mask_ratio: Overall fraction of true tokens to mask
            mask_only_true: If True, only mask true spacepoints
            ghost_mask_ratio: Fraction of ghost points to mask (if not mask_only_true)
        """
        super().__init__(mask_ratio)
        self.mask_only_true = mask_only_true
        self.ghost_mask_ratio = ghost_mask_ratio

    def __call__(self, n_tokens: int, is_true: torch.Tensor = None,
                device: str = 'cpu', **kwargs) -> torch.Tensor:
        """
        Generate ghost-aware mask.

        Args:
            n_tokens: Number of tokens
            is_true: Boolean tensor indicating true (non-ghost) spacepoints
            device: Device for tensor

        Returns:
            Boolean mask tensor
        """
        if is_true is None:
            return RandomMasking(self.mask_ratio)(n_tokens, device)

        is_true = is_true.to(device)
        mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)

        # Get indices
        true_indices = is_true.nonzero(as_tuple=True)[0]
        ghost_indices = (~is_true).nonzero(as_tuple=True)[0]

        # Mask true spacepoints
        n_true = len(true_indices)
        n_true_masked = int(n_true * self.mask_ratio)
        if n_true > 0 and n_true_masked > 0:
            perm = torch.randperm(n_true, device=device)
            mask[true_indices[perm[:n_true_masked]]] = True

        # Optionally mask ghost spacepoints
        if not self.mask_only_true:
            n_ghost = len(ghost_indices)
            n_ghost_masked = int(n_ghost * self.ghost_mask_ratio)
            if n_ghost > 0 and n_ghost_masked > 0:
                perm = torch.randperm(n_ghost, device=device)
                mask[ghost_indices[perm[:n_ghost_masked]]] = True

        return mask


class CurriculumMasking(BaseMasking):
    """
    Curriculum masking - gradually increases mask ratio during training.

    Starts with lower mask ratio and increases over training.
    """

    def __init__(self, initial_ratio: float = 0.5,
                 final_ratio: float = 0.85,
                 warmup_steps: int = 10000):
        """
        Args:
            initial_ratio: Starting mask ratio
            final_ratio: Final mask ratio
            warmup_steps: Steps to reach final ratio
        """
        super().__init__(initial_ratio)
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Increment training step."""
        self.current_step += 1

    def get_current_ratio(self) -> float:
        """Get current mask ratio based on training progress."""
        if self.current_step >= self.warmup_steps:
            return self.final_ratio

        progress = self.current_step / self.warmup_steps
        return self.initial_ratio + (self.final_ratio - self.initial_ratio) * progress

    def __call__(self, n_tokens: int, device: str = 'cpu', **kwargs) -> torch.Tensor:
        """Generate mask with current ratio."""
        current_ratio = self.get_current_ratio()
        return RandomMasking(current_ratio)(n_tokens, device)


def create_masking_strategy(config: dict) -> BaseMasking:
    """
    Factory function to create masking strategy from config.

    Args:
        config: Configuration dictionary with masking parameters

    Returns:
        Masking strategy instance
    """
    strategy_type = config.get('MASKING_STRATEGY', 'random')
    mask_ratio = config.get('MASK_RATIO', 0.75)

    if strategy_type == 'random':
        return RandomMasking(mask_ratio)

    elif strategy_type == 'spatial':
        block_size = config.get('SPATIAL_BLOCK_SIZE', 50.0)
        return SpatialMasking(mask_ratio, block_size)

    elif strategy_type == 'particle':
        mask_entire = config.get('MASK_ENTIRE_PARTICLES', False)
        min_frac = config.get('MIN_PARTICLE_FRACTION', 0.3)
        return ParticleAwareMasking(mask_ratio, mask_entire, min_frac)

    elif strategy_type == 'ghost_aware':
        mask_only_true = config.get('MASK_ONLY_TRUE', False)
        ghost_ratio = config.get('GHOST_MASK_RATIO', 0.5)
        return GhostAwareMasking(mask_ratio, mask_only_true, ghost_ratio)

    elif strategy_type == 'curriculum':
        initial = config.get('INITIAL_MASK_RATIO', 0.5)
        final = config.get('FINAL_MASK_RATIO', 0.85)
        warmup = config.get('MASK_WARMUP_STEPS', 10000)
        return CurriculumMasking(initial, final, warmup)

    else:
        raise ValueError(f"Unknown masking strategy: {strategy_type}")
