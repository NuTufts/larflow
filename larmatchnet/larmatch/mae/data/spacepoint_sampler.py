"""
Spacepoint Sampling Strategies

Implements various sampling strategies to handle large numbers of spacepoints
(200k-600k per event) and the high ghost-to-true ratio (10:1 to 20:1).
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List


class BaseSampler:
    """Base class for spacepoint sampling strategies."""

    def __init__(self, max_points: int = 50000):
        """
        Args:
            max_points: Maximum number of points to sample
        """
        self.max_points = max_points

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Sample spacepoints from data.

        Args:
            data: Dictionary containing spacepoint data arrays

        Returns:
            Dictionary with sampled data
        """
        raise NotImplementedError

    def _apply_indices(self, data: Dict[str, np.ndarray],
                      indices: np.ndarray,
                      keys_to_sample: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Apply sampling indices to data dictionary.

        Args:
            data: Data dictionary
            indices: Indices to sample
            keys_to_sample: Keys to apply sampling to (default: all spacepoint arrays)

        Returns:
            Sampled data dictionary
        """
        # Default keys that are per-spacepoint
        if keys_to_sample is None:
            keys_to_sample = [
                'matchtriplet', 'spacepoints', 'edep', 'trackid',
                'ssnet_label', 'ssnet_top_weight', 'kplabel', 'origin',
                'larmatch_truth', 'query_coord_0', 'query_coord_1', 'query_coord_2'
            ]

        sampled = {}
        for key, value in data.items():
            if key in keys_to_sample and isinstance(value, np.ndarray):
                if len(value) == len(indices) or len(value) >= indices.max() + 1:
                    sampled[key] = value[indices]
                else:
                    sampled[key] = value
            else:
                sampled[key] = value

        sampled['sample_indices'] = indices
        sampled['npts'] = len(indices)

        return sampled


class RandomSampler(BaseSampler):
    """
    Random uniform sampling of spacepoints.

    Simple but may undersample rare particles or important regions.
    """

    def __init__(self, max_points: int = 50000):
        super().__init__(max_points)

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sample random subset of spacepoints."""
        n_points = data['npts']

        if n_points <= self.max_points:
            # Keep all points
            indices = np.arange(n_points)
        else:
            # Random sample
            indices = np.random.choice(n_points, size=self.max_points, replace=False)
            indices = np.sort(indices)

        return self._apply_indices(data, indices)


class ImportanceSampler(BaseSampler):
    """
    Importance sampling - preferentially samples based on importance scores.

    Prioritizes:
    - True (non-ghost) spacepoints
    - Boundary/edge spacepoints
    - Keypoint regions
    """

    def __init__(self, max_points: int = 50000,
                 true_weight: float = 5.0,
                 boundary_weight: float = 2.0,
                 keypoint_weight: float = 3.0,
                 min_true_fraction: float = 0.3):
        """
        Args:
            max_points: Maximum points to sample
            true_weight: Weight multiplier for true spacepoints
            boundary_weight: Weight for boundary spacepoints
            keypoint_weight: Weight for keypoint regions
            min_true_fraction: Minimum fraction of true points in sample
        """
        super().__init__(max_points)
        self.true_weight = true_weight
        self.boundary_weight = boundary_weight
        self.keypoint_weight = keypoint_weight
        self.min_true_fraction = min_true_fraction

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sample based on importance weights."""
        n_points = data['npts']

        if n_points <= self.max_points:
            return self._apply_indices(data, np.arange(n_points))

        # Compute importance weights
        weights = np.ones(n_points, dtype=np.float32)

        # True spacepoints get higher weight
        if 'larmatch_truth' in data:
            is_true = data['larmatch_truth'] > 0
            weights[is_true] *= self.true_weight

        # Boundary spacepoints (ssnet_top_weight indicates boundary)
        if 'ssnet_top_weight' in data:
            is_boundary = data['ssnet_top_weight'] > 0
            weights[is_boundary] *= self.boundary_weight

        # Keypoint regions (high keypoint scores)
        if 'kplabel' in data:
            kp_scores = data['kplabel']
            if kp_scores.ndim > 1:
                max_kp_score = kp_scores.max(axis=1)
            else:
                max_kp_score = kp_scores
            is_keypoint = max_kp_score > 0.5
            weights[is_keypoint] *= self.keypoint_weight

        # Normalize weights
        weights = weights / weights.sum()

        # Sample with importance weights
        indices = np.random.choice(
            n_points,
            size=self.max_points,
            replace=False,
            p=weights
        )

        # Ensure minimum true fraction
        if 'larmatch_truth' in data:
            is_true = data['larmatch_truth'] > 0
            sampled_true = is_true[indices]
            true_count = sampled_true.sum()
            min_true = int(self.max_points * self.min_true_fraction)

            if true_count < min_true:
                # Need to add more true points
                n_to_add = min_true - true_count

                # Find unsampled true points
                all_true_idx = np.where(is_true)[0]
                sampled_set = set(indices)
                unsampled_true = [i for i in all_true_idx if i not in sampled_set]

                if len(unsampled_true) >= n_to_add:
                    # Replace some ghost points with true points
                    add_indices = np.random.choice(unsampled_true, size=n_to_add, replace=False)

                    # Find ghost points to replace
                    ghost_in_sample = np.where(~sampled_true)[0]
                    replace_positions = np.random.choice(
                        ghost_in_sample, size=n_to_add, replace=False
                    )

                    indices[replace_positions] = add_indices

        indices = np.sort(indices)
        return self._apply_indices(data, indices)


class StratifiedGhostSampler(BaseSampler):
    """
    Stratified sampling to balance ghost and true spacepoints.

    Samples separately from ghost and true populations to achieve
    a target ratio, addressing the 10:1 to 20:1 ghost imbalance.
    """

    def __init__(self, max_points: int = 50000,
                 target_ghost_ratio: float = 2.0):
        """
        Args:
            max_points: Maximum points to sample
            target_ghost_ratio: Target ratio of ghost:true (e.g., 2.0 means 2:1)
        """
        super().__init__(max_points)
        self.target_ghost_ratio = target_ghost_ratio

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sample with balanced ghost/true ratio."""
        n_points = data['npts']

        if n_points <= self.max_points:
            return self._apply_indices(data, np.arange(n_points))

        # Get ghost/true indices
        if 'larmatch_truth' in data:
            is_true = data['larmatch_truth'] > 0
        else:
            # Assume all are true if no label
            return RandomSampler(self.max_points)(data)

        true_indices = np.where(is_true)[0]
        ghost_indices = np.where(~is_true)[0]

        n_true = len(true_indices)
        n_ghost = len(ghost_indices)

        # Compute target counts
        # ghost/true = target_ratio, ghost + true = max_points
        # true = max_points / (1 + ratio)
        target_true = int(self.max_points / (1 + self.target_ghost_ratio))
        target_ghost = self.max_points - target_true

        # Adjust if not enough points
        actual_true = min(target_true, n_true)
        actual_ghost = min(target_ghost, n_ghost)

        # If we can't hit target, redistribute
        if actual_true < target_true:
            # Take more ghosts
            actual_ghost = min(self.max_points - actual_true, n_ghost)
        elif actual_ghost < target_ghost:
            # Take more true
            actual_true = min(self.max_points - actual_ghost, n_true)

        # Sample from each population
        if n_true > 0:
            sampled_true = np.random.choice(true_indices, size=actual_true, replace=False)
        else:
            sampled_true = np.array([], dtype=np.int64)

        if n_ghost > 0:
            sampled_ghost = np.random.choice(ghost_indices, size=actual_ghost, replace=False)
        else:
            sampled_ghost = np.array([], dtype=np.int64)

        # Combine and sort
        indices = np.sort(np.concatenate([sampled_true, sampled_ghost]))

        return self._apply_indices(data, indices)


class ParticleLevelSampler(BaseSampler):
    """
    Sample a fixed number of spacepoints per particle instance.

    Ensures representation from all particles regardless of size.
    """

    def __init__(self, max_points: int = 50000,
                 points_per_particle: int = 100,
                 max_particles: int = 200):
        """
        Args:
            max_points: Maximum total points
            points_per_particle: Target points per particle
            max_particles: Maximum number of particles to sample
        """
        super().__init__(max_points)
        self.points_per_particle = points_per_particle
        self.max_particles = max_particles

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sample fixed points per particle."""
        n_points = data['npts']

        if n_points <= self.max_points:
            return self._apply_indices(data, np.arange(n_points))

        if 'trackid' not in data:
            return RandomSampler(self.max_points)(data)

        trackids = data['trackid']
        unique_particles = np.unique(trackids)

        # Remove background (label 0)
        unique_particles = unique_particles[unique_particles > 0]

        # Limit number of particles
        if len(unique_particles) > self.max_particles:
            unique_particles = np.random.choice(
                unique_particles, size=self.max_particles, replace=False
            )

        sampled_indices = []

        for particle_id in unique_particles:
            particle_indices = np.where(trackids == particle_id)[0]
            n_particle = len(particle_indices)

            # Sample points from this particle
            n_sample = min(self.points_per_particle, n_particle)
            if n_particle > n_sample:
                selected = np.random.choice(particle_indices, size=n_sample, replace=False)
            else:
                selected = particle_indices

            sampled_indices.extend(selected)

        # Fill remaining with background or random points
        remaining = self.max_points - len(sampled_indices)
        if remaining > 0:
            sampled_set = set(sampled_indices)
            available = [i for i in range(n_points) if i not in sampled_set]

            if len(available) >= remaining:
                additional = np.random.choice(available, size=remaining, replace=False)
                sampled_indices.extend(additional)

        # Sort and convert
        indices = np.sort(np.array(sampled_indices[:self.max_points], dtype=np.int64))

        return self._apply_indices(data, indices)


class SpatialGridSampler(BaseSampler):
    """
    Sample uniformly across a spatial grid.

    Divides detector into grid cells and samples evenly from each.
    """

    def __init__(self, max_points: int = 50000,
                 grid_size: Tuple[int, int, int] = (10, 10, 10)):
        """
        Args:
            max_points: Maximum points to sample
            grid_size: Number of grid cells in (x, y, z)
        """
        super().__init__(max_points)
        self.grid_size = grid_size

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sample uniformly across spatial grid."""
        n_points = data['npts']

        if n_points <= self.max_points:
            return self._apply_indices(data, np.arange(n_points))

        if 'spacepoints' not in data:
            return RandomSampler(self.max_points)(data)

        positions = data['spacepoints']

        # Compute grid cell for each point
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        cell_size = (max_pos - min_pos) / np.array(self.grid_size)

        # Avoid division by zero
        cell_size = np.maximum(cell_size, 1e-6)

        cell_indices = ((positions - min_pos) / cell_size).astype(np.int32)
        cell_indices = np.clip(cell_indices, 0, np.array(self.grid_size) - 1)

        # Convert to linear cell index
        linear_cells = (
            cell_indices[:, 0] * self.grid_size[1] * self.grid_size[2] +
            cell_indices[:, 1] * self.grid_size[2] +
            cell_indices[:, 2]
        )

        # Get unique cells and their point indices
        unique_cells = np.unique(linear_cells)
        n_cells = len(unique_cells)

        # Target points per cell
        points_per_cell = max(1, self.max_points // n_cells)

        sampled_indices = []
        for cell in unique_cells:
            cell_points = np.where(linear_cells == cell)[0]
            n_sample = min(points_per_cell, len(cell_points))

            if len(cell_points) > n_sample:
                selected = np.random.choice(cell_points, size=n_sample, replace=False)
            else:
                selected = cell_points

            sampled_indices.extend(selected)

            if len(sampled_indices) >= self.max_points:
                break

        indices = np.sort(np.array(sampled_indices[:self.max_points], dtype=np.int64))

        return self._apply_indices(data, indices)


def create_sampler(config: dict) -> BaseSampler:
    """
    Factory function to create sampler from config.

    Args:
        config: Configuration dictionary

    Returns:
        Sampler instance
    """
    sampler_type = config.get('SAMPLER_TYPE', 'importance')
    max_points = config.get('MAX_SPACEPOINTS', 50000)

    if sampler_type == 'random':
        return RandomSampler(max_points)

    elif sampler_type == 'importance':
        return ImportanceSampler(
            max_points=max_points,
            true_weight=config.get('TRUE_WEIGHT', 5.0),
            boundary_weight=config.get('BOUNDARY_WEIGHT', 2.0),
            keypoint_weight=config.get('KEYPOINT_WEIGHT', 3.0),
            min_true_fraction=config.get('MIN_TRUE_FRACTION', 0.3)
        )

    elif sampler_type == 'stratified':
        return StratifiedGhostSampler(
            max_points=max_points,
            target_ghost_ratio=config.get('TARGET_GHOST_RATIO', 2.0)
        )

    elif sampler_type == 'particle':
        return ParticleLevelSampler(
            max_points=max_points,
            points_per_particle=config.get('POINTS_PER_PARTICLE', 100),
            max_particles=config.get('MAX_PARTICLES', 200)
        )

    elif sampler_type == 'spatial':
        return SpatialGridSampler(
            max_points=max_points,
            grid_size=tuple(config.get('GRID_SIZE', [10, 10, 10]))
        )

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
