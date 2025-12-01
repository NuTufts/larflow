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


class SamplerChain(BaseSampler):
    """
    Chain multiple samplers together.

    Data passes through each sampler in sequence. This allows combining
    different sampling strategies, e.g., spatial box filtering followed
    by importance sampling.

    Example:
        chain = SamplerChain([
            SpatialBoxSampler(box_min=[0,0,0], box_max=[100,100,100]),
            ImportanceSampler(max_points=5000)
        ])
        sampled_data = chain(data)
    """

    def __init__(self, samplers: List['BaseSampler']):
        """
        Args:
            samplers: List of sampler instances to chain together
        """
        # Use max_points from last sampler in chain
        if samplers:
            super().__init__(samplers[-1].max_points)
        else:
            super().__init__()

        self.samplers = samplers

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply samplers in sequence.

        Args:
            data: Dictionary containing spacepoint data arrays

        Returns:
            Dictionary with sampled data after all samplers applied
        """
        result = data
        for sampler in self.samplers:
            result = sampler(result)
        return result

    def __len__(self):
        return len(self.samplers)

    def __getitem__(self, idx):
        return self.samplers[idx]

    def append(self, sampler: 'BaseSampler'):
        """Add a sampler to the end of the chain."""
        self.samplers.append(sampler)
        self.max_points = sampler.max_points

    def get_sampler_by_name(self, name: str) -> Optional['BaseSampler']:
        """
        Get a sampler by its class name or config name.

        Args:
            name: Name to search for (matches class name or 'name' attribute)

        Returns:
            Matching sampler or None
        """
        for sampler in self.samplers:
            if sampler.__class__.__name__.lower() == name.lower():
                return sampler
            if hasattr(sampler, 'name') and sampler.name == name:
                return sampler
        return None


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


class SpatialBoxSampler(BaseSampler):
    """
    Sample all spacepoints within a defined spatial box.

    This sampler selects all points that fall within a 3D box region,
    useful for processing large events in spatial chunks during inference.

    The sampling box is defined relative to a detector/data coordinate system
    with a specified origin and overall extent.

    Supports two modes:
    1. Fixed box mode: box_min and box_max define a fixed sampling region
    2. Random box mode: box_size defines dimensions, box is randomly placed
       within the detector volume on each call
    """

    def __init__(self,
                 max_points: int = 50000,
                 box_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 box_max: Tuple[float, float, float] = (100.0, 100.0, 100.0),
                 detector_origin: Tuple[float, float, float] = (0.0, -117.0, 0.0),
                 detector_extent: Tuple[float, float, float] = (256.0, 234.0, 1036.0),
                 padding: float = 0.0,
                 fallback_to_random: bool = True,
                 random_box_mode: bool = False,
                 box_size: Optional[Tuple[float, float, float]] = None,
                 min_points: int = 0,
                 max_resample_attempts: int = 100):
        """
        Args:
            max_points: Maximum number of points to return (if box contains more)
            box_min: Minimum corner of sampling box in cm (x, y, z) - used in fixed mode
            box_max: Maximum corner of sampling box in cm (x, y, z) - used in fixed mode
            detector_origin: Origin of detector coordinate system (x, y, z) in cm
            detector_extent: Size of detector in each dimension (x, y, z) in cm
            padding: Extra padding around the box in cm (extends the box)
            fallback_to_random: If True, randomly sample if box contains > max_points
            random_box_mode: If True, randomly place box within detector on each call
            box_size: Size of the box (x, y, z) in cm - required for random_box_mode
            min_points: Minimum number of points required in box (random_box_mode only).
                        If box contains fewer points, resample box position.
            max_resample_attempts: Maximum attempts to find a box with min_points (default: 100)
        """
        super().__init__(max_points)
        self.box_min = np.array(box_min, dtype=np.float32)
        self.box_max = np.array(box_max, dtype=np.float32)
        self.detector_origin = np.array(detector_origin, dtype=np.float32)
        self.detector_extent = np.array(detector_extent, dtype=np.float32)
        self.padding = padding
        self.fallback_to_random = fallback_to_random
        self.random_box_mode = random_box_mode
        self.min_points = min_points
        self.max_resample_attempts = max_resample_attempts

        # For random box mode
        if box_size is not None:
            self.box_size = np.array(box_size, dtype=np.float32)
        else:
            # Default: derive from box_min/box_max
            self.box_size = self.box_max - self.box_min

        # Store the last randomly generated box for visualization/debugging
        self.last_random_box_min = None
        self.last_random_box_max = None

        # Track resampling statistics
        self.last_resample_attempts = 0

    def set_box(self, box_min: Tuple[float, float, float],
                box_max: Tuple[float, float, float]):
        """
        Update the sampling box position.

        Args:
            box_min: New minimum corner (x, y, z)
            box_max: New maximum corner (x, y, z)
        """
        self.box_min = np.array(box_min, dtype=np.float32)
        self.box_max = np.array(box_max, dtype=np.float32)
        self.box_size = self.box_max - self.box_min

    def set_box_size(self, box_size: Tuple[float, float, float]):
        """
        Set the box size for random box mode.

        Args:
            box_size: Size of box (x, y, z) in cm
        """
        self.box_size = np.array(box_size, dtype=np.float32)

    def generate_random_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random box position within the detector volume.

        The box is constrained to fit entirely within the detector bounds.

        Returns:
            Tuple of (box_min, box_max) arrays
        """
        # Compute the valid range for the box minimum corner
        # The box must fit within [detector_origin, detector_origin + detector_extent]
        detector_max = self.detector_origin + self.detector_extent

        # Valid range for box_min: [detector_origin, detector_max - box_size]
        valid_min = self.detector_origin.copy()
        valid_max = detector_max - self.box_size

        # Ensure valid_max >= valid_min (box fits in detector)
        if np.any(valid_max < valid_min):
            # Box is larger than detector in some dimension - clamp to detector
            valid_max = np.maximum(valid_max, valid_min)

        # Generate random position for box minimum corner
        random_box_min = np.random.uniform(valid_min, valid_max).astype(np.float32)
        random_box_max = random_box_min + self.box_size

        # Clamp to detector bounds (safety check)
        random_box_min = np.maximum(random_box_min, self.detector_origin)
        random_box_max = np.minimum(random_box_max, detector_max)

        # Store for visualization/debugging
        self.last_random_box_min = random_box_min.copy()
        self.last_random_box_max = random_box_max.copy()

        return random_box_min, random_box_max

    def get_last_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the last box used (either fixed or randomly generated).

        Returns:
            Tuple of (box_min, box_max) arrays
        """
        if self.random_box_mode and self.last_random_box_min is not None:
            return self.last_random_box_min, self.last_random_box_max
        return self.box_min, self.box_max

    def set_box_from_grid(self, grid_index: Tuple[int, int, int],
                          grid_divisions: Tuple[int, int, int]):
        """
        Set the sampling box based on a grid index.

        Divides the detector into a grid and sets the box to the specified cell.

        Args:
            grid_index: (ix, iy, iz) index of the grid cell
            grid_divisions: (nx, ny, nz) number of divisions in each dimension
        """
        cell_size = self.detector_extent / np.array(grid_divisions, dtype=np.float32)

        self.box_min = self.detector_origin + np.array(grid_index, dtype=np.float32) * cell_size
        self.box_max = self.box_min + cell_size

    def get_box_bounds(self, use_current: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current box bounds with padding applied.

        In random_box_mode, this generates a new random box unless use_current=True.

        Args:
            use_current: If True, use the last generated box instead of generating new

        Returns:
            Tuple of (box_min, box_max) with padding
        """
        if self.random_box_mode:
            if use_current and self.last_random_box_min is not None:
                box_min, box_max = self.last_random_box_min, self.last_random_box_max
            else:
                box_min, box_max = self.generate_random_box()
        else:
            box_min, box_max = self.box_min, self.box_max

        padded_min = box_min - self.padding
        padded_max = box_max + self.padding
        return padded_min, padded_max

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Sample all spacepoints within the defined box.

        In random_box_mode, a new random box position is generated each call.
        If min_points > 0, will resample until finding a box with enough points.

        Args:
            data: Dictionary containing spacepoint data arrays

        Returns:
            Dictionary with sampled data (only points in box)
        """
        n_points = data['npts']

        if 'spacepoints' not in data:
            raise ValueError("SpatialBoxSampler requires 'spacepoints' in data")

        positions = data['spacepoints']

        # Reset resample counter
        self.last_resample_attempts = 0

        # In random box mode with min_points, we may need to resample
        if self.random_box_mode and self.min_points > 0:
            indices = self._sample_with_min_points(positions)
        else:
            # Standard sampling (no resampling)
            box_min, box_max = self.get_box_bounds(use_current=False)
            inside_mask = np.all(
                (positions >= box_min) & (positions <= box_max),
                axis=1
            )
            indices = np.where(inside_mask)[0]

        # Handle case where box contains too many points
        if len(indices) > self.max_points:
            if self.fallback_to_random:
                # Randomly sample from points in box
                indices = np.random.choice(indices, size=self.max_points, replace=False)
                indices = np.sort(indices)
            else:
                # Just truncate (maintain spatial ordering)
                indices = indices[:self.max_points]

        return self._apply_indices(data, indices)

    def _sample_with_min_points(self, positions: np.ndarray) -> np.ndarray:
        """
        Sample a random box that contains at least min_points.

        Resamples box position up to max_resample_attempts times.
        If no valid box is found, returns the best box found (most points).

        Args:
            positions: Array of spacepoint positions (N, 3)

        Returns:
            Indices of points inside the selected box
        """
        best_indices = None
        best_count = 0

        for attempt in range(self.max_resample_attempts):
            self.last_resample_attempts = attempt + 1

            # Generate new random box
            box_min, box_max = self.get_box_bounds(use_current=False)

            # Find points inside
            inside_mask = np.all(
                (positions >= box_min) & (positions <= box_max),
                axis=1
            )
            indices = np.where(inside_mask)[0]
            n_inside = len(indices)

            # Track best result
            if n_inside > best_count:
                best_count = n_inside
                best_indices = indices
                # Also save the best box coordinates
                best_box_min = self.last_random_box_min.copy()
                best_box_max = self.last_random_box_max.copy()

            # Check if we have enough points
            if n_inside >= self.min_points:
                return indices

        # If we didn't find a box with min_points, use the best one found
        # Restore the best box coordinates
        if best_indices is not None:
            self.last_random_box_min = best_box_min
            self.last_random_box_max = best_box_max

        return best_indices if best_indices is not None else np.array([], dtype=np.int64)

    def count_points_in_box(self, positions: np.ndarray) -> int:
        """
        Count how many points are in the current box.

        Args:
            positions: Array of spacepoint positions (N, 3)

        Returns:
            Number of points inside the box
        """
        box_min, box_max = self.get_box_bounds()
        inside_mask = np.all(
            (positions >= box_min) & (positions <= box_max),
            axis=1
        )
        return inside_mask.sum()

    @staticmethod
    def compute_grid_divisions(detector_extent: Tuple[float, float, float],
                               target_points_per_cell: int,
                               total_points: int) -> Tuple[int, int, int]:
        """
        Compute optimal grid divisions based on point density.

        Args:
            detector_extent: Size of detector (x, y, z) in cm
            target_points_per_cell: Target number of points per grid cell
            total_points: Total number of points in detector

        Returns:
            Tuple of (nx, ny, nz) grid divisions
        """
        # Estimate point density
        volume = np.prod(detector_extent)
        density = total_points / volume

        # Target cell volume
        target_cell_volume = target_points_per_cell / density

        # Compute cell size (assuming cubic cells)
        cell_size = target_cell_volume ** (1/3)

        # Compute divisions
        divisions = np.maximum(1, np.round(detector_extent / cell_size)).astype(int)

        return tuple(divisions)


class SpatialBoxIterator:
    """
    Iterator that yields SpatialBoxSampler configurations for processing
    an entire detector volume in spatial chunks.

    Useful for inference when you need to process all spacepoints but
    can only handle a limited number at a time.
    """

    def __init__(self,
                 detector_origin: Tuple[float, float, float] = (0.0, -117.0, 0.0),
                 detector_extent: Tuple[float, float, float] = (256.0, 234.0, 1036.0),
                 grid_divisions: Tuple[int, int, int] = (4, 4, 16),
                 padding: float = 5.0,
                 max_points_per_box: int = 50000):
        """
        Args:
            detector_origin: Origin of detector (x, y, z) in cm
            detector_extent: Size of detector (x, y, z) in cm
            grid_divisions: Number of divisions in each dimension (nx, ny, nz)
            padding: Overlap padding between boxes in cm
            max_points_per_box: Maximum points per box
        """
        self.detector_origin = np.array(detector_origin, dtype=np.float32)
        self.detector_extent = np.array(detector_extent, dtype=np.float32)
        self.grid_divisions = grid_divisions
        self.padding = padding
        self.max_points_per_box = max_points_per_box

        # Compute cell size
        self.cell_size = self.detector_extent / np.array(grid_divisions, dtype=np.float32)

        # Total number of cells
        self.n_cells = np.prod(grid_divisions)

    def __len__(self):
        return self.n_cells

    def __iter__(self):
        """Iterate over all grid cells."""
        for ix in range(self.grid_divisions[0]):
            for iy in range(self.grid_divisions[1]):
                for iz in range(self.grid_divisions[2]):
                    yield self.get_sampler_for_cell(ix, iy, iz)

    def get_sampler_for_cell(self, ix: int, iy: int, iz: int) -> SpatialBoxSampler:
        """
        Get a SpatialBoxSampler configured for a specific grid cell.

        Args:
            ix, iy, iz: Grid cell indices

        Returns:
            Configured SpatialBoxSampler
        """
        box_min = self.detector_origin + np.array([ix, iy, iz], dtype=np.float32) * self.cell_size
        box_max = box_min + self.cell_size

        return SpatialBoxSampler(
            max_points=self.max_points_per_box,
            box_min=tuple(box_min),
            box_max=tuple(box_max),
            detector_origin=tuple(self.detector_origin),
            detector_extent=tuple(self.detector_extent),
            padding=self.padding,
            fallback_to_random=True
        )

    def get_cell_index(self, linear_index: int) -> Tuple[int, int, int]:
        """
        Convert linear index to grid cell index.

        Args:
            linear_index: Linear cell index

        Returns:
            Tuple of (ix, iy, iz)
        """
        iz = linear_index % self.grid_divisions[2]
        iy = (linear_index // self.grid_divisions[2]) % self.grid_divisions[1]
        ix = linear_index // (self.grid_divisions[1] * self.grid_divisions[2])
        return (ix, iy, iz)

    def get_sampler_by_index(self, linear_index: int) -> SpatialBoxSampler:
        """
        Get sampler for a cell by linear index.

        Args:
            linear_index: Linear cell index (0 to n_cells-1)

        Returns:
            Configured SpatialBoxSampler
        """
        ix, iy, iz = self.get_cell_index(linear_index)
        return self.get_sampler_for_cell(ix, iy, iz)


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

    elif sampler_type == 'spatial_box':
        # Check for random box mode
        random_box_mode = config.get('RANDOM_BOX_MODE', False)
        box_size = config.get('BOX_SIZE', None)
        if box_size is not None:
            box_size = tuple(box_size)

        return SpatialBoxSampler(
            max_points=max_points,
            box_min=tuple(config.get('BOX_MIN', [0.0, -117.0, 0.0])),
            box_max=tuple(config.get('BOX_MAX', [256.0, 117.0, 1036.0])),
            detector_origin=tuple(config.get('DETECTOR_ORIGIN', [0.0, -117.0, 0.0])),
            detector_extent=tuple(config.get('DETECTOR_EXTENT', [256.0, 234.0, 1036.0])),
            padding=config.get('BOX_PADDING', 0.0),
            fallback_to_random=config.get('BOX_FALLBACK_TO_RANDOM', True),
            random_box_mode=random_box_mode,
            box_size=box_size,
            min_points=config.get('MIN_POINTS_IN_BOX', 0),
            max_resample_attempts=config.get('MAX_RESAMPLE_ATTEMPTS', 100)
        )

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def create_sampler_from_block(sampler_config: dict) -> BaseSampler:
    """
    Create a single sampler from a configuration block.

    This function creates a sampler from a self-contained config block
    that includes the 'type' field and all sampler-specific parameters.

    Args:
        sampler_config: Dictionary with sampler configuration including:
            - type: Sampler type string (required)
            - max_points: Maximum points (optional, default 50000)
            - Other sampler-specific parameters

    Returns:
        Configured sampler instance

    Example config block:
        {
            "type": "spatial_box",
            "max_points": 100000,
            "random_box_mode": true,
            "box_size": [100, 100, 200],
            "min_points": 500
        }
    """
    sampler_type = sampler_config.get('type', sampler_config.get('TYPE', 'importance'))
    max_points = sampler_config.get('max_points', sampler_config.get('MAX_POINTS', 50000))

    if sampler_type == 'random':
        return RandomSampler(max_points)

    elif sampler_type == 'importance':
        return ImportanceSampler(
            max_points=max_points,
            true_weight=sampler_config.get('true_weight', sampler_config.get('TRUE_WEIGHT', 5.0)),
            boundary_weight=sampler_config.get('boundary_weight', sampler_config.get('BOUNDARY_WEIGHT', 2.0)),
            keypoint_weight=sampler_config.get('keypoint_weight', sampler_config.get('KEYPOINT_WEIGHT', 3.0)),
            min_true_fraction=sampler_config.get('min_true_fraction', sampler_config.get('MIN_TRUE_FRACTION', 0.3))
        )

    elif sampler_type == 'stratified':
        return StratifiedGhostSampler(
            max_points=max_points,
            target_ghost_ratio=sampler_config.get('target_ghost_ratio', sampler_config.get('TARGET_GHOST_RATIO', 2.0))
        )

    elif sampler_type == 'particle':
        return ParticleLevelSampler(
            max_points=max_points,
            points_per_particle=sampler_config.get('points_per_particle', sampler_config.get('POINTS_PER_PARTICLE', 100)),
            max_particles=sampler_config.get('max_particles', sampler_config.get('MAX_PARTICLES', 200))
        )

    elif sampler_type == 'spatial':
        grid_size = sampler_config.get('grid_size', sampler_config.get('GRID_SIZE', [10, 10, 10]))
        return SpatialGridSampler(
            max_points=max_points,
            grid_size=tuple(grid_size)
        )

    elif sampler_type == 'spatial_box':
        # Handle both lowercase and uppercase config keys
        random_box_mode = sampler_config.get('random_box_mode', sampler_config.get('RANDOM_BOX_MODE', False))
        box_size = sampler_config.get('box_size', sampler_config.get('BOX_SIZE', None))
        if box_size is not None:
            box_size = tuple(box_size)

        box_min = sampler_config.get('box_min', sampler_config.get('BOX_MIN', [0.0, -117.0, 0.0]))
        box_max = sampler_config.get('box_max', sampler_config.get('BOX_MAX', [256.0, 117.0, 1036.0]))
        detector_origin = sampler_config.get('detector_origin', sampler_config.get('DETECTOR_ORIGIN', [0.0, -117.0, 0.0]))
        detector_extent = sampler_config.get('detector_extent', sampler_config.get('DETECTOR_EXTENT', [256.0, 234.0, 1036.0]))

        return SpatialBoxSampler(
            max_points=max_points,
            box_min=tuple(box_min),
            box_max=tuple(box_max),
            detector_origin=tuple(detector_origin),
            detector_extent=tuple(detector_extent),
            padding=sampler_config.get('padding', sampler_config.get('BOX_PADDING', 0.0)),
            fallback_to_random=sampler_config.get('fallback_to_random',
                              sampler_config.get('box_fallback_to_random',
                              sampler_config.get('BOX_FALLBACK_TO_RANDOM', True))),
            random_box_mode=random_box_mode,
            box_size=box_size,
            min_points=sampler_config.get('min_points', sampler_config.get('MIN_POINTS_IN_BOX', 0)),
            max_resample_attempts=sampler_config.get('max_resample_attempts', sampler_config.get('MAX_RESAMPLE_ATTEMPTS', 100))
        )

    else:
        raise ValueError(f"Unknown sampler type in block: {sampler_type}")


def create_sampler_chain(config: dict) -> BaseSampler:
    """
    Create a sampler or sampler chain from configuration.

    This function handles both single-sampler configs (legacy) and
    sampler chain configs (new). It returns either a single sampler
    or a SamplerChain depending on the configuration.

    Config formats supported:

    1. Legacy single sampler (flat config):
        SAMPLER_TYPE: "importance"
        MAX_SPACEPOINTS: 5000
        TRUE_WEIGHT: 5.0

    2. New sampler chain (list of sampler blocks):
        SAMPLER_CHAIN:
          - type: spatial_box
            max_points: 100000
            random_box_mode: true
            box_size: [100, 100, 200]
            min_points: 500
          - type: importance
            max_points: 5000
            true_weight: 5.0
            min_true_fraction: 0.3

    Args:
        config: Configuration dictionary

    Returns:
        BaseSampler instance (either single sampler or SamplerChain)
    """
    # Check for new sampler chain format
    sampler_chain_config = config.get('SAMPLER_CHAIN', config.get('sampler_chain', None))

    if sampler_chain_config is not None:
        # New format: list of sampler configuration blocks
        if not isinstance(sampler_chain_config, list):
            raise ValueError("SAMPLER_CHAIN must be a list of sampler configurations")

        if len(sampler_chain_config) == 0:
            raise ValueError("SAMPLER_CHAIN cannot be empty")

        # Create each sampler from its config block
        samplers = []
        for i, sampler_block in enumerate(sampler_chain_config):
            if not isinstance(sampler_block, dict):
                raise ValueError(f"Sampler config at index {i} must be a dictionary")

            sampler = create_sampler_from_block(sampler_block)

            # Optionally assign a name for later retrieval
            if 'name' in sampler_block:
                sampler.name = sampler_block['name']

            samplers.append(sampler)

        # If only one sampler, return it directly
        if len(samplers) == 1:
            return samplers[0]

        return SamplerChain(samplers)

    else:
        # Legacy format: use create_sampler
        return create_sampler(config)
