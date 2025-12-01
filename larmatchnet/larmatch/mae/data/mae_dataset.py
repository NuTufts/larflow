"""
MAE Dataset Wrapper

Wraps the LArMatchSimChHDF5Dataset with masking and sampling for MAE training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Callable

from larmatch.data.larmatch_simchhdf5_reader import LArMatchSimChHDF5Dataset
from .spacepoint_sampler import create_sampler, create_sampler_chain, BaseSampler, SamplerChain
from .masking import create_masking_strategy, BaseMasking


class MAEDataset(Dataset):
    """
    Dataset wrapper for MAE training.

    Wraps LArMatchSimChHDF5Dataset and adds:
    - Spacepoint sampling to handle large point clouds
    - Masking for MAE training
    - Data preprocessing and augmentation

    Args:
        base_dataset: The underlying HDF5 dataset
        sampler: Spacepoint sampler instance
        masking: Masking strategy instance
        config: Configuration dictionary
    """

    def __init__(
        self,
        file_paths: List[str] = None,
        load_from_cachefile: str = None,
        sampler: BaseSampler = None,
        masking: BaseMasking = None,
        max_spacepoints: int = 50000,
        mask_ratio: float = 0.75,
        config: dict = None
    ):
        # Create base dataset
        if load_from_cachefile is not None:
            self.base_dataset = LArMatchSimChHDF5Dataset(
                load_from_cachefile=load_from_cachefile
            )
        else:

            if type(file_paths) is str:
                # assume its a textfile with list of paths
                with open(file_paths,'r') as filelist:
                    files = filelist.readlines()
                    fpaths = []
                    for f in files:
                        fpaths.append( f.strip() )
            elif type(file_paths) is list:
                fpaths = file_paths
            else:
                raise ValueError("invalid type for file_paths parameter. given: ",type(file_paths))

            self.base_dataset = LArMatchSimChHDF5Dataset(
                file_paths=fpaths
            )

        # Create sampler and masking from config or use provided
        if config is not None:
            # Use create_sampler_chain to support both single samplers and chains
            self.sampler = create_sampler_chain(config) if sampler is None else sampler
            self.masking = create_masking_strategy(config) if masking is None else masking
            self.max_spacepoints = config.get('MAX_SPACEPOINTS', max_spacepoints)
            self.mask_ratio = config.get('MASK_RATIO', mask_ratio)
        else:
            from .spacepoint_sampler import ImportanceSampler
            from .masking import RandomMasking
            self.sampler = sampler or ImportanceSampler(max_spacepoints)
            self.masking = masking or RandomMasking(mask_ratio)
            self.max_spacepoints = max_spacepoints
            self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with sampling and preprocessing.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - positions: 3D positions (N, 3)
                - pixel_values: Target pixel values (N, 3)
                - instance_labels: Particle instance labels (N,)
                - is_true: Ghost vs true labels (N,)
                - ssnet_labels: SSNet class labels (N,)
                - keypoint_scores: Keypoint proximity scores (N, 6)
                - coord_0, coord_1, coord_2: Sparse image coordinates
                - feat_0, feat_1, feat_2: Sparse image features
                - query_coord_0, query_coord_1, query_coord_2: Query coordinates
        """
        # Get base data
        data = self.base_dataset[idx]

        # Apply sampling
        data = self.sampler(data)

        # Convert to tensors and prepare output
        output = self._prepare_output(data)

        return output

    def _prepare_output(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert numpy arrays to tensors and prepare for model input.

        Args:
            data: Sampled data dictionary

        Returns:
            Dictionary of tensors
        """
        output = {}

        # 3D positions
        if 'spacepoints' in data:
            output['positions'] = torch.from_numpy(data['spacepoints']).float()

        # Pixel values (reconstruction target) from edep
        if 'edep' in data:
            output['pixel_values'] = torch.from_numpy(data['edep']).float()

        # Instance labels (for contrastive loss)
        if 'trackid' in data:
            output['instance_labels'] = torch.from_numpy(data['trackid']).long()

        # Ghost vs true labels
        if 'larmatch_truth' in data:
            output['is_true'] = torch.from_numpy(data['larmatch_truth']).bool()

        # SSNet labels
        if 'ssnet_label' in data:
            output['ssnet_labels'] = torch.from_numpy(data['ssnet_label']).long()

        # Keypoint scores
        if 'kplabel' in data:
            output['keypoint_scores'] = torch.from_numpy(data['kplabel']).float()

        # Origin labels
        if 'origin' in data:
            output['origin'] = torch.from_numpy(data['origin']).long()

        # Sparse image data for each plane
        for p in range(3):
            coord_key = f'coord_{p}'
            feat_key = f'feat_{p}'
            query_key = f'query_coord_{p}'

            if coord_key in data:
                output[coord_key] = torch.from_numpy(data[coord_key]).long()
            if feat_key in data:
                output[feat_key] = torch.from_numpy(data[feat_key]).float()
            if query_key in data:
                output[query_key] = torch.from_numpy(data[query_key]).float()

        # Number of points
        output['npts'] = data['npts']
        output['idx'] = data['idx']

        return output


class MAECollator:
    """
    Collate function for MAE batching.

    Handles variable-length sequences and creates sparse tensor inputs.
    """

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary
        """
        batch_size = len(batch)

        # Find max points in batch
        max_pts = max(sample['npts'] for sample in batch)

        # Keys to batch (variable length)
        variable_keys = [
            'positions', 'pixel_values', 'instance_labels', 'is_true',
            'ssnet_labels', 'keypoint_scores', 'origin'
        ]

        output = {
            'batch_size': batch_size,
            'npts': [sample['npts'] for sample in batch],
            'idx': [sample['idx'] for sample in batch]
        }

        # Pad variable length tensors
        for key in variable_keys:
            if key not in batch[0]:
                continue

            sample_tensor = batch[0][key]
            if sample_tensor.dim() == 1:
                padded = torch.full((batch_size, max_pts), self.pad_value,
                                   dtype=sample_tensor.dtype)
            else:
                padded = torch.full((batch_size, max_pts, sample_tensor.shape[-1]),
                                   self.pad_value, dtype=sample_tensor.dtype)

            for b, sample in enumerate(batch):
                n = sample['npts']
                padded[b, :n] = sample[key]

            output[key] = padded

        # Create attention mask (True for valid positions)
        attention_mask = torch.zeros(batch_size, max_pts, dtype=torch.bool)
        for b, sample in enumerate(batch):
            attention_mask[b, :sample['npts']] = True
        output['attention_mask'] = attention_mask

        # Handle sparse image data - need special batching
        for p in range(3):
            coord_key = f'coord_{p}'
            feat_key = f'feat_{p}'
            query_key = f'query_coord_{p}'

            if coord_key in batch[0]:
                # Concatenate with batch index
                coords = []
                feats = []
                queries = []

                for b, sample in enumerate(batch):
                    coord = sample[coord_key]
                    # Add batch index as first column
                    batch_idx = torch.full((coord.shape[0], 1), b, dtype=coord.dtype)
                    coord_with_batch = torch.cat([batch_idx, coord], dim=1)
                    coords.append(coord_with_batch)

                    if feat_key in sample:
                        feats.append(sample[feat_key])

                    if query_key in sample:
                        query = sample[query_key]
                        # Update batch index in query coordinates
                        query[:, 0] = b
                        queries.append(query)

                output[coord_key] = torch.cat(coords, dim=0)
                if feats:
                    output[feat_key] = torch.cat(feats, dim=0)
                if queries:
                    output[query_key] = torch.cat(queries, dim=0)

        return output


def create_mae_dataloader(
    file_paths: List[str] = None,
    load_from_cachefile: str = None,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    config: dict = None
) -> DataLoader:
    """
    Create DataLoader for MAE training.

    Args:
        file_paths: List of HDF5 file paths
        load_from_cachefile: Path to cache file listing datasets
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        config: Configuration dictionary

    Returns:
        DataLoader instance
    """
    dataset = MAEDataset(
        file_paths=file_paths,
        load_from_cachefile=load_from_cachefile,
        config=config
    )

    collator = MAECollator()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )


def prepare_mae_batch(batch: Dict[str, torch.Tensor], device: str = 'cuda'):
    """
    Move batch to device and create sparse tensors.

    Args:
        batch: Collated batch dictionary
        device: Target device

    Returns:
        Prepared batch ready for model input
    """
    try:
        import MinkowskiEngine as ME
    except ImportError:
        raise ImportError("MinkowskiEngine required for sparse tensor creation")

    # Move tensors to device
    prepared = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.to(device)
        else:
            prepared[key] = value

    # Create sparse tensors for wireplane images
    wireplane_sparsetensors = []
    for p in range(3):
        coord_key = f'coord_{p}'
        feat_key = f'feat_{p}'

        if coord_key in prepared:
            coords = prepared[coord_key].int()
            feats = prepared[feat_key]

            sparse_tensor = ME.SparseTensor(
                features=feats,
                coordinates=coords
            )
            wireplane_sparsetensors.append(sparse_tensor)

    prepared['wireplane_sparsetensors'] = wireplane_sparsetensors

    # Prepare query coordinates
    query_v = []
    for p in range(3):
        query_key = f'query_coord_{p}'
        if query_key in prepared:
            query_v.append(prepared[query_key])
    prepared['query_v'] = query_v

    return prepared
