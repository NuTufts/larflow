#!/usr/bin/env python
"""
MAE Inference Script

Runs inference with a trained MAE model and saves auxiliary predictions
(ghost scores, ssnet logits, keypoint scores) to HDF5.

This script processes ALL spacepoints in each event by running inference
in chunks, then combining the results.
"""

import os
import sys
import argparse
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from larmatch.mae.models import SpacepointMAE
from larmatch.mae.utils.mae_engine import load_config, build_model
from larmatch.data.larmatch_simchhdf5_reader import LArMatchSimChHDF5Dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MAE Inference for Auxiliary Predictions')

    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file (same as training)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input file list (txt) or HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output HDF5 file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--chunk-size', type=int, default=5000,
                       help='Number of spacepoints to process at once (default: 5000)')
    parser.add_argument('--save-truth', action='store_true',
                       help='Also save ground truth labels for comparison')
    parser.add_argument('--entries', type=str, default=None,
                       help='Comma-separated list of entries to process, or "all" (default: all)')

    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    """
    Load trained MAE model from checkpoint.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Build model architecture
    model = build_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_model' in checkpoint:
        state_dict = checkpoint['state_model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'iteration' in checkpoint:
        print(f"  Iteration: {checkpoint['iteration']}")
    if 'loss' in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.4f}")

    return model


def load_entry_data(base_dataset, entry_idx):
    """
    Load all data for an entry (without sampling).

    Args:
        base_dataset: LArMatchSimChHDF5Dataset instance
        entry_idx: Entry index

    Returns:
        Dictionary with all entry data
    """
    return base_dataset[entry_idx]


def prepare_chunk_batch(entry_data, start_idx, end_idx, device):
    """
    Prepare a chunk of spacepoints for inference.

    Args:
        entry_data: Full entry data dictionary
        start_idx: Start index of chunk
        end_idx: End index of chunk
        device: Device to move tensors to

    Returns:
        Prepared batch dictionary
    """
    try:
        import MinkowskiEngine as ME
    except ImportError:
        raise ImportError("MinkowskiEngine required for sparse tensor creation")

    chunk_size = end_idx - start_idx

    # Extract chunk of spacepoints
    positions = torch.from_numpy(
        entry_data['spacepoints'][start_idx:end_idx]
    ).float().unsqueeze(0).to(device)  # (1, chunk_size, 3)

    # Create sparse tensors for wireplane images (full images, not chunked)
    wireplane_sparsetensors = []
    for p in range(3):
        coords = torch.from_numpy(entry_data[f'coord_{p}']).int()
        feats = torch.from_numpy(entry_data[f'feat_{p}']).float()

        # Add batch dimension to coordinates
        batch_idx = torch.zeros((coords.shape[0], 1), dtype=torch.int)
        coords_with_batch = torch.cat([batch_idx, coords], dim=1).to(device)
        feats = feats.to(device)

        sparse_tensor = ME.SparseTensor(
            features=feats,
            coordinates=coords_with_batch
        )
        wireplane_sparsetensors.append(sparse_tensor)

    # Prepare query coordinates for this chunk
    query_v = []
    for p in range(3):
        query_coord = entry_data[f'query_coord_{p}'][start_idx:end_idx].copy()
        query_coord[:, 0] = 0  # batch index
        query_v.append(torch.from_numpy(query_coord).float().to(device))

    batch = {
        'wireplane_sparsetensors': wireplane_sparsetensors,
        'query_v': query_v,
        'positions': positions,
    }

    return batch


def run_inference_on_chunk(model, batch):
    """
    Run inference on a chunk of spacepoints.

    Args:
        model: Trained MAE model
        batch: Prepared batch dictionary

    Returns:
        Dictionary with predictions for this chunk
    """
    with torch.no_grad():
        # Get encoder output for all spacepoints in chunk (no masking)
        encoded = model.get_encoder_output(
            batch['wireplane_sparsetensors'],
            batch['query_v'],
            batch['positions']
        )

        # Run auxiliary heads
        aux_outputs = model.forward_auxiliary(encoded)

        # Convert logits to probabilities/predictions
        results = {}

        # Ghost classification (2 classes: ghost=0, true=1)
        if 'ghost_logits' in aux_outputs:
            ghost_logits = aux_outputs['ghost_logits']
            ghost_probs = F.softmax(ghost_logits, dim=-1)
            results['ghost_score'] = ghost_probs[0, :, 1].cpu().numpy()
            results['ghost_pred'] = ghost_logits[0].argmax(dim=-1).cpu().numpy()

        # SSNet classification
        if 'ssnet_logits' in aux_outputs:
            ssnet_logits = aux_outputs['ssnet_logits']
            ssnet_probs = F.softmax(ssnet_logits, dim=-1)
            results['ssnet_probs'] = ssnet_probs[0].cpu().numpy()
            results['ssnet_pred'] = ssnet_logits[0].argmax(dim=-1).cpu().numpy()

        # Keypoint scores (regression)
        if 'keypoint_scores' in aux_outputs:
            results['keypoint_scores'] = aux_outputs['keypoint_scores'][0].cpu().numpy()

        return results


def process_entry(model, entry_data, chunk_size, device):
    """
    Process all spacepoints in an entry by chunking.

    Args:
        model: Trained MAE model
        entry_data: Full entry data dictionary
        chunk_size: Number of points per chunk
        device: Device for inference

    Returns:
        Dictionary with predictions for all spacepoints
    """
    npts = entry_data['npts']
    n_chunks = (npts + chunk_size - 1) // chunk_size

    # Initialize result arrays
    results = {
        'ghost_score': np.zeros(npts, dtype=np.float32),
        'ghost_pred': np.zeros(npts, dtype=np.int64),
        'ssnet_probs': None,  # Will initialize after first chunk
        'ssnet_pred': np.zeros(npts, dtype=np.int64),
        'keypoint_scores': None,  # Will initialize after first chunk
    }

    # Process each chunk
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks", leave=False):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, npts)

        # Prepare batch for this chunk
        batch = prepare_chunk_batch(entry_data, start_idx, end_idx, device)

        # Run inference
        chunk_results = run_inference_on_chunk(model, batch)

        # Store results
        results['ghost_score'][start_idx:end_idx] = chunk_results['ghost_score']
        results['ghost_pred'][start_idx:end_idx] = chunk_results['ghost_pred']
        results['ssnet_pred'][start_idx:end_idx] = chunk_results['ssnet_pred']

        # Initialize arrays on first chunk
        if results['ssnet_probs'] is None:
            n_ssnet_classes = chunk_results['ssnet_probs'].shape[1]
            results['ssnet_probs'] = np.zeros((npts, n_ssnet_classes), dtype=np.float32)

        if results['keypoint_scores'] is None:
            n_kp_types = chunk_results['keypoint_scores'].shape[1]
            results['keypoint_scores'] = np.zeros((npts, n_kp_types), dtype=np.float32)

        results['ssnet_probs'][start_idx:end_idx] = chunk_results['ssnet_probs']
        results['keypoint_scores'][start_idx:end_idx] = chunk_results['keypoint_scores']

        # Clear GPU memory
        del batch
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    return results


def create_output_hdf5(output_path, config):
    """
    Create output HDF5 file with appropriate structure.

    Args:
        output_path: Path to output file
        config: Configuration dictionary

    Returns:
        Opened HDF5 file handle
    """
    f = h5py.File(output_path, 'w')

    # Store config as attributes
    f.attrs['model_config'] = str(config)
    f.attrs['num_ssnet_classes'] = config.get('NUM_SSNET_CLASSES', 5)
    f.attrs['num_keypoint_types'] = config.get('NUM_KEYPOINT_TYPES', 6)

    return f


def save_entry_to_hdf5(f, entry_idx, results, entry_data, save_truth=False):
    """
    Save predictions for one entry to HDF5.

    Args:
        f: HDF5 file handle
        entry_idx: Entry index
        results: Prediction results dictionary
        entry_data: Original entry data (for truth labels and positions)
        save_truth: Whether to save ground truth labels
    """
    grp = f.create_group(f'entry_{entry_idx}')

    npts = entry_data['npts']
    grp.attrs['npts'] = npts

    # Save positions
    grp.create_dataset('positions', data=entry_data['spacepoints'],
                      compression='gzip', compression_opts=4)

    # Save ghost predictions
    grp.create_dataset('ghost_score', data=results['ghost_score'],
                      compression='gzip', compression_opts=4)
    grp.create_dataset('ghost_pred', data=results['ghost_pred'],
                      compression='gzip', compression_opts=4)

    # Save SSNet predictions
    grp.create_dataset('ssnet_probs', data=results['ssnet_probs'],
                      compression='gzip', compression_opts=4)
    grp.create_dataset('ssnet_pred', data=results['ssnet_pred'],
                      compression='gzip', compression_opts=4)

    # Save keypoint scores
    grp.create_dataset('keypoint_scores', data=results['keypoint_scores'],
                      compression='gzip', compression_opts=4)

    # Save ground truth if requested
    if save_truth:
        # Ghost truth (larmatch_truth: 0=ghost, 1=true)
        grp.create_dataset('truth_is_true', data=entry_data['larmatch_truth'].astype(bool),
                          compression='gzip', compression_opts=4)

        # SSNet truth
        grp.create_dataset('truth_ssnet_labels', data=entry_data['ssnet_label'],
                          compression='gzip', compression_opts=4)

        # Keypoint truth
        grp.create_dataset('truth_keypoint_scores', data=entry_data['kplabel'],
                          compression='gzip', compression_opts=4)

        # Track ID (particle instance)
        grp.create_dataset('truth_trackid', data=entry_data['trackid'],
                          compression='gzip', compression_opts=4)

        # Origin (bg=0, nu=1, cosmic=2)
        if 'origin' in entry_data:
            grp.create_dataset('truth_origin', data=entry_data['origin'],
                              compression='gzip', compression_opts=4)


def main():
    """Main inference function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    device = torch.device(args.device)

    # Load model
    print("Loading model...")
    model = load_model(config, args.checkpoint, device)

    # Create base dataset (without sampling)
    print("Loading dataset...")
    if args.input.endswith('.txt'):
        with open(args.input, 'r') as f:
            file_paths = [line.strip() for line in f if line.strip()]
    else:
        file_paths = [args.input]

    base_dataset = LArMatchSimChHDF5Dataset(file_paths=file_paths)
    n_entries = len(base_dataset)
    print(f"Dataset contains {n_entries} entries")

    # Determine which entries to process
    if args.entries is None or args.entries.lower() == 'all':
        entries_to_process = list(range(n_entries))
    else:
        entries_to_process = [int(e.strip()) for e in args.entries.split(',')]

    print(f"Will process {len(entries_to_process)} entries")

    # Create output HDF5 file
    print(f"Creating output file: {args.output}")
    output_file = create_output_hdf5(args.output, config)

    # Process each entry
    print("Running inference...")
    for entry_idx in tqdm(entries_to_process, desc="Processing entries"):
        # Load all data for this entry
        entry_data = load_entry_data(base_dataset, entry_idx)
        npts = entry_data['npts']

        print(f"\nEntry {entry_idx}: {npts} spacepoints")

        # Process in chunks
        results = process_entry(model, entry_data, args.chunk_size, device)

        # Save to HDF5
        save_entry_to_hdf5(output_file, entry_idx, results, entry_data,
                          save_truth=args.save_truth)

    # Close output file
    output_file.close()

    print(f"\nInference complete!")
    print(f"Processed {len(entries_to_process)} entries")
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
