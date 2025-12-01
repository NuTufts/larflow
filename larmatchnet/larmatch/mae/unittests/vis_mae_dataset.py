#!/bin/env python3
"""
Visualize MAE Dataset Training Data

This script loads data through the MAEDataset class (exactly as the training
script would) and visualizes the spacepoints with various coloring options.

Supports coloring by:
- ssnet: SSNet particle class labels
- keypoint scores: kpnu, kptrackstart, kptrackend, kpshower, kpmichel, kpdelta
- edep: Target energy deposition values (MAE prediction target)
- trackid: Particle instance ID
- is_true: Ghost vs true spacepoints

Also supports showing masked points vs visible points to verify masking behavior.

Configuration can be provided via:
1. Command line arguments (for quick testing)
2. YAML config file (for reproducible setups)
3. Combination of both (command line overrides config file)

Usage:
    # Using command line arguments
    python vis_mae_dataset.py -i /path/to/file.h5 -e 0 --colorby ssnet
    python vis_mae_dataset.py -c cache_file.txt -e 0 --colorby edep --show-mask

    # Using config file
    python vis_mae_dataset.py --config config.yaml -e 0 --colorby ssnet

    # Using spatial_box sampler to view a specific region (fixed mode)
    python vis_mae_dataset.py -i file.h5 -e 0 --sampler-type spatial_box \
        --box-min 0 -117 0 --box-max 128 0 518

    # Using spatial_box sampler with random box placement
    python vis_mae_dataset.py -i file.h5 -e 0 --sampler-type spatial_box \
        --random-box --box-size 100 100 200 --show-box

    # Config file with command line overrides
    python vis_mae_dataset.py --config config.yaml -e 5 --colorby trackid
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Colorby options
COLORBY_OPTIONS = [
    'ssnet',
    'edep',
    'trackid',
    'kpnu', 'kptrackstart', 'kptrackend', 'kpshower', 'kpmichel', 'kpdelta',
    'is_true'
]

# Keypoint index mapping
KPINDEX = {
    'kpnu': 0,
    'kptrackstart': 1,
    'kptrackend': 2,
    'kpshower': 3,
    'kpmichel': 4,
    'kpdelta': 5
}

# SSNet class colors
SSNET_CLASS_COLORS = {
    -1: 'rgba(50,50,50,1.0)',    # background
    0: 'rgba(255,0,0,1.0)',     # electron
    1: 'rgba(200,125,0,1)',     # photon
    2: 'rgba(0,0,255,1)',       # muon
    3: 'rgba(0,125,255,1)',     # proton
    4: 'rgba(125,0,255,1)',     # pion/kaon
    5: 'rgba(125,125,0,1)',     # other
}

SSNET_CLASS_NAMES = {
    -1: 'Background',
    0: 'Electron',
    1: 'Photon',
    2: 'Muon',
    3: 'Proton',
    4: 'Pion/Kaon',
    5: 'Other',
}

# Default MicroBooNE detector parameters
DEFAULT_DETECTOR_ORIGIN = [0.0, -117.0, 0.0]
DEFAULT_DETECTOR_EXTENT = [256.0, 234.0, 1036.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MAE Dataset training data using Plotly/Dash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization with importance sampler
  python vis_mae_dataset.py -i data.h5 -e 0 --colorby ssnet

  # View a spatial box region
  python vis_mae_dataset.py -i data.h5 -e 0 --sampler-type spatial_box \\
      --box-min 0 -117 0 --box-max 128 0 518 --colorby is_true

  # Use config file
  python vis_mae_dataset.py --config vis_config.yaml -e 0

  # Show sampling box outline
  python vis_mae_dataset.py -i data.h5 -e 0 --sampler-type spatial_box \\
      --box-min 64 -60 200 --box-max 192 60 800 --show-box
        """
    )

    # Config file (optional, settings can be overridden by command line)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )

    # Data source - either cache file or direct HDF5 files
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "-c", "--cache-file",
        type=str,
        help="Path to cache file listing HDF5 dataset files"
    )
    data_group.add_argument(
        "-i", "--input-files",
        type=str,
        nargs='+',
        help="Direct paths to HDF5 files"
    )

    # Entry selection
    parser.add_argument(
        "-e", "--entry",
        type=int,
        default=0,
        help="Entry/event index to visualize (default: 0)"
    )

    # Visualization options
    parser.add_argument(
        "--colorby",
        type=str,
        default=None,
        choices=COLORBY_OPTIONS,
        help=f"Color mode. Options: {COLORBY_OPTIONS} (default: ssnet)"
    )
    parser.add_argument(
        "--show-mask",
        action='store_true',
        default=False,
        help="Show masked points alongside visible points (different marker)"
    )
    parser.add_argument(
        "--show-box",
        action='store_true',
        default=False,
        help="Show sampling box outline (for spatial_box sampler)"
    )

    # Sampler options
    parser.add_argument(
        "--sampler-type",
        type=str,
        default=None,
        choices=['random', 'importance', 'stratified', 'particle', 'spatial', 'spatial_box'],
        help="Sampler type (default: importance)"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum spacepoints after sampling (default: 50000)"
    )

    # Spatial box sampler options
    parser.add_argument(
        "--box-min",
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=None,
        help="Minimum corner of sampling box (x y z) in cm (fixed mode)"
    )
    parser.add_argument(
        "--box-max",
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=None,
        help="Maximum corner of sampling box (x y z) in cm (fixed mode)"
    )
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=None,
        help="Size of sampling box (x y z) in cm (for random box mode)"
    )
    parser.add_argument(
        "--random-box",
        action='store_true',
        default=False,
        help="Enable random box mode: randomly place box within detector"
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=None,
        help="Minimum points required in box (random mode only). Resamples if fewer."
    )
    parser.add_argument(
        "--max-resample-attempts",
        type=int,
        default=None,
        help="Maximum resample attempts to find box with min-points (default: 100)"
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        default=None,
        help="Padding around sampling box in cm (default: 0)"
    )
    parser.add_argument(
        "--detector-origin",
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=None,
        help="Detector origin (x y z) in cm"
    )
    parser.add_argument(
        "--detector-extent",
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=None,
        help="Detector extent (x y z) in cm"
    )

    # Masking options
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Mask ratio for MAE masking (default: 0.75)"
    )
    parser.add_argument(
        "--masking-strategy",
        type=str,
        default=None,
        choices=['random', 'spatial', 'particle', 'ghost_aware'],
        help="Masking strategy (default: random)"
    )

    # Importance sampler options
    parser.add_argument(
        "--true-weight",
        type=float,
        default=None,
        help="Weight for true spacepoints in importance sampling"
    )
    parser.add_argument(
        "--min-true-fraction",
        type=float,
        default=None,
        help="Minimum fraction of true points in sample"
    )

    # Server options
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for Dash server (default: 8050)"
    )
    parser.add_argument(
        "--no-detector",
        action='store_true',
        default=False,
        help="Don't show detector outline"
    )

    return parser.parse_args()


def load_config_file(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_config(args):
    """
    Build configuration dictionary from config file and command line args.

    Command line arguments override config file settings.
    """
    # Start with defaults
    config = {
        'SAMPLER_TYPE': 'importance',
        'MAX_SPACEPOINTS': 50000,
        'MASK_RATIO': 0.75,
        'MASKING_STRATEGY': 'random',
        'MIN_TRUE_FRACTION': 0.3,
        'TRUE_WEIGHT': 5.0,
        'BOUNDARY_WEIGHT': 2.0,
        'KEYPOINT_WEIGHT': 3.0,
        'TARGET_GHOST_RATIO': 2.0,
        'GRID_SIZE': [10, 10, 10],
        # Spatial box defaults
        'BOX_MIN': DEFAULT_DETECTOR_ORIGIN.copy(),
        'BOX_MAX': [
            DEFAULT_DETECTOR_ORIGIN[0] + DEFAULT_DETECTOR_EXTENT[0],
            DEFAULT_DETECTOR_ORIGIN[1] + DEFAULT_DETECTOR_EXTENT[1],
            DEFAULT_DETECTOR_ORIGIN[2] + DEFAULT_DETECTOR_EXTENT[2]
        ],
        'BOX_SIZE': None,  # For random box mode
        'RANDOM_BOX_MODE': False,
        'MIN_POINTS_IN_BOX': 0,  # Minimum points in random box mode
        'MAX_RESAMPLE_ATTEMPTS': 100,
        'DETECTOR_ORIGIN': DEFAULT_DETECTOR_ORIGIN.copy(),
        'DETECTOR_EXTENT': DEFAULT_DETECTOR_EXTENT.copy(),
        'BOX_PADDING': 0.0,
        'BOX_FALLBACK_TO_RANDOM': True,
    }

    # Load from config file if provided
    if args.config:
        file_config = load_config_file(args.config)
        config.update(file_config)

    # Override with command line arguments
    if args.sampler_type is not None:
        config['SAMPLER_TYPE'] = args.sampler_type
    if args.max_points is not None:
        config['MAX_SPACEPOINTS'] = args.max_points
    if args.mask_ratio is not None:
        config['MASK_RATIO'] = args.mask_ratio
    if args.masking_strategy is not None:
        config['MASKING_STRATEGY'] = args.masking_strategy
    if args.true_weight is not None:
        config['TRUE_WEIGHT'] = args.true_weight
    if args.min_true_fraction is not None:
        config['MIN_TRUE_FRACTION'] = args.min_true_fraction

    # Spatial box options
    if args.box_min is not None:
        config['BOX_MIN'] = list(args.box_min)
    if args.box_max is not None:
        config['BOX_MAX'] = list(args.box_max)
    if args.box_size is not None:
        config['BOX_SIZE'] = list(args.box_size)
    if args.random_box:
        config['RANDOM_BOX_MODE'] = True
    if args.min_points is not None:
        config['MIN_POINTS_IN_BOX'] = args.min_points
    if args.max_resample_attempts is not None:
        config['MAX_RESAMPLE_ATTEMPTS'] = args.max_resample_attempts
    if args.box_padding is not None:
        config['BOX_PADDING'] = args.box_padding
    if args.detector_origin is not None:
        config['DETECTOR_ORIGIN'] = list(args.detector_origin)
    if args.detector_extent is not None:
        config['DETECTOR_EXTENT'] = list(args.detector_extent)

    # Data source from args or config
    if args.cache_file:
        config['CACHE_FILE'] = args.cache_file
    if args.input_files:
        config['INPUT_FILES'] = args.input_files

    # Visualization options
    config['COLORBY'] = args.colorby if args.colorby else config.get('COLORBY', 'ssnet')
    config['SHOW_MASK'] = args.show_mask or config.get('SHOW_MASK', False)
    config['SHOW_BOX'] = args.show_box or config.get('SHOW_BOX', False)
    config['ENTRY'] = args.entry
    config['PORT'] = args.port
    config['NO_DETECTOR'] = args.no_detector

    return config


def load_mae_data(config):
    """
    Load data through MAEDataset class.

    Returns:
        Tuple of (sample, masking, config, sampler)
        sampler is returned so we can access random box coordinates if needed
    """
    # Add path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mae_dir = os.path.dirname(script_dir)
    larmatch_dir = os.path.dirname(mae_dir)
    if larmatch_dir not in sys.path:
        sys.path.insert(0, larmatch_dir)

    from mae.data.mae_dataset import MAEDataset
    from mae.data.masking import create_masking_strategy

    # Determine data source
    cache_file = config.get('CACHE_FILE')
    input_files = config.get('INPUT_FILES')

    if not cache_file and not input_files:
        raise ValueError("Must provide either --cache-file, --input-files, or INPUT_FILES in config")

    # Create dataset
    if cache_file:
        dataset = MAEDataset(
            load_from_cachefile=cache_file,
            config=config
        )
    else:
        dataset = MAEDataset(
            file_paths=input_files,
            config=config
        )

    print(f"Dataset loaded with {len(dataset)} entries")

    entry = config['ENTRY']
    if entry >= len(dataset):
        print(f"Error: Entry {entry} out of range (max: {len(dataset)-1})")
        sys.exit(1)

    # Get sample (this applies sampling)
    sample = dataset[entry]

    # Create masking
    masking = create_masking_strategy(config)

    # Get sampler for accessing random box coordinates
    sampler = dataset.sampler if hasattr(dataset, 'sampler') else None

    return sample, masking, config, sampler


def create_mask(sample, masking):
    """
    Apply masking to get visible and masked point indices.

    Args:
        sample: Dataset sample dictionary
        masking: Masking strategy instance

    Returns:
        Boolean tensor: True = masked, False = visible
    """
    npts = sample['npts']

    # Get additional info for masking strategies that need it
    kwargs = {}
    if 'positions' in sample:
        kwargs['positions'] = sample['positions']
    if 'instance_labels' in sample:
        kwargs['instance_labels'] = sample['instance_labels']
    if 'is_true' in sample:
        kwargs['is_true'] = sample['is_true']

    mask = masking(npts, device='cpu', **kwargs)
    return mask


def tensor_to_numpy(tensor):
    """Convert PyTorch tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def get_spatial_box_sampler(sampler):
    """
    Find the SpatialBoxSampler in a sampler (or sampler chain).

    Args:
        sampler: A sampler instance (may be SamplerChain or single sampler)

    Returns:
        SpatialBoxSampler instance or None
    """
    if sampler is None:
        return None

    # Check if it's a SamplerChain
    if hasattr(sampler, 'samplers'):
        # It's a chain - search for SpatialBoxSampler
        for s in sampler.samplers:
            if s.__class__.__name__ == 'SpatialBoxSampler':
                return s
        return None
    elif sampler.__class__.__name__ == 'SpatialBoxSampler':
        return sampler
    else:
        return None


def build_box_traces(config, sampler=None):
    """
    Build traces showing the sampling box outline.

    Args:
        config: Configuration dictionary
        sampler: Optional sampler instance (used to get random box coordinates)

    Returns:
        List of Plotly trace dictionaries for the box edges
    """
    # Find the spatial box sampler (might be in a chain)
    box_sampler = get_spatial_box_sampler(sampler)

    # Get box coordinates - from sampler if in random mode, otherwise from config
    if box_sampler is not None and hasattr(box_sampler, 'get_last_box'):
        # Get the actual box used (works for both fixed and random mode)
        box_min, box_max = box_sampler.get_last_box()
        box_min = np.array(box_min)
        box_max = np.array(box_max)
    else:
        box_min = np.array(config['BOX_MIN'])
        box_max = np.array(config['BOX_MAX'])

    padding = config.get('BOX_PADDING', 0.0)

    # Apply padding
    box_min = box_min - padding
    box_max = box_max + padding

    # Define the 8 corners of the box
    corners = np.array([
        [box_min[0], box_min[1], box_min[2]],
        [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_max[1], box_min[2]],
        [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]],
        [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_max[2]],
        [box_min[0], box_max[1], box_max[2]],
    ])

    # Define edges (pairs of corner indices)
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    traces = []
    for i, (start, end) in enumerate(edges):
        traces.append({
            "type": "scatter3d",
            "x": [corners[start, 0], corners[end, 0]],
            "y": [corners[start, 1], corners[end, 1]],
            "z": [corners[start, 2], corners[end, 2]],
            "mode": "lines",
            "name": "Sampling Box" if i == 0 else None,
            "showlegend": i == 0,
            "line": {
                "color": "rgba(255, 255, 0, 1.0)",
                "width": 3
            }
        })

    return traces


def build_plots(sample, mask, config):
    """
    Build Plotly traces for visualization.

    Args:
        sample: Dataset sample dictionary with tensors
        mask: Boolean mask tensor (True = masked)
        config: Configuration dictionary

    Returns:
        List of Plotly trace dictionaries
    """
    plots = []

    # Convert tensors to numpy
    positions = tensor_to_numpy(sample['positions'])
    npts = sample['npts']

    # Get visible and masked indices
    mask_np = tensor_to_numpy(mask)
    visible_idx = ~mask_np
    masked_idx = mask_np

    opacity = 0.8
    marker_size = 2.0
    masked_marker_size = 1.5

    colorby = config['COLORBY']
    show_mask = config['SHOW_MASK']

    # Build hover template
    hovertemplate = """
<b>x</b>: %{x:.1f}<br>
<b>y</b>: %{y:.1f}<br>
<b>z</b>: %{z:.1f}<br>
<b>idx</b>: %{customdata[0]:d}<br>
"""

    # Base customdata is just the index
    customdata = np.arange(npts).reshape(-1, 1)

    # Color by SSNet labels
    if colorby == 'ssnet':
        if 'ssnet_labels' not in sample:
            print("Warning: ssnet_labels not in sample, falling back to single color")
            plots.append(_single_color_plot(
                positions, visible_idx, "Visible", opacity, marker_size,
                'rgba(0,255,0,1)', hovertemplate, customdata
            ))
        else:
            ssnet_labels = tensor_to_numpy(sample['ssnet_labels'])

            for iclass in range(7):
                class_mask = (ssnet_labels == iclass) & visible_idx
                if np.sum(class_mask) == 0:
                    continue

                color = SSNET_CLASS_COLORS[iclass]
                name = f"{SSNET_CLASS_NAMES[iclass]} (visible)"

                plots.append({
                    "type": "scatter3d",
                    "x": positions[class_mask, 0],
                    "y": positions[class_mask, 1],
                    "z": positions[class_mask, 2],
                    "mode": "markers",
                    "name": name,
                    "hovertemplate": hovertemplate,
                    "customdata": customdata[class_mask],
                    "marker": {
                        "color": color,
                        "opacity": opacity,
                        "size": marker_size
                    }
                })

            # Add masked points if requested
            if show_mask:
                for iclass in range(7):
                    class_mask = (ssnet_labels == iclass) & masked_idx
                    if np.sum(class_mask) == 0:
                        continue

                    # Use lighter/different color for masked
                    color = SSNET_CLASS_COLORS[iclass].replace('1.0)', '0.3)')
                    color = color.replace(',1)', ',0.3)')
                    name = f"{SSNET_CLASS_NAMES[iclass]} (masked)"

                    plots.append({
                        "type": "scatter3d",
                        "x": positions[class_mask, 0],
                        "y": positions[class_mask, 1],
                        "z": positions[class_mask, 2],
                        "mode": "markers",
                        "name": name,
                        "hovertemplate": hovertemplate,
                        "customdata": customdata[class_mask],
                        "marker": {
                            "color": color,
                            "opacity": 0.3,
                            "size": masked_marker_size,
                            "symbol": "diamond"
                        }
                    })

    # Color by edep (MAE prediction target)
    elif colorby == 'edep':
        if 'pixel_values' not in sample:
            print("Warning: pixel_values (edep) not in sample")
            return plots

        edep = tensor_to_numpy(sample['pixel_values'])

        # Use sum or max of edep across planes for coloring
        if edep.ndim > 1:
            edep_color = edep.sum(axis=1)  # Sum across 3 planes
        else:
            edep_color = edep

        # Visible points
        plots.append({
            "type": "scatter3d",
            "x": positions[visible_idx, 0],
            "y": positions[visible_idx, 1],
            "z": positions[visible_idx, 2],
            "mode": "markers",
            "name": "Edep (visible)",
            "hovertemplate": hovertemplate + "<b>edep</b>: %{marker.color:.3f}<br>",
            "customdata": customdata[visible_idx],
            "marker": {
                "color": edep_color[visible_idx],
                "opacity": opacity,
                "size": marker_size,
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": np.percentile(edep_color, 95),
                "colorbar": {"title": "Edep (MeV)"}
            }
        })

        # Masked points
        if show_mask:
            plots.append({
                "type": "scatter3d",
                "x": positions[masked_idx, 0],
                "y": positions[masked_idx, 1],
                "z": positions[masked_idx, 2],
                "mode": "markers",
                "name": "Edep (masked)",
                "hovertemplate": hovertemplate + "<b>edep</b>: %{marker.color:.3f}<br>",
                "customdata": customdata[masked_idx],
                "marker": {
                    "color": edep_color[masked_idx],
                    "opacity": 0.3,
                    "size": masked_marker_size,
                    "colorscale": "Viridis",
                    "cmin": 0.0,
                    "cmax": np.percentile(edep_color, 95),
                    "symbol": "diamond"
                }
            })

    # Color by trackid (particle instance)
    elif colorby == 'trackid':
        if 'instance_labels' not in sample:
            print("Warning: instance_labels (trackid) not in sample")
            return plots

        trackids = tensor_to_numpy(sample['instance_labels'])
        unique_ids = np.unique(trackids)

        np.random.seed(42)  # For reproducible colors

        for tid in unique_ids:
            if tid <= 0:
                continue

            tid_mask = (trackids == tid) & visible_idx
            if np.sum(tid_mask) == 0:
                continue

            xcolor = np.random.randint(0, 255, 3)
            scolor = f'rgba({xcolor[0]},{xcolor[1]},{xcolor[2]},1)'

            plots.append({
                "type": "scatter3d",
                "x": positions[tid_mask, 0],
                "y": positions[tid_mask, 1],
                "z": positions[tid_mask, 2],
                "mode": "markers",
                "name": f"tid[{tid}] (visible)",
                "hovertemplate": hovertemplate,
                "customdata": customdata[tid_mask],
                "marker": {
                    "color": scolor,
                    "opacity": opacity,
                    "size": marker_size
                }
            })

        # Masked points - show in gray
        if show_mask:
            plots.append({
                "type": "scatter3d",
                "x": positions[masked_idx, 0],
                "y": positions[masked_idx, 1],
                "z": positions[masked_idx, 2],
                "mode": "markers",
                "name": "All masked",
                "hovertemplate": hovertemplate,
                "customdata": customdata[masked_idx],
                "marker": {
                    "color": 'rgba(128,128,128,0.3)',
                    "opacity": 0.3,
                    "size": masked_marker_size,
                    "symbol": "diamond"
                }
            })

    # Color by keypoint scores
    elif colorby in KPINDEX:
        if 'keypoint_scores' not in sample:
            print(f"Warning: keypoint_scores not in sample")
            return plots

        kp_scores = tensor_to_numpy(sample['keypoint_scores'])
        kp_idx = KPINDEX[colorby]

        if kp_scores.ndim > 1:
            kp_color = kp_scores[:, kp_idx]
        else:
            kp_color = kp_scores

        kp_names = {
            'kpnu': 'Neutrino Vertex',
            'kptrackstart': 'Track Start',
            'kptrackend': 'Track End',
            'kpshower': 'Shower Start',
            'kpmichel': 'Michel Start',
            'kpdelta': 'Delta Start'
        }

        # Visible points
        plots.append({
            "type": "scatter3d",
            "x": positions[visible_idx, 0],
            "y": positions[visible_idx, 1],
            "z": positions[visible_idx, 2],
            "mode": "markers",
            "name": f"{kp_names[colorby]} (visible)",
            "hovertemplate": hovertemplate + f"<b>{colorby}</b>: " + "%{marker.color:.3f}<br>",
            "customdata": customdata[visible_idx],
            "marker": {
                "color": kp_color[visible_idx],
                "opacity": opacity,
                "size": marker_size,
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {"title": f"{kp_names[colorby]} Score"}
            }
        })

        # Masked points
        if show_mask:
            plots.append({
                "type": "scatter3d",
                "x": positions[masked_idx, 0],
                "y": positions[masked_idx, 1],
                "z": positions[masked_idx, 2],
                "mode": "markers",
                "name": f"{kp_names[colorby]} (masked)",
                "hovertemplate": hovertemplate + f"<b>{colorby}</b>: " + "%{marker.color:.3f}<br>",
                "customdata": customdata[masked_idx],
                "marker": {
                    "color": kp_color[masked_idx],
                    "opacity": 0.3,
                    "size": masked_marker_size,
                    "colorscale": "Viridis",
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "symbol": "diamond"
                }
            })

    # Color by is_true (ghost vs true)
    elif colorby == 'is_true':
        if 'is_true' not in sample:
            print("Warning: is_true not in sample")
            return plots

        is_true = tensor_to_numpy(sample['is_true'])
        print("is_true.sum=",is_true.sum()," shape=",is_true.shape)

        # True points (visible)
        true_visible = is_true & visible_idx
        ghost_visible = (~is_true) & visible_idx

        plots.append({
            "type": "scatter3d",
            "x": positions[true_visible, 0],
            "y": positions[true_visible, 1],
            "z": positions[true_visible, 2],
            "mode": "markers",
            "name": f"True (visible, n={true_visible.sum()})",
            "hovertemplate": hovertemplate,
            "customdata": customdata[true_visible],
            "marker": {
                "color": 'rgba(0,255,0,1)',
                "opacity": opacity*0.5,
                "size": marker_size
            }
        })

        plots.append({
            "type": "scatter3d",
            "x": positions[ghost_visible, 0],
            "y": positions[ghost_visible, 1],
            "z": positions[ghost_visible, 2],
            "mode": "markers",
            "name": f"Ghost (visible, n={ghost_visible.sum()})",
            "hovertemplate": hovertemplate,
            "customdata": customdata[ghost_visible],
            "marker": {
                "color": 'rgba(255,0,0,1)',
                "opacity": opacity*0.5,
                "size": marker_size*0.5
            }
        })

        # Masked points
        if show_mask:
            true_masked = is_true & masked_idx
            ghost_masked = (~is_true) & masked_idx

            plots.append({
                "type": "scatter3d",
                "x": positions[true_masked, 0],
                "y": positions[true_masked, 1],
                "z": positions[true_masked, 2],
                "mode": "markers",
                "name": f"True (masked, n={true_masked.sum()})",
                "hovertemplate": hovertemplate,
                "customdata": customdata[true_masked],
                "marker": {
                    "color": 'rgba(0,255,0,0.3)',
                    "opacity": 0.3,
                    "size": masked_marker_size,
                    "symbol": "diamond"
                }
            })

            plots.append({
                "type": "scatter3d",
                "x": positions[ghost_masked, 0],
                "y": positions[ghost_masked, 1],
                "z": positions[ghost_masked, 2],
                "mode": "markers",
                "name": f"Ghost (masked, n={ghost_masked.sum()})",
                "hovertemplate": hovertemplate,
                "customdata": customdata[ghost_masked],
                "marker": {
                    "color": 'rgba(255,0,0,0.3)',
                    "opacity": 0.3,
                    "size": masked_marker_size,
                    "symbol": "diamond"
                }
            })

    return plots


def _single_color_plot(positions, mask, name, opacity, marker_size, color,
                       hovertemplate, customdata):
    """Helper to create single-color scatter plot."""
    return {
        "type": "scatter3d",
        "x": positions[mask, 0],
        "y": positions[mask, 1],
        "z": positions[mask, 2],
        "mode": "markers",
        "name": name,
        "hovertemplate": hovertemplate,
        "customdata": customdata[mask],
        "marker": {
            "color": color,
            "opacity": opacity,
            "size": marker_size
        }
    }


def run_app(traces, config, sample, mask):
    """Launch Dash app for visualization."""

    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    server = app.server

    axis_template = {
        "showbackground": True,
        "backgroundcolor": "#141414",
        "gridcolor": "rgb(255, 255, 255)",
        "zerolinecolor": "rgb(255, 255, 255)",
    }

    # Compute statistics for display
    npts = sample['npts']
    n_visible = int((~tensor_to_numpy(mask)).sum())
    n_masked = int(tensor_to_numpy(mask).sum())

    sampler_type = config['SAMPLER_TYPE']

    # Build title
    title_parts = [
        f"Entry {config['ENTRY']}",
        f"Total: {npts}",
        f"Visible: {n_visible}",
        f"Masked: {n_masked}",
        f"Color: {config['COLORBY']}"
    ]

    if sampler_type == 'spatial_box':
        box_min = config['BOX_MIN']
        box_max = config['BOX_MAX']
        title_parts.append(f"Box: [{box_min[0]:.0f},{box_min[1]:.0f},{box_min[2]:.0f}]-[{box_max[0]:.0f},{box_max[1]:.0f},{box_max[2]:.0f}]")

    title_text = " | ".join(title_parts)

    # Build info text
    info_parts = [
        f"Sampler: {sampler_type}",
        f"Max Points: {config['MAX_SPACEPOINTS']}",
        f"Mask Ratio: {config['MASK_RATIO']}",
        f"Masking: {config['MASKING_STRATEGY']}"
    ]

    if sampler_type == 'spatial_box':
        info_parts.append(f"Padding: {config.get('BOX_PADDING', 0)}cm")

    plot_layout = {
        "title": {"text": title_text, "font": {"color": "white", "size": 14}},
        "height": 800,
        "margin": {"t": 50, "b": 0, "l": 0, "r": 0},
        "font": {"size": 12, "color": "white"},
        "showlegend": True,
        "legend": {"x": 0.02, "y": 0.98, "bgcolor": "rgba(50,50,50,0.8)"},
        "plot_bgcolor": "#141414",
        "paper_bgcolor": "#141414",
        "scene": {
            "xaxis": {**axis_template, "title": "x (cm)"},
            "yaxis": {**axis_template, "title": "y (cm)"},
            "zaxis": {**axis_template, "title": "z (cm)"},
            "aspectratio": {"x": 1, "y": 1, "z": 4},
            "camera": {
                "eye": {"x": 2, "y": 2, "z": 2},
                "up": {"x": 0, "y": 1, "z": 0}
            },
            "annotations": [],
        },
    }

    app.layout = html.Div([
        html.Div([
            html.H3(
                f"MAE Dataset Visualizer - Entry {config['ENTRY']}",
                style={"color": "white", "textAlign": "center", "margin": "10px"}
            ),
            html.Div([
                html.Span(f"{part} | ", style={"color": "white"})
                for part in info_parts[:-1]
            ] + [html.Span(info_parts[-1], style={"color": "white"})],
            style={"textAlign": "center", "marginBottom": "10px"}),
            dcc.Graph(
                id="det3d",
                figure={
                    "data": traces,
                    "layout": plot_layout,
                },
                config={"editable": True, "scrollZoom": True},
            )
        ], className="graph__container"),
    ], style={"backgroundColor": "#141414"})

    print(f"\nStarting Dash server on port {config['PORT']}...")
    print(f"Open http://localhost:{config['PORT']} in your browser")

    app.run_server(debug=True, port=config['PORT'])


def main():
    args = parse_args()

    # Build configuration
    config = build_config(args)

    print("=" * 60)
    print("MAE Dataset Visualizer")
    print("=" * 60)
    if args.config:
        print(f"Config file: {args.config}")
    print(f"Entry: {config['ENTRY']}")
    print(f"Color mode: {config['COLORBY']}")
    print(f"Show mask: {config['SHOW_MASK']}")

    # Check for sampler chain config
    sampler_chain_config = config.get('SAMPLER_CHAIN', config.get('sampler_chain', None))
    if sampler_chain_config is not None:
        print(f"Sampler chain: {len(sampler_chain_config)} samplers")
        for i, sc in enumerate(sampler_chain_config):
            sampler_type = sc.get('type', sc.get('TYPE', 'unknown'))
            max_pts = sc.get('max_points', sc.get('MAX_POINTS', 'default'))
            print(f"  [{i+1}] {sampler_type} (max_points={max_pts})")
    else:
        print(f"Sampler: {config['SAMPLER_TYPE']}")
        print(f"Max points: {config['MAX_SPACEPOINTS']}")

    print(f"Mask ratio: {config['MASK_RATIO']}")
    print(f"Masking strategy: {config['MASKING_STRATEGY']}")

    if config['SAMPLER_TYPE'] == 'spatial_box':
        if config.get('RANDOM_BOX_MODE', False):
            print(f"Random box mode: enabled")
            box_size = config.get('BOX_SIZE')
            if box_size:
                print(f"Box size: {box_size}")
            else:
                print(f"Box size: derived from box_min/box_max")
            min_pts = config.get('MIN_POINTS_IN_BOX', 0)
            if min_pts > 0:
                print(f"Min points in box: {min_pts}")
                print(f"Max resample attempts: {config.get('MAX_RESAMPLE_ATTEMPTS', 100)}")
        else:
            print(f"Box min: {config['BOX_MIN']}")
            print(f"Box max: {config['BOX_MAX']}")
        print(f"Box padding: {config.get('BOX_PADDING', 0)} cm")
        print(f"Detector origin: {config['DETECTOR_ORIGIN']}")
        print(f"Detector extent: {config['DETECTOR_EXTENT']}")

    print("=" * 60)

    # Load data through MAEDataset
    print("\nLoading data through MAEDataset...")
    sample, masking, config, sampler = load_mae_data(config)

    # Print sample contents
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Number of points after sampling: {sample['npts']}")

    # Print position range
    if 'positions' in sample:
        positions = tensor_to_numpy(sample['positions'])
        print(f"Position range:")
        print(f"  x: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]")
        print(f"  y: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]")
        print(f"  z: [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")

    # Create mask
    print("\nApplying masking strategy...")
    mask = create_mask(sample, masking)
    n_masked = mask.sum().item()
    n_visible = sample['npts'] - n_masked
    print(f"Visible points: {n_visible}")
    print(f"Masked points: {n_masked}")
    print(f"Actual mask ratio: {n_masked / sample['npts']:.3f}")

    # Build plots
    print("\nBuilding visualization...")
    plots = build_plots(sample, mask, config)

    # Add detector outline
    traces = []
    if not config['NO_DETECTOR']:
        try:
            from lardly.detectoroutline import DetectorOutline
            detdata = DetectorOutline()
            traces = detdata.getlines()
            print("Added detector outline")
        except ImportError:
            print("Warning: lardly not available, skipping detector outline")

    # Add sampling box outline if requested
    # Check if there's a spatial_box sampler (either direct or in chain)
    box_sampler = get_spatial_box_sampler(sampler)
    has_spatial_box = box_sampler is not None or config['SAMPLER_TYPE'] == 'spatial_box'

    if config['SHOW_BOX'] and has_spatial_box:
        box_traces = build_box_traces(config, sampler)
        traces.extend(box_traces)
        # Print actual box coordinates used (especially useful for random box mode)
        if box_sampler is not None and hasattr(box_sampler, 'get_last_box'):
            actual_box_min, actual_box_max = box_sampler.get_last_box()
            print(f"Actual box used: [{actual_box_min[0]:.1f},{actual_box_min[1]:.1f},{actual_box_min[2]:.1f}] to [{actual_box_max[0]:.1f},{actual_box_max[1]:.1f},{actual_box_max[2]:.1f}]")
            # Show resample attempts if applicable
            if hasattr(box_sampler, 'last_resample_attempts') and box_sampler.last_resample_attempts > 0:
                print(f"Resample attempts: {box_sampler.last_resample_attempts}")
        print("Added sampling box outline")

    traces.extend(plots)

    # Run visualization app
    run_app(traces, config, sample, mask)


if __name__ == "__main__":
    main()
