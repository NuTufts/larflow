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

Also supports showing masked points vs visible points to verify masking behavior.

Usage:
    python vis_mae_dataset.py -c cache_file.txt -e 0 --colorby ssnet
    python vis_mae_dataset.py -c cache_file.txt -e 0 --colorby edep --show-mask
    python vis_mae_dataset.py -i /path/to/file.h5 -e 0 --colorby kpnu --show-mask
"""

import os
import sys
import argparse
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
    0: 'rgba(50,50,50,1.0)',    # background
    1: 'rgba(255,0,0,1.0)',     # electron
    2: 'rgba(200,125,0,1)',     # photon
    3: 'rgba(0,0,255,1)',       # muon
    4: 'rgba(0,125,255,1)',     # proton
    5: 'rgba(125,0,255,1)',     # pion/kaon
    6: 'rgba(125,125,0,1)',     # other
}

SSNET_CLASS_NAMES = {
    0: 'Background',
    1: 'Electron',
    2: 'Photon',
    3: 'Muon',
    4: 'Proton',
    5: 'Pion/Kaon',
    6: 'Other',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MAE Dataset training data using Plotly/Dash"
    )

    # Data source - either cache file or direct HDF5 files
    data_group = parser.add_mutually_exclusive_group(required=True)
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

    parser.add_argument(
        "-e", "--entry",
        type=int,
        default=0,
        help="Entry/event index to visualize (default: 0)"
    )
    parser.add_argument(
        "--colorby",
        type=str,
        default='ssnet',
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
        "--mask-ratio",
        type=float,
        default=0.75,
        help="Mask ratio for MAE masking (default: 0.75)"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum spacepoints after sampling (default: 50000)"
    )
    parser.add_argument(
        "--sampler-type",
        type=str,
        default='importance',
        choices=['random', 'importance', 'stratified', 'particle', 'spatial'],
        help="Sampler type (default: importance)"
    )
    parser.add_argument(
        "--masking-strategy",
        type=str,
        default='random',
        choices=['random', 'spatial', 'particle', 'ghost_aware'],
        help="Masking strategy (default: random)"
    )
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


def load_mae_data(args):
    """
    Load data through MAEDataset class.

    Returns:
        Dictionary with PyTorch tensors as the training loop would see them.
    """
    # Add path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mae_dir = os.path.dirname(script_dir)
    larmatch_dir = os.path.dirname(mae_dir)
    if larmatch_dir not in sys.path:
        sys.path.insert(0, larmatch_dir)

    from mae.data.mae_dataset import MAEDataset
    from mae.data.masking import create_masking_strategy

    # Build config
    config = {
        'SAMPLER_TYPE': args.sampler_type,
        'MAX_SPACEPOINTS': args.max_points,
        'MASK_RATIO': args.mask_ratio,
        'MASKING_STRATEGY': args.masking_strategy,
    }

    # Create dataset
    if args.cache_file:
        dataset = MAEDataset(
            load_from_cachefile=args.cache_file,
            config=config
        )
    else:
        dataset = MAEDataset(
            file_paths=args.input_files,
            config=config
        )

    print(f"Dataset loaded with {len(dataset)} entries")

    if args.entry >= len(dataset):
        print(f"Error: Entry {args.entry} out of range (max: {len(dataset)-1})")
        sys.exit(1)

    # Get sample (this applies sampling)
    sample = dataset[args.entry]

    # Create masking
    masking = create_masking_strategy(config)

    return sample, masking, config


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


def build_plots(sample, mask, args):
    """
    Build Plotly traces for visualization.

    Args:
        sample: Dataset sample dictionary with tensors
        mask: Boolean mask tensor (True = masked)
        args: Command line arguments

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

    colorby = args.colorby

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
            if args.show_mask:
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
        if args.show_mask:
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
        if args.show_mask:
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
        if args.show_mask:
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

        # True points (visible)
        true_visible = is_true & visible_idx
        ghost_visible = (~is_true) & visible_idx

        plots.append({
            "type": "scatter3d",
            "x": positions[true_visible, 0],
            "y": positions[true_visible, 1],
            "z": positions[true_visible, 2],
            "mode": "markers",
            "name": "True (visible)",
            "hovertemplate": hovertemplate,
            "customdata": customdata[true_visible],
            "marker": {
                "color": 'rgba(0,255,0,1)',
                "opacity": opacity,
                "size": marker_size
            }
        })

        plots.append({
            "type": "scatter3d",
            "x": positions[ghost_visible, 0],
            "y": positions[ghost_visible, 1],
            "z": positions[ghost_visible, 2],
            "mode": "markers",
            "name": "Ghost (visible)",
            "hovertemplate": hovertemplate,
            "customdata": customdata[ghost_visible],
            "marker": {
                "color": 'rgba(255,0,0,1)',
                "opacity": opacity,
                "size": marker_size
            }
        })

        # Masked points
        if args.show_mask:
            true_masked = is_true & masked_idx
            ghost_masked = (~is_true) & masked_idx

            plots.append({
                "type": "scatter3d",
                "x": positions[true_masked, 0],
                "y": positions[true_masked, 1],
                "z": positions[true_masked, 2],
                "mode": "markers",
                "name": "True (masked)",
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
                "name": "Ghost (masked)",
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


def run_app(traces, args, sample, mask):
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

    title_text = (
        f"Entry {args.entry} | "
        f"Total: {npts} | Visible: {n_visible} | Masked: {n_masked} | "
        f"Color: {args.colorby}"
    )

    plot_layout = {
        "title": {"text": title_text, "font": {"color": "white"}},
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
                f"MAE Dataset Visualizer - Entry {args.entry}",
                style={"color": "white", "textAlign": "center", "margin": "10px"}
            ),
            html.Div([
                html.Span(f"Sampler: {args.sampler_type} | ", style={"color": "white"}),
                html.Span(f"Max Points: {args.max_points} | ", style={"color": "white"}),
                html.Span(f"Mask Ratio: {args.mask_ratio} | ", style={"color": "white"}),
                html.Span(f"Masking: {args.masking_strategy}", style={"color": "white"}),
            ], style={"textAlign": "center", "marginBottom": "10px"}),
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

    print(f"\nStarting Dash server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run_server(debug=True, port=args.port)


def main():
    args = parse_args()

    print("=" * 60)
    print("MAE Dataset Visualizer")
    print("=" * 60)
    print(f"Color mode: {args.colorby}")
    print(f"Show mask: {args.show_mask}")
    print(f"Sampler: {args.sampler_type}")
    print(f"Max points: {args.max_points}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Masking strategy: {args.masking_strategy}")
    print("=" * 60)

    # Load data through MAEDataset
    print("\nLoading data through MAEDataset...")
    sample, masking, config = load_mae_data(args)

    # Print sample contents
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Number of points after sampling: {sample['npts']}")

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
    plots = build_plots(sample, mask, args)

    # Add detector outline
    traces = []
    if not args.no_detector:
        try:
            from lardly.detectoroutline import DetectorOutline
            detdata = DetectorOutline()
            traces = detdata.getlines()
            print("Added detector outline")
        except ImportError:
            print("Warning: lardly not available, skipping detector outline")

    traces.extend(plots)

    # Run visualization app
    run_app(traces, args, sample, mask)


if __name__ == "__main__":
    main()
