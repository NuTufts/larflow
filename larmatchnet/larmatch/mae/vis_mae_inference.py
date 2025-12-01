#!/bin/env python3
"""
Visualize MAE Inference Output

This script visualizes the auxiliary predictions from run_mae_inference.py
in 3D scatter plots.

Supports coloring by:
- ghost_score: Predicted probability of being a true (non-ghost) point
- ghost_pred: Binary ghost prediction (0=ghost, 1=true)
- ssnet_pred: Predicted SSNet class
- ssnet_prob_X: Probability for specific SSNet class (0-4)
- keypoint_X: Keypoint score for specific type (nu, trackstart, trackend, shower, michel, delta)

If ground truth was saved (--save-truth in inference), can also show:
- Comparison between predictions and truth
- Accuracy metrics

Usage:
    python vis_mae_inference.py -i inference_output.h5 -e 0 --colorby ghost_score
    python vis_mae_inference.py -i inference_output.h5 -e 0 --colorby ssnet_pred
    python vis_mae_inference.py -i inference_output.h5 -e 0 --colorby keypoint_nu
    python vis_mae_inference.py -i inference_output.h5 -e 0 --colorby ghost_score --show-truth
"""

import os
import sys
import argparse
import numpy as np
import h5py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Colorby options
COLORBY_OPTIONS = [
    'ghost_score',
    'ghost_pred',
    'ghost_correct',  # Shows where prediction matches truth
    'ssnet_pred',
    'ssnet_prob_0', 'ssnet_prob_1', 'ssnet_prob_2', 'ssnet_prob_3', 'ssnet_prob_4',
    'keypoint_nu', 'keypoint_trackstart', 'keypoint_trackend',
    'keypoint_shower', 'keypoint_michel', 'keypoint_delta',
]

# Keypoint index mapping
KPINDEX = {
    'keypoint_nu': 0,
    'keypoint_trackstart': 1,
    'keypoint_trackend': 2,
    'keypoint_shower': 3,
    'keypoint_michel': 4,
    'keypoint_delta': 5
}

KPNAMES = {
    'keypoint_nu': 'Neutrino Vertex',
    'keypoint_trackstart': 'Track Start',
    'keypoint_trackend': 'Track End',
    'keypoint_shower': 'Shower Start',
    'keypoint_michel': 'Michel Start',
    'keypoint_delta': 'Delta Start'
}

# SSNet class colors and names
SSNET_CLASS_COLORS = {
    0: 'rgba(50,50,50,1.0)',    # background
    1: 'rgba(255,0,0,1.0)',     # electron
    2: 'rgba(200,125,0,1)',     # photon
    3: 'rgba(0,0,255,1)',       # muon
    4: 'rgba(0,125,255,1)',     # proton
}

SSNET_CLASS_NAMES = {
    0: 'Background',
    1: 'Electron',
    2: 'Photon',
    3: 'Muon',
    4: 'Proton',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MAE inference output using Plotly/Dash"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to inference output HDF5 file"
    )
    parser.add_argument(
        "-e", "--entry",
        type=int,
        default=0,
        help="Entry index to visualize (default: 0)"
    )
    parser.add_argument(
        "--colorby",
        type=str,
        default='ghost_score',
        choices=COLORBY_OPTIONS,
        help=f"Color mode (default: ghost_score)"
    )
    parser.add_argument(
        "--show-truth",
        action='store_true',
        default=False,
        help="Show ground truth comparison (if available)"
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary predictions (default: 0.5)"
    )

    return parser.parse_args()


def load_inference_data(input_path, entry_idx):
    """
    Load inference output from HDF5 file.

    Args:
        input_path: Path to HDF5 file
        entry_idx: Entry index to load

    Returns:
        Dictionary with inference results
    """
    f = h5py.File(input_path, 'r')

    entry_key = f'entry_{entry_idx}'
    if entry_key not in f:
        available = [k for k in f.keys() if k.startswith('entry_')]
        print(f"Error: Entry {entry_idx} not found. Available entries: {len(available)}")
        print(f"  First few: {available[:5]}")
        sys.exit(1)

    grp = f[entry_key]

    data = {
        'npts': grp.attrs['npts'],
        'positions': grp['positions'][:],
    }

    # Load predictions
    if 'ghost_score' in grp:
        data['ghost_score'] = grp['ghost_score'][:]
    if 'ghost_pred' in grp:
        data['ghost_pred'] = grp['ghost_pred'][:]
    if 'ssnet_probs' in grp:
        data['ssnet_probs'] = grp['ssnet_probs'][:]
    if 'ssnet_pred' in grp:
        data['ssnet_pred'] = grp['ssnet_pred'][:]
    if 'keypoint_scores' in grp:
        data['keypoint_scores'] = grp['keypoint_scores'][:]

    # Load truth if available
    if 'truth_is_true' in grp:
        data['truth_is_true'] = grp['truth_is_true'][:]
    if 'truth_ssnet_labels' in grp:
        data['truth_ssnet_labels'] = grp['truth_ssnet_labels'][:]
    if 'truth_keypoint_scores' in grp:
        data['truth_keypoint_scores'] = grp['truth_keypoint_scores'][:]

    # Get file-level attributes
    data['num_ssnet_classes'] = f.attrs.get('num_ssnet_classes', 5)
    data['num_keypoint_types'] = f.attrs.get('num_keypoint_types', 6)

    f.close()

    return data


def compute_metrics(data):
    """
    Compute accuracy metrics if truth data is available.

    Args:
        data: Data dictionary

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Ghost accuracy
    if 'ghost_pred' in data and 'truth_is_true' in data:
        truth = data['truth_is_true'].astype(int)
        pred = data['ghost_pred']
        metrics['ghost_accuracy'] = (pred == truth).mean()
        metrics['ghost_true_as_true'] = pred[truth == 1].mean() if (truth == 1).sum() > 0 else 0
        metrics['ghost_ghost_as_ghost'] = (1 - pred[truth == 0]).mean() if (truth == 0).sum() > 0 else 0

    # SSNet accuracy
    if 'ssnet_pred' in data and 'truth_ssnet_labels' in data:
        truth = data['truth_ssnet_labels']
        pred = data['ssnet_pred']
        valid = truth >= 0  # -1 means no label
        if valid.sum() > 0:
            metrics['ssnet_accuracy'] = (pred[valid] == truth[valid]).mean()

    return metrics


def build_plots(data, args):
    """
    Build Plotly traces for visualization.

    Args:
        data: Data dictionary from load_inference_data
        args: Command line arguments

    Returns:
        List of Plotly trace dictionaries
    """
    plots = []

    positions = data['positions']
    npts = data['npts']

    opacity = 0.8
    marker_size = 2.0

    # Build hover template
    hovertemplate = """
<b>x</b>: %{x:.1f}<br>
<b>y</b>: %{y:.1f}<br>
<b>z</b>: %{z:.1f}<br>
<b>idx</b>: %{customdata[0]:d}<br>
"""

    customdata = np.arange(npts).reshape(-1, 1)
    colorby = args.colorby

    # Ghost score (continuous 0-1)
    if colorby == 'ghost_score':
        if 'ghost_score' not in data:
            print("Warning: ghost_score not in data")
            return plots

        ghost_score = data['ghost_score']

        plots.append({
            "type": "scatter3d",
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "mode": "markers",
            "name": "Ghost Score",
            "hovertemplate": hovertemplate + "<b>ghost_score</b>: %{marker.color:.3f}<br>",
            "customdata": customdata,
            "marker": {
                "color": ghost_score,
                "opacity": opacity,
                "size": marker_size,
                "colorscale": "RdYlGn",  # Red=ghost, Green=true
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {"title": "Ghost Score<br>(1=true)"}
            }
        })

        # Show truth comparison if requested
        if args.show_truth and 'truth_is_true' in data:
            truth = data['truth_is_true']
            # Mark incorrect predictions
            incorrect = (ghost_score > args.threshold) != truth
            if incorrect.sum() > 0:
                plots.append({
                    "type": "scatter3d",
                    "x": positions[incorrect, 0],
                    "y": positions[incorrect, 1],
                    "z": positions[incorrect, 2],
                    "mode": "markers",
                    "name": f"Incorrect ({incorrect.sum()})",
                    "hovertemplate": hovertemplate,
                    "customdata": customdata[incorrect],
                    "marker": {
                        "color": 'rgba(255,255,0,1)',
                        "opacity": 1.0,
                        "size": marker_size * 2,
                        "symbol": "x"
                    }
                })

    # Ghost prediction (binary)
    elif colorby == 'ghost_pred':
        if 'ghost_pred' not in data:
            print("Warning: ghost_pred not in data")
            return plots

        ghost_pred = data['ghost_pred']

        # True predictions (pred=1)
        true_mask = ghost_pred == 1
        plots.append({
            "type": "scatter3d",
            "x": positions[true_mask, 0],
            "y": positions[true_mask, 1],
            "z": positions[true_mask, 2],
            "mode": "markers",
            "name": f"Pred True ({true_mask.sum()})",
            "hovertemplate": hovertemplate,
            "customdata": customdata[true_mask],
            "marker": {
                "color": 'rgba(0,255,0,1)',
                "opacity": opacity,
                "size": marker_size
            }
        })

        # Ghost predictions (pred=0)
        ghost_mask = ghost_pred == 0
        plots.append({
            "type": "scatter3d",
            "x": positions[ghost_mask, 0],
            "y": positions[ghost_mask, 1],
            "z": positions[ghost_mask, 2],
            "mode": "markers",
            "name": f"Pred Ghost ({ghost_mask.sum()})",
            "hovertemplate": hovertemplate,
            "customdata": customdata[ghost_mask],
            "marker": {
                "color": 'rgba(255,0,0,1)',
                "opacity": opacity,
                "size": marker_size
            }
        })

    # Ghost correctness (where prediction matches truth)
    elif colorby == 'ghost_correct':
        if 'ghost_pred' not in data or 'truth_is_true' not in data:
            print("Warning: Need both ghost_pred and truth_is_true for ghost_correct")
            return plots

        pred = data['ghost_pred']
        truth = data['truth_is_true'].astype(int)
        correct = pred == truth

        # Correct predictions
        plots.append({
            "type": "scatter3d",
            "x": positions[correct, 0],
            "y": positions[correct, 1],
            "z": positions[correct, 2],
            "mode": "markers",
            "name": f"Correct ({correct.sum()})",
            "hovertemplate": hovertemplate,
            "customdata": customdata[correct],
            "marker": {
                "color": 'rgba(0,255,0,1)',
                "opacity": opacity,
                "size": marker_size
            }
        })

        # Incorrect predictions
        incorrect = ~correct
        plots.append({
            "type": "scatter3d",
            "x": positions[incorrect, 0],
            "y": positions[incorrect, 1],
            "z": positions[incorrect, 2],
            "mode": "markers",
            "name": f"Incorrect ({incorrect.sum()})",
            "hovertemplate": hovertemplate,
            "customdata": customdata[incorrect],
            "marker": {
                "color": 'rgba(255,0,0,1)',
                "opacity": opacity,
                "size": marker_size * 1.5
            }
        })

    # SSNet prediction (categorical)
    elif colorby == 'ssnet_pred':
        if 'ssnet_pred' not in data:
            print("Warning: ssnet_pred not in data")
            return plots

        ssnet_pred = data['ssnet_pred']
        num_classes = data.get('num_ssnet_classes', 5)

        for iclass in range(num_classes):
            class_mask = ssnet_pred == iclass
            if class_mask.sum() == 0:
                continue

            color = SSNET_CLASS_COLORS.get(iclass, f'rgba({iclass*50},100,{255-iclass*50},1)')
            name = SSNET_CLASS_NAMES.get(iclass, f'Class {iclass}')

            plots.append({
                "type": "scatter3d",
                "x": positions[class_mask, 0],
                "y": positions[class_mask, 1],
                "z": positions[class_mask, 2],
                "mode": "markers",
                "name": f"{name} ({class_mask.sum()})",
                "hovertemplate": hovertemplate,
                "customdata": customdata[class_mask],
                "marker": {
                    "color": color,
                    "opacity": opacity,
                    "size": marker_size
                }
            })

    # SSNet probability for specific class
    elif colorby.startswith('ssnet_prob_'):
        if 'ssnet_probs' not in data:
            print("Warning: ssnet_probs not in data")
            return plots

        class_idx = int(colorby.split('_')[-1])
        ssnet_probs = data['ssnet_probs']

        if class_idx >= ssnet_probs.shape[1]:
            print(f"Warning: Class {class_idx} out of range")
            return plots

        class_prob = ssnet_probs[:, class_idx]
        class_name = SSNET_CLASS_NAMES.get(class_idx, f'Class {class_idx}')

        plots.append({
            "type": "scatter3d",
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "mode": "markers",
            "name": f"{class_name} Probability",
            "hovertemplate": hovertemplate + f"<b>{class_name} prob</b>: " + "%{marker.color:.3f}<br>",
            "customdata": customdata,
            "marker": {
                "color": class_prob,
                "opacity": opacity,
                "size": marker_size,
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {"title": f"{class_name}<br>Probability"}
            }
        })

    # Keypoint scores
    elif colorby.startswith('keypoint_'):
        if 'keypoint_scores' not in data:
            print("Warning: keypoint_scores not in data")
            return plots

        kp_key = colorby
        if kp_key not in KPINDEX:
            print(f"Warning: Unknown keypoint type {kp_key}")
            return plots

        kp_idx = KPINDEX[kp_key]
        kp_scores = data['keypoint_scores']

        if kp_idx >= kp_scores.shape[1]:
            print(f"Warning: Keypoint index {kp_idx} out of range")
            return plots

        kp_score = kp_scores[:, kp_idx]
        kp_name = KPNAMES[kp_key]

        # Normalize for visualization (scores can be negative from regression)
        vmin = np.percentile(kp_score, 5)
        vmax = np.percentile(kp_score, 95)

        plots.append({
            "type": "scatter3d",
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "mode": "markers",
            "name": f"{kp_name} Score",
            "hovertemplate": hovertemplate + f"<b>{kp_name}</b>: " + "%{marker.color:.3f}<br>",
            "customdata": customdata,
            "marker": {
                "color": kp_score,
                "opacity": opacity,
                "size": marker_size,
                "colorscale": "Viridis",
                "cmin": vmin,
                "cmax": vmax,
                "colorbar": {"title": f"{kp_name}<br>Score"}
            }
        })

        # Show truth comparison if requested
        if args.show_truth and 'truth_keypoint_scores' in data:
            truth_scores = data['truth_keypoint_scores'][:, kp_idx]
            # Highlight high-score truth points
            high_truth = truth_scores > 0.5
            if high_truth.sum() > 0:
                plots.append({
                    "type": "scatter3d",
                    "x": positions[high_truth, 0],
                    "y": positions[high_truth, 1],
                    "z": positions[high_truth, 2],
                    "mode": "markers",
                    "name": f"Truth {kp_name} ({high_truth.sum()})",
                    "hovertemplate": hovertemplate,
                    "customdata": customdata[high_truth],
                    "marker": {
                        "color": 'rgba(255,255,0,1)',
                        "opacity": 1.0,
                        "size": marker_size * 3,
                        "symbol": "diamond"
                    }
                })

    return plots


def run_app(traces, args, data, metrics):
    """Launch Dash app for visualization."""

    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    axis_template = {
        "showbackground": True,
        "backgroundcolor": "#141414",
        "gridcolor": "rgb(255, 255, 255)",
        "zerolinecolor": "rgb(255, 255, 255)",
    }

    # Build title with metrics
    npts = data['npts']
    title_parts = [
        f"Entry {args.entry}",
        f"Points: {npts}",
        f"Color: {args.colorby}"
    ]

    if 'ghost_accuracy' in metrics:
        title_parts.append(f"Ghost Acc: {metrics['ghost_accuracy']:.1%}")
    if 'ssnet_accuracy' in metrics:
        title_parts.append(f"SSNet Acc: {metrics['ssnet_accuracy']:.1%}")

    title_text = " | ".join(title_parts)

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
        },
    }

    # Build metrics display
    metrics_divs = []
    if metrics:
        metrics_divs.append(html.H4("Metrics:", style={"color": "white", "marginTop": "10px"}))
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_divs.append(
                    html.Div(f"{key}: {value:.4f}", style={"color": "white", "marginLeft": "20px"})
                )

    app.layout = html.Div([
        html.Div([
            html.H3(
                f"MAE Inference Visualizer",
                style={"color": "white", "textAlign": "center", "margin": "10px"}
            ),
            html.Div([
                html.Span(f"Input: {os.path.basename(args.input)} | ", style={"color": "white"}),
                html.Span(f"Entry: {args.entry} | ", style={"color": "white"}),
                html.Span(f"Color: {args.colorby}", style={"color": "white"}),
            ], style={"textAlign": "center", "marginBottom": "10px"}),
            dcc.Graph(
                id="det3d",
                figure={
                    "data": traces,
                    "layout": plot_layout,
                },
                config={"editable": True, "scrollZoom": True},
            ),
            html.Div(metrics_divs, style={"padding": "10px"}) if metrics_divs else None,
        ], className="graph__container"),
    ], style={"backgroundColor": "#141414"})

    print(f"\nStarting Dash server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run_server(debug=True, port=args.port)


def main():
    args = parse_args()

    print("=" * 60)
    print("MAE Inference Visualizer")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Entry: {args.entry}")
    print(f"Color mode: {args.colorby}")
    print(f"Show truth: {args.show_truth}")
    print("=" * 60)

    # Load data
    print("\nLoading inference data...")
    data = load_inference_data(args.input, args.entry)

    print(f"Loaded {data['npts']} points")
    print(f"Available fields: {[k for k in data.keys() if not k.startswith('num_')]}")

    # Check for truth data
    has_truth = 'truth_is_true' in data
    print(f"Has truth data: {has_truth}")

    # Compute metrics
    metrics = {}
    if has_truth:
        print("\nComputing metrics...")
        metrics = compute_metrics(data)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    # Build plots
    print("\nBuilding visualization...")
    plots = build_plots(data, args)

    if not plots:
        print("Error: No plots generated")
        sys.exit(1)

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
    run_app(traces, args, data, metrics)


if __name__ == "__main__":
    main()
