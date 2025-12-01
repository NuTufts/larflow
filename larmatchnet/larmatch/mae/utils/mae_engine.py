"""
MAE Training Engine Utilities

Provides helper functions for model building, training, and evaluation.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from typing import Dict, Any, Optional, Tuple
import math


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build MAE model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SpacepointMAE model instance
    """
    from larmatch.mae.models import SpacepointMAE, SpacepointMAEForPretraining

    backbone_config = {
        'BACKBONE_CHECKPOINT': config.get('BACKBONE_CHECKPOINT', None),
        'FREEZE_BACKBONE': config.get('FREEZE_BACKBONE', False),
        'STEM_NFEATURES': config.get('STEM_NFEATURES', 16),
        'NORM_LAYER': config.get('NORM_LAYER', 'batchnorm'),
        'DEVICE': config.get('DEVICE', 'cuda')
    }

    use_pretraining = config.get('USE_PRETRAINING_MODEL', True)

    if use_pretraining:
        model = SpacepointMAEForPretraining(
            backbone_config=backbone_config,
            d_model=config.get('D_MODEL', 256),
            encoder_layers=config.get('ENCODER_LAYERS', 6),
            encoder_heads=config.get('ENCODER_HEADS', 8),
            decoder_layers=config.get('DECODER_LAYERS', 2),
            decoder_heads=config.get('DECODER_HEADS', 4),
            decoder_dim=config.get('DECODER_DIM', 128),
            output_dim=config.get('OUTPUT_DIM', 3),
            mask_ratio=config.get('MASK_RATIO', 0.75),
            dropout=config.get('DROPOUT', 0.1),
            attention_type=config.get('ATTENTION_TYPE', 'standard'),
            pos_encoding_type=config.get('POS_ENCODING_TYPE', 'sinusoidal'),
            max_spatial_extent=config.get('MAX_SPATIAL_EXTENT', 1000.0),
            use_auxiliary_heads=config.get('USE_AUXILIARY_HEADS', True),
            num_ssnet_classes=config.get('NUM_SSNET_CLASSES', 5),
            num_keypoint_classes=config.get('NUM_KEYPOINT_TYPES', 6),
            use_ema_teacher=config.get('USE_EMA_TEACHER', False),
            ema_decay=config.get('EMA_DECAY', 0.999),
            contrastive_dim=config.get('CONTRASTIVE_DIM', 128)
        )
    else:
        model = SpacepointMAE(
            backbone_config=backbone_config,
            d_model=config.get('D_MODEL', 256),
            encoder_layers=config.get('ENCODER_LAYERS', 6),
            encoder_heads=config.get('ENCODER_HEADS', 8),
            decoder_layers=config.get('DECODER_LAYERS', 2),
            decoder_heads=config.get('DECODER_HEADS', 4),
            decoder_dim=config.get('DECODER_DIM', 128),
            output_dim=config.get('OUTPUT_DIM', 3),
            mask_ratio=config.get('MASK_RATIO', 0.75),
            dropout=config.get('DROPOUT', 0.1),
            attention_type=config.get('ATTENTION_TYPE', 'standard'),
            pos_encoding_type=config.get('POS_ENCODING_TYPE', 'sinusoidal'),
            max_spatial_extent=config.get('MAX_SPATIAL_EXTENT', 1000.0),
            use_auxiliary_heads=config.get('USE_AUXILIARY_HEADS', True),
            num_ssnet_classes=config.get('NUM_SSNET_CLASSES', 5),
            num_keypoint_classes=config.get('NUM_KEYPOINT_TYPES', 6)
        )

    return model


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Build optimizer from configuration.

    Args:
        model: Model to optimize
        config: Configuration dictionary

    Returns:
        Optimizer instance
    """
    optimizer_type = config.get('OPTIMIZER', 'adamw')
    lr = float(config.get('LEARNING_RATE', 1e-4))
    weight_decay = float(config.get('WEIGHT_DECAY', 0.05))

    # Separate backbone and transformer parameters
    backbone_params = []
    transformer_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            transformer_params.append(param)

    # Different learning rates for backbone vs transformer
    backbone_lr_scale = config.get('BACKBONE_LR_SCALE', 0.1)
    param_groups = [
        {'params': backbone_params, 'lr': lr * backbone_lr_scale},
        {'params': transformer_params, 'lr': lr}
    ]

    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer


def build_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Any:
    """
    Build learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary

    Returns:
        Scheduler instance
    """
    scheduler_type = config.get('SCHEDULER', 'cosine')
    warmup_epochs = float(config.get('WARMUP_EPOCHS', 5))
    total_epochs = int(config.get('NUM_EPOCHS', 100))
    min_lr = float(config.get('MIN_LR', 1e-6))

    if scheduler_type == 'cosine':
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == 'cosine_warmup':
        # Custom cosine with linear warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'linear':
        def lr_lambda(epoch):
            return 1 - epoch / total_epochs

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    return scheduler


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    auxiliary_outputs: Dict[str, torch.Tensor] = None,
    labels: Dict[str, torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        mask: Mask indicating masked positions
        auxiliary_outputs: Auxiliary head outputs
        labels: Ground truth labels for auxiliary tasks

    Returns:
        Dictionary of metric values
    """
    metrics = {}

    # Reconstruction metrics
    with torch.no_grad():
        B = predictions.shape[0]
        N_pred = predictions.shape[1]
        N_total = mask.shape[1]
        N_masked = mask.sum(dim=1).max().item()
        D = predictions.shape[2]

        # Check if predictions are already for masked positions only
        if N_pred == N_masked and N_pred != N_total:
            # Predictions are already for masked positions
            masked_pred = predictions
            # Extract masked targets to match
            masked_target = torch.zeros(B, N_masked, D, device=targets.device, dtype=targets.dtype)
            for b in range(B):
                masked_idx = mask[b].nonzero(as_tuple=True)[0]
                n_masked_b = len(masked_idx)
                masked_target[b, :n_masked_b] = targets[b, masked_idx]
        else:
            # Predictions are full, extract masked positions
            masked_pred = predictions[mask]
            masked_target = targets[mask]

        mse = ((masked_pred - masked_target) ** 2).mean().item()
        metrics['reconstruction_mse'] = mse

        # MAE (L1)
        mae = (masked_pred - masked_target).abs().mean().item()
        metrics['reconstruction_mae'] = mae

        # Per-plane MSE
        for p in range(min(3, predictions.shape[-1])):
            if masked_pred.dim() == 3:
                plane_mse = ((masked_pred[:, :, p] - masked_target[:, :, p]) ** 2).mean().item()
            else:
                plane_mse = ((masked_pred[:, p] - masked_target[:, p]) ** 2).mean().item()
            metrics[f'reconstruction_mse_plane{p}'] = plane_mse

    # Auxiliary task metrics
    # Note: auxiliary_outputs are for unmasked tokens only, so we need to extract
    # the corresponding labels from the full label tensors
    if auxiliary_outputs is not None and labels is not None:
        unmasked_mask = ~mask  # True for unmasked positions
        N_unmasked_max = unmasked_mask.sum(dim=1).max().item()

        # Ghost classification accuracy
        if 'ghost_logits' in auxiliary_outputs and labels.get('is_true') is not None:
            ghost_pred = auxiliary_outputs['ghost_logits'].argmax(dim=-1)
            is_true_full = labels['is_true']

            # Extract unmasked is_true labels
            unmasked_is_true = torch.zeros(B, N_unmasked_max, dtype=is_true_full.dtype, device=ghost_pred.device)
            for b in range(B):
                unmasked_idx = unmasked_mask[b].nonzero(as_tuple=True)[0]
                n_unmasked = len(unmasked_idx)
                unmasked_is_true[b, :n_unmasked] = is_true_full[b, unmasked_idx]

            ghost_acc = (ghost_pred == unmasked_is_true.long()).float().mean().item()
            metrics['ghost_accuracy'] = ghost_acc

        # SSNet accuracy
        if 'ssnet_logits' in auxiliary_outputs and labels.get('ssnet_labels') is not None:
            ssnet_pred = auxiliary_outputs['ssnet_logits'].argmax(dim=-1)
            ssnet_labels_full = labels['ssnet_labels']

            # Extract unmasked ssnet labels
            unmasked_ssnet = torch.zeros(B, N_unmasked_max, dtype=ssnet_labels_full.dtype, device=ssnet_pred.device)
            for b in range(B):
                unmasked_idx = unmasked_mask[b].nonzero(as_tuple=True)[0]
                n_unmasked = len(unmasked_idx)
                unmasked_ssnet[b, :n_unmasked] = ssnet_labels_full[b, unmasked_idx]

            valid = unmasked_ssnet >= 0
            if valid.sum() > 0:
                ssnet_acc = (ssnet_pred[valid] == unmasked_ssnet[valid]).float().mean().item()
                metrics['ssnet_accuracy'] = ssnet_acc

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    iteration: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_dir: str,
    is_best: bool = False,
    ema_teacher: Any = None
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        iteration: Current iteration
        loss: Current loss value
        config: Training configuration
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
        ema_teacher: Optional EMA teacher state
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'loss': loss,
        'config': config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if ema_teacher is not None:
        checkpoint['ema_teacher_state_dict'] = ema_teacher.state_dict()

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.tar')
    torch.save(checkpoint, checkpoint_path)

    # Save as latest
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.tar')
    torch.save(checkpoint, latest_path)

    # Save as best if applicable
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.tar')
        torch.save(checkpoint, best_path)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler: Any = None,
    device: str = 'cuda'
) -> Tuple[int, int, float]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device to load to

    Returns:
        Tuple of (epoch, iteration, loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    iteration = checkpoint.get('iteration', 0)
    loss = checkpoint.get('loss', float('inf'))

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Iteration: {iteration}, Loss: {loss:.6f}")

    return epoch, iteration, loss


class MetricTracker:
    """
    Track and smooth training metrics.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}

    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            # Keep only recent values
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key] = self.metrics[key][-self.window_size:]

    def get_smoothed(self) -> Dict[str, float]:
        """Get smoothed (averaged) metrics."""
        return {
            key: sum(values) / len(values)
            for key, values in self.metrics.items()
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}


def setup_wandb(config: Dict[str, Any], project_name: str = 'mae-spacepoint'):
    """
    Setup Weights & Biases logging.

    Args:
        config: Training configuration
        project_name: W&B project name

    Returns:
        W&B run object
    """
    try:
        import wandb
        run = wandb.init(
            project=project_name,
            config=config,
            name=config.get('RUN_NAME', None)
        )
        return run
    except ImportError:
        print("wandb not installed, skipping W&B setup")
        return None


def log_to_wandb(wandb_run, metrics: Dict[str, float], step: int):
    """
    Log metrics to Weights & Biases.

    Args:
        wandb_run: W&B run object
        metrics: Metrics to log
        step: Current step
    """
    if wandb_run is not None:
        import wandb
        wandb.log(metrics, step=step)
