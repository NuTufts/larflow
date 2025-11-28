#!/usr/bin/env python
"""
MAE Training Script for Spacepoint Encodings

Trains a Masked Auto-Encoder model for learning robust spacepoint representations.
Supports distributed training, mixed precision, and wandb logging.
"""

import os
import sys
import argparse
import time
import datetime
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

# Import MAE modules
from larmatch.mae.models import SpacepointMAE, SpacepointMAEForPretraining
from larmatch.mae.loss import MAELoss
from larmatch.mae.data import create_mae_dataloader, prepare_mae_batch
from larmatch.mae.utils.mae_engine import (
    load_config, build_model, build_optimizer, build_scheduler,
    compute_metrics, save_checkpoint, load_checkpoint,
    MetricTracker, setup_wandb, log_to_wandb
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MAE Training for Spacepoints')

    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')

    return parser.parse_args()


def setup_distributed(local_rank: int, world_size: int):
    """Initialize distributed training."""
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=local_rank,
            timeout=datetime.timedelta(minutes=30)
        )
        torch.cuda.set_device(local_rank)


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    scaler,
    config: Dict[str, Any],
    epoch: int,
    start_iter: int,
    device: str,
    wandb_run=None,
    metric_tracker=None,
    ema_teacher=None
) -> int:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for AMP
        config: Training configuration
        epoch: Current epoch
        start_iter: Starting iteration
        device: Device to train on
        wandb_run: W&B run object
        metric_tracker: Metric tracking object
        ema_teacher: Optional EMA teacher

    Returns:
        Current iteration count
    """
    model.train()
    iteration = start_iter

    use_amp = config.get('USE_AMP', True)
    grad_accum_steps = config.get('GRADIENT_ACCUMULATION_STEPS', 1)
    clip_grad_norm = config.get('CLIP_GRAD_NORM', 1.0)
    log_interval = config.get('LOG_INTERVAL', 100)
    checkpoint_interval = config.get('CHECKPOINT_INTERVAL', 5000)
    checkpoint_dir = config.get('CHECKPOINT_DIR', 'checkpoints')

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        iter_start = time.time()

        # Prepare batch
        batch = prepare_mae_batch(batch, device)

        # Forward pass with automatic mixed precision
        with autocast(enabled=use_amp):
            # Run model
            outputs = model(
                batch['wireplane_sparsetensors'],
                batch['query_v'],
                batch['positions'],
                return_encoder_features=True
            )

            # Compute loss
            loss_dict = criterion(
                predictions=outputs['reconstruction'],
                targets=batch['pixel_values'],
                mask=outputs['mask'],
                encoder_features=outputs.get('encoder_output'),
                instance_labels=batch.get('instance_labels'),
                is_true=batch.get('is_true'),
                auxiliary_outputs=outputs.get('auxiliary'),
                ssnet_labels=batch.get('ssnet_labels'),
                keypoint_targets=batch.get('keypoint_scores'),
                teacher_features=None  # Add teacher features if using EMA
            )

            loss = loss_dict['total'] / grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA teacher
            if ema_teacher is not None:
                ema_teacher.update(model)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        iteration += 1

        # Logging
        iter_time = time.time() - iter_start

        if iteration % log_interval == 0:
            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(
                    outputs['reconstruction'],
                    batch['pixel_values'],
                    outputs['mask'],
                    outputs.get('auxiliary'),
                    {
                        'is_true': batch.get('is_true'),
                        'ssnet_labels': batch.get('ssnet_labels')
                    }
                )

            # Add loss values
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    metrics[f'loss_{key}'] = value.item()

            # Add learning rate
            metrics['lr'] = optimizer.param_groups[0]['lr']
            metrics['iter_time'] = iter_time

            # Update tracker
            if metric_tracker is not None:
                metric_tracker.update(metrics)
                smoothed = metric_tracker.get_smoothed()
            else:
                smoothed = metrics

            # Print
            print(f"Epoch {epoch} | Iter {iteration} | "
                  f"Loss: {smoothed['loss_total']:.4f} | "
                  f"Recon: {smoothed.get('loss_reconstruction', 0):.4f} | "
                  f"LR: {metrics['lr']:.2e} | "
                  f"Time: {iter_time:.3f}s")

            # Log to wandb
            if wandb_run is not None:
                log_to_wandb(wandb_run, smoothed, iteration)

        # Save checkpoint
        if iteration % checkpoint_interval == 0:
            save_checkpoint(
                model=model.module if hasattr(model, 'module') else model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                iteration=iteration,
                loss=loss_dict['total'].item(),
                config=config,
                checkpoint_dir=checkpoint_dir,
                ema_teacher=ema_teacher
            )

    return iteration


def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    config: Dict[str, Any],
    device: str
) -> Dict[str, float]:
    """
    Run validation.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        config: Configuration
        device: Device

    Returns:
        Dictionary of validation metrics
    """
    model.eval()

    num_batches = config.get('NUM_VALID_BATCHES', 10)
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            batch = prepare_mae_batch(batch, device)

            outputs = model(
                batch['wireplane_sparsetensors'],
                batch['query_v'],
                batch['positions'],
                return_encoder_features=True
            )

            loss_dict = criterion(
                predictions=outputs['reconstruction'],
                targets=batch['pixel_values'],
                mask=outputs['mask'],
                encoder_features=outputs.get('encoder_output'),
                instance_labels=batch.get('instance_labels'),
                is_true=batch.get('is_true'),
                auxiliary_outputs=outputs.get('auxiliary'),
                ssnet_labels=batch.get('ssnet_labels'),
                keypoint_targets=batch.get('keypoint_scores'),
            )

            metrics = compute_metrics(
                outputs['reconstruction'],
                batch['pixel_values'],
                outputs['mask'],
                outputs.get('auxiliary'),
                {
                    'is_true': batch.get('is_true'),
                    'ssnet_labels': batch.get('ssnet_labels')
                }
            )

            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    metrics[f'loss_{key}'] = value.item()

            all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[f'val_{key}'] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return avg_metrics


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.resume:
        config['CHECKPOINT_FILE'] = args.resume
        config['RESUME_FROM_CHECKPOINT'] = True
    config['CHECKPOINT_DIR'] = args.checkpoint_dir

    # Setup device
    device = torch.device(config.get('DEVICE', 'cuda'))
    local_rank = args.local_rank
    world_size = args.gpus

    # Setup distributed training
    if world_size > 1:
        setup_distributed(local_rank, world_size)

    # Set random seed
    seed = config.get('SEED', 42) + local_rank
    torch.manual_seed(seed)

    # Build model
    print("Building model...")
    model = build_model(config)
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Print model info
    if local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Build loss function
    criterion = MAELoss(config=config)

    # Setup gradient scaler for AMP
    scaler = GradScaler(enabled=config.get('USE_AMP', True))

    # Setup EMA teacher if enabled
    ema_teacher = None
    if config.get('USE_EMA_TEACHER', False):
        from larmatch.mae.loss.distillation_loss import EMATeacher
        ema_teacher = EMATeacher(model, config.get('EMA_DECAY', 0.999))

    # Resume from checkpoint
    start_epoch = 0
    start_iter = 0
    if config.get('RESUME_FROM_CHECKPOINT', False):
        checkpoint_path = config.get('CHECKPOINT_FILE')
        if os.path.exists(checkpoint_path):
            start_epoch, start_iter, _ = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )

    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_mae_dataloader(
        load_from_cachefile=config.get('TRAIN_DATASET_INPUT_TXTFILE'),
        batch_size=config.get('BATCH_SIZE', 4),
        num_workers=config.get('NUM_TRAIN_WORKERS', 4),
        shuffle=True,
        config=config
    )

    valid_loader = None
    if config.get('VALID_DATASET_INPUT_TXTFILE'):
        valid_loader = create_mae_dataloader(
            load_from_cachefile=config.get('VALID_DATASET_INPUT_TXTFILE'),
            batch_size=config.get('BATCH_SIZE', 4),
            num_workers=config.get('NUM_VALID_WORKERS', 2),
            shuffle=False,
            config=config
        )

    # Setup logging
    wandb_run = None
    if local_rank == 0 and not args.no_wandb and config.get('LOGGER') == 'wandb':
        wandb_run = setup_wandb(config, project_name='mae-spacepoint')

    # Setup metric tracker
    metric_tracker = MetricTracker(window_size=100)

    # Training loop
    print("Starting training...")
    num_epochs = config.get('NUM_EPOCHS', 100)
    current_iter = start_iter

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        current_iter = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            start_iter=current_iter,
            device=device,
            wandb_run=wandb_run,
            metric_tracker=metric_tracker,
            ema_teacher=ema_teacher
        )

        # Validate
        if valid_loader is not None and local_rank == 0:
            print("\nValidating...")
            val_metrics = validate(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                config=config,
                device=device
            )

            print(f"Validation - Loss: {val_metrics['val_loss_total']:.4f} | "
                  f"Recon MSE: {val_metrics.get('val_reconstruction_mse', 0):.4f}")

            if wandb_run is not None:
                log_to_wandb(wandb_run, val_metrics, current_iter)

            # Save best model
            is_best = val_metrics['val_loss_total'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss_total']
                save_checkpoint(
                    model=model.module if hasattr(model, 'module') else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    iteration=current_iter,
                    loss=val_metrics['val_loss_total'],
                    config=config,
                    checkpoint_dir=args.checkpoint_dir,
                    is_best=True,
                    ema_teacher=ema_teacher
                )

    # Cleanup
    if wandb_run is not None:
        wandb_run.finish()

    if world_size > 1:
        cleanup_distributed()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
