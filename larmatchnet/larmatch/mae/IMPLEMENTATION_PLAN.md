# MAE Training Implementation Plan

This document outlines the implementation plan for Masked Auto-Encoder (MAE) training for spacepoint encodings.

## Overview

The goal is to train a model that builds robust encodings for each energy deposition spacepoint using a semi-supervised masked auto-encoding approach. The encoder representations can then be used for downstream tasks.

## Core Model Components

### 1.1 Image Feature Backbone (Refactored `LArMatchMinkowski`)
- **File**: `mae/models/mae_backbone.py`
- Refactor `LArMatchMinkowski.forward()` into separate methods:
  - `encode_images(input_wireplane_sparsetensors)` → Run stem + encoder + decoder on each plane
  - `extract_spacepoint_features(x_feat_v, query_v)` → Extract per-spacepoint feature vectors (existing logic)
- Keep the UNet encoder/decoder frozen or with slower learning rate during MAE pretraining

### 1.2 Position Encoding Module
- **File**: `mae/models/position_encoding.py`
- Sinusoidal 3D position encoding for (x, y, z) coordinates
- Options:
  - Standard sinusoidal (Vaswani et al.)
  - Learnable Fourier features
  - RoPE (Rotary Position Embedding)

### 1.3 Transformer Encoder
- **File**: `mae/models/transformer_encoder.py`
- Standard transformer encoder blocks
- Configurable depth, heads, dimension
- Pre-norm architecture (more stable for training)

### 1.4 MAE Decoder
- **File**: `mae/models/mae_decoder.py`
- Shallow transformer decoder (2-4 layers as per MAE paper)
- Reconstructs masked pixel values from unmasked context + learnable mask tokens

### 1.5 Main MAE Model
- **File**: `mae/models/spacepoint_mae.py`
- Combines all components
- Handles masking strategy, token assembly, loss computation

## Loss Functions

### 2.1 Reconstruction Loss
- **File**: `mae/loss/reconstruction_loss.py`
- MSE or L1 loss on masked pixel values
- Option: Per-plane reconstruction or combined

### 2.2 Contrastive Loss
- **File**: `mae/loss/contrastive_loss.py`
- Same-particle vs different-particle cosine similarity
- Options:
  - InfoNCE / NT-Xent loss
  - Supervised contrastive loss
- Apply only to non-ghost spacepoints

### 2.3 Auxiliary Supervised Losses
- **File**: `mae/loss/auxiliary_losses.py`
- SSNet classification (particle ID)
- Keypoint score regression
- Instance segmentation head

### 2.4 Co-Distillation Loss
- **File**: `mae/loss/distillation_loss.py`
- Exponential moving average (EMA) teacher model
- Cosine similarity between student/teacher embeddings

### 2.5 Combined Loss
- **File**: `mae/loss/mae_loss.py`
- Learnable or fixed loss weights
- Aggregates all loss components

## Data Processing

### 3.1 Masking Strategy
- **File**: `mae/data/masking.py`
- Random masking with configurable ratio
- Options for structured masking (spatial, per-particle)

### 3.2 Spacepoint Sampler
- **File**: `mae/data/spacepoint_sampler.py`
- Handles subsampling for memory efficiency
- Strategies detailed in options below

### 3.3 MAE Dataset Wrapper
- **File**: `mae/data/mae_dataset.py`
- Wraps `LArMatchSimChHDF5Dataset`
- Applies masking and sampling

## Training Infrastructure

### 4.1 Training Script
- **File**: `mae/train_mae.py`
- Distributed data parallel support
- wandb logging integration
- Checkpoint saving/loading
- Cosine annealing with warmup

### 4.2 Configuration
- **File**: `mae/config/config_mae.yaml`
- All hyperparameters in YAML format

### 4.3 Engine/Utilities
- **File**: `mae/utils/mae_engine.py`
- Model construction, loss computation, accuracy metrics

---

## Options for Addressing Data Challenges

### Issue 1: Large Number of Spacepoints (200k-600k)

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. Fixed Subsampling** | Randomly sample N spacepoints per event (e.g., 50k-100k) | Simple, predictable memory | May miss important particles |
| **B. Importance Sampling** | Preferentially sample true (non-ghost) spacepoints and boundary regions | Better coverage of physics | Introduces sampling bias |
| **C. Local Patch Processing** | Process spatial patches (like image MAE) and aggregate | Natural for transformers | Loses global context |
| **D. Hierarchical Encoding** | Encode at multiple resolutions, downsample progressively | Captures multi-scale structure | More complex architecture |
| **E. Sparse Attention** | Use linear attention or local windowed attention | Scales O(N) not O(N²) | May reduce quality |
| **F. Sequential Mini-batches** | Process one event across multiple forward passes with gradient accumulation | Full event coverage | Slower training |

**Recommendation**: Start with **B (Importance Sampling)** combined with **E (Sparse/Linear Attention)** for scalability.

### Issue 2: High Ghost-to-True Ratio (10:1 to 20:1)

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. Ghost Filtering** | Pre-filter using existing LArMatch scores | Cleaner data | Propagates upstream errors |
| **B. Stratified Sampling** | Enforce balanced ghost/true ratio (e.g., 2:1) during sampling | Balanced learning signal | May undersample rare cases |
| **C. Ghost-Aware Masking** | Only mask/reconstruct true spacepoints | Focus on physics signal | Ignores ghost structure |
| **D. Ghost Classification Auxiliary** | Add auxiliary loss for ghost vs true | Joint learning | Adds complexity |
| **E. Curriculum Learning** | Start with true-only, gradually add ghosts | Easier initial learning | Requires careful scheduling |
| **F. Weighted Losses** | Weight reconstruction loss by ghost/true status | Simple integration | May be hard to tune |

**Recommendation**: Use **B (Stratified Sampling)** with **D (Ghost Classification Auxiliary)** to learn to distinguish ghosts while maintaining focus on true spacepoints.

### Issue 3: Variable Spacepoint Count per Particle

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. Particle-Level Sampling** | Sample fixed number of spacepoints per particle | Balanced particle representation | Variable total count |
| **B. Particle Pooling** | Pool spacepoint features to particle-level before contrastive loss | Natural for contrastive task | Loses fine-grained info |
| **C. Weighted Contrastive** | Weight contrastive pairs inversely by particle size | Fair comparison across scales | Complex weighting |
| **D. Hard Negative Mining** | Focus on difficult same/different particle pairs | Better discrimination | May overfit to hard cases |
| **E. Hierarchical Contrastive** | Both point-level and particle-level contrastive losses | Multi-scale learning | More losses to balance |

**Recommendation**: Use **B (Particle Pooling)** with mean/attention pooling for the contrastive loss, combined with **C (Weighted Contrastive)** to handle size imbalance.

---

## Directory Structure

```
larmatchnet/larmatch/mae/
├── README.md                    # (existing)
├── IMPLEMENTATION_PLAN.md       # This file
├── __init__.py
├── config/
│   └── config_mae.yaml
├── models/
│   ├── __init__.py
│   ├── mae_backbone.py          # Refactored image encoder
│   ├── position_encoding.py     # 3D sinusoidal encoding
│   ├── transformer_encoder.py   # Encoder blocks
│   ├── mae_decoder.py           # Shallow decoder
│   └── spacepoint_mae.py        # Main MAE model
├── loss/
│   ├── __init__.py
│   ├── reconstruction_loss.py
│   ├── contrastive_loss.py
│   ├── auxiliary_losses.py
│   ├── distillation_loss.py
│   └── mae_loss.py              # Combined loss
├── data/
│   ├── __init__.py
│   ├── masking.py               # Masking strategies
│   ├── spacepoint_sampler.py    # Sampling strategies
│   └── mae_dataset.py           # Dataset wrapper
├── utils/
│   ├── __init__.py
│   └── mae_engine.py            # Training utilities
└── train_mae.py                 # Main training script
```

---

## Implementation Priority

1. **Phase 1 - Core MAE**: Position encoding, transformer encoder/decoder, reconstruction loss, basic masking
2. **Phase 2 - Sampling**: Implement importance sampling and stratified ghost/true sampling
3. **Phase 3 - Contrastive**: Contrastive loss with particle pooling
4. **Phase 4 - Auxiliary Tasks**: SSNet, keypoint auxiliary losses
5. **Phase 5 - Advanced**: Co-distillation, sparse attention, curriculum learning
