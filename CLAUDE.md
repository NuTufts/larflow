# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LArFlow is a deep learning framework for 3D neutrino interaction reconstruction in Liquid Argon Time Projection Chambers (LArTPC). The core network (LArMatch) predicts pixel correspondence between wireplane images to reconstruct 3D spacepoints.

## Environment Setup

1. Navigate to the parent `ubdl` directory
2. Setup environment:
   ```bash
   source setenv_py3.sh     # Python 3 environment
   source configure.sh      # Configure all submodules
   ```
3. Return to larflow directory:
   ```bash
   cd larflow
   source configure.sh      # LArFlow-specific configuration
   ```

## Build Commands

### C++ Libraries (CMake)
```bash
# Configure build
cd build
cmake ..

# Build all
make -j$(nproc)

# Install to build directory
make install
```

### Running Tests
```bash
# C++ unit tests (if built)
cd build
ctest

# Python tests for neural networks
cd larmatchnet/larmatch
python3 test_loading_from_cache.py
```

## LArMatch Network Commands

### Training
```bash
# Setup environment for training
cd larmatchnet
source set_pythonpath.sh

# Single GPU training
cd larmatch
python3 train_dist_larmatchme.py --config config/config_larmatchme.yaml --gpus 1

# Multi-GPU training
python3 train_dist_larmatchme.py --config config/config_larmatchme.yaml --gpus 4
```

### Deployment (Inference)
```bash
# Deploy on GPU
python3 deploy_larmatchme.py --config-file config/config_larmatchme_deploygpu.yaml \
    --supera input.root --output output_prefix --device-name "cuda:0"

# Deploy on CPU
python3 deploy_larmatchme.py --config-file config/config_larmatchme_deploycpu.yaml \
    --supera input.root --output output_prefix --device-name "cpu"
```

## High-Level Architecture

### Core Components

1. **larmatchnet/** - Neural network implementation
   - `larmatch/` - Main LArMatch network (MinkowskiEngine-based sparse 3D CNN)
   - `larvoxel/` - Voxelized network variants
   - Training uses distributed data parallel (DDP) for multi-GPU
   - Outputs: 3D spacepoints with scores and features

2. **larflow/** - C++ reconstruction libraries
   - `PrepFlowMatchData/` - Prepares spacepoint data from network outputs
   - `KeyPoints/` - Identifies physics keypoints (vertices, track ends, shower starts)
   - `Reco/` - Full event reconstruction algorithms
   - `CRTMatch/` - Cosmic ray tagger integration

3. **dlshowermodel/** - Graph neural network for shower reconstruction
   - Uses transformer and GAT architectures
   - Specialized for electromagnetic shower analysis

### Data Flow

1. Input: LArTPC wireplane images (larcv format)
2. LArMatch network → 3D spacepoints with features
3. Post-processing → Physics objects (tracks, showers, vertices)
4. Output: ROOT files with reconstruction results

### Key Design Patterns

- Networks use configuration YAML files for hyperparameters
- C++ code follows ROOT/LArSoft conventions
- Python code uses PyTorch/MinkowskiEngine for sparse 3D operations
- Mixed C++/Python workflow with ROOT I/O interfaces

### Important Dependencies

- MinkowskiEngine (sparse 3D convolutions)
- PyTorch (deep learning framework)
- ROOT (I/O and physics analysis)
- larcv/larlite (LArTPC data formats)
- OpenCV (image processing)
- Eigen3 (linear algebra)
- xgboost (BDT-based selections in reconstruction)

## Current Branch Notes

Branch `dlgen2_larmatchhdf5_retrain` focuses on retraining with HDF5 data format support. Key changes likely in:
- `larmatchnet/larmatch/larmatch_dataset.py` - data loading
- `dlshowermodel/data/` - HDF5 reader implementations