# LArFlow Reconstruction Module

This directory contains the reconstruction algorithms that process LArMatch network outputs into physics objects for neutrino interaction analysis.

## Overview

The reconstruction chain transforms raw 3D spacepoints from the LArMatch neural network into complete neutrino interaction candidates with characterized tracks and showers. The process is orchestrated by the `KPSRecoManager` class.

## Documentation

### 📚 Main Documentation Files

- **[KPSRecoManager Documentation](KPSRecoManager_Documentation.md)** - Detailed documentation of the main reconstruction manager class
  - Reconstruction flow stages
  - Active vs deprecated algorithms
  - Configuration options and debug features
  
- **[Reconstruction Flow Diagram](ReconstructionFlow.md)** - Visual representation of the reconstruction pipeline
  - Flow diagrams with algorithm connections
  - Data products at each stage
  - Performance considerations

- **[Algorithm Documentation](AlgorithmDocumentation.md)** - In-depth documentation of all active algorithms
  - Detailed algorithm descriptions
  - Parameters and methods
  - Common issues and solutions

- **[Quick Start Guide](RECONSTRUCTION_README.md)** - Getting started with the reconstruction
  - Usage examples
  - Common tasks and troubleshooting
  - Future improvements

## Quick Start

### Running the Reconstruction

The main script to run reconstruction is `test/run_kpsrecoman.py`:

```bash
cd test/
python3 run_kpsrecoman.py -i merged_dlreco.root -l larmatch_output.root -o reco_output.root
```

See **[run_kpsrecoman.py Documentation](test/run_kpsrecoman_documentation.md)** for detailed usage instructions, options, and examples.

### C++ API Usage

```cpp
#include "larflow/Reco/KPSRecoManager.h"

// Create manager with output file
larflow::reco::KPSRecoManager reco_manager("output_ana.root", 2);

// Configure and run
reco_manager.set_spacepoint_input_container_name("larmatch");
reco_manager.process(iolcv, ioll);

// Access results
auto& nu_candidates = reco_manager.get_mutable_output_candidates();
```

## Key Components

- **Hit Processing**: SplitHitsBySSNet, ChooseMaxLArFlowHit, KeypointFilterByWCTagger
- **Keypoint Detection**: KeypointReco (6 instances for different particle features)
- **Clustering**: ProjectionDefectSplitter, ShowerRecoKeypoint
- **Vertex Formation**: NuVertexMaker, NuTrackBuilder, NuVertexShowerReco
- **Analysis**: Track/Shower kinematics, PID, selection variables

