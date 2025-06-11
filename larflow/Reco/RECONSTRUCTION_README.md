# LArFlow Reconstruction Module

This directory contains the reconstruction algorithms that process LArMatch network outputs into physics objects for neutrino interaction analysis.

## Overview

The reconstruction chain transforms raw 3D spacepoints from the LArMatch neural network into complete neutrino interaction candidates with characterized tracks and showers. The process is orchestrated by the `KPSRecoManager` class.

## Quick Start

```cpp
// Basic usage
#include "larflow/Reco/KPSRecoManager.h"

// Create manager with output file
larflow::reco::KPSRecoManager reco_manager("output_ana.root", 2, "RecoManager");

// Configure input
reco_manager.set_spacepoint_input_container_name("larmatch");
reco_manager.set_spacepoint_input_datatype_name("larflow3dhit");

// Enable MC truth (optional)
reco_manager.saveEventMCinfo(true);

// Process event
reco_manager.process(iolcv, ioll);

// Access results
auto& nu_candidates = reco_manager.get_mutable_output_candidates();
```

## Documentation

- [KPSRecoManager Documentation](KPSRecoManager_Documentation.md) - Main reconstruction manager
- [Reconstruction Flow Diagram](ReconstructionFlow.md) - Visual flow and data products
- [Algorithm Documentation](AlgorithmDocumentation.md) - Detailed algorithm descriptions

## Key Components

### Hit Processing
- **SplitHitsBySSNet** - Classifies hits as track/shower using SSNet scores
- **ChooseMaxLArFlowHit** - Reduces hit multiplicity per pixel
- **KeypointFilterByWCTagger** - Separates cosmic and in-time hits

### Keypoint Detection  
- **KeypointReco** - Finds vertices, track ends, shower starts from network scores
- **KPCluster** - Data structure for reconstructed keypoints

### Clustering
- **ProjectionDefectSplitter** - Segments tracks into straight pieces
- **ShowerRecoKeypoint** - Builds shower clusters from keypoints
- **ShortProtonClusterReco** - Identifies short heavily-ionizing tracks

### Vertex Formation
- **NuVertexMaker** - Creates neutrino interaction candidates
- **NuVertexCandidate** - Data structure for interaction hypothesis
- **ClusterBookKeeper** - Tracks cluster usage across vertices

### Particle Building
- **NuTrackBuilder** - Assembles track fragments into particles
- **NuVertexShowerReco** - Reconstructs EM showers from vertices
- **NuVertexAddSecondaries** - Adds delta rays and recoils

### Refinement
- **PostNuCheckShowerTrunkOverlap** - Removes track/shower overlaps
- **NuVertexRestoreKPHits** - Recovers wrongly vetoed hits
- **CompressRecoTrack** - Optimizes track representation

### Analysis
- **NuTrackKinematics** - Calculates track momentum and angles
- **NuShowerKinematics** - Determines shower energy and direction
- **NuSelectionVariables** - Computes event selection metrics

## Output Data Products

### Primary Output: NuVertexCandidate
```cpp
struct NuVertexCandidate {
    float pos[3];                    // 3D vertex position
    std::vector<larlite::track> track_v;   // Reconstructed tracks
    std::vector<larlite::shower> shower_v;  // Reconstructed showers
    // ... additional members
};
```

### Analysis Tree Variables
- Event indices (run, subrun, event)
- Vertex candidates with full reconstruction
- Selection variables for physics analysis
- Optional: MC truth matching information

## Configuration Options

### Reconstruction Versions
- Version 1: Original algorithm set
- Version 2: Improved shower reconstruction (recommended)

### Debug Modes
```cpp
// Stop at intermediate stages for visualization
reco_manager.debug_stop_at_spacepoint_prep(true);
reco_manager.debug_stop_at_keypoint_reco(true);  
reco_manager.debug_stop_at_subclustering(true);
```

### Output Control
```cpp
// Minimize file size
reco_manager.minimze_output_size(true);

// Save only selected vertices
reco_manager.saveSelectedNuVerticesOnly(true);
```

## Algorithm Development

### Adding New Algorithms

1. Create new class inheriting from `larcv::larcv_base`
2. Implement `process()` method
3. Add to appropriate stage in KPSRecoManager
4. Update flow documentation

### Testing Algorithms
```cpp
// Use visualization script
python vis_kpsreco.py output_file.root

// Enable debug output
algorithm.set_verbosity(larcv::msg::kDEBUG);
```

## Common Tasks

### Tuning Keypoint Detection
Edit thresholds in `KPSRecoManager::recoKeypoints()`:
```cpp
_kpreco_nu.set_keypoint_threshold(0.2, 0);  // Lower = more keypoints
_kpreco_nu.set_min_cluster_size(10, 0);     // Smaller = more keypoints
```

### Adjusting Clustering
In `KPSRecoManager::clusterSubparticleFragments()`:
```cpp
_projsplitter.set_dbscan_pars(maxdist, minsize, maxkd);
// maxdist: larger = fewer, bigger clusters
```

### Modifying Vertex Formation
In `NuVertexMaker`:
```cpp
_cluster_type_max_impact_radius[kTrack] = 10.0;  // cm
_cluster_type_max_gap[kTrack] = 20.0;  // cm
```

## Troubleshooting

### No vertices found
1. Check keypoint thresholds aren't too high
2. Verify WireCell tagger isn't rejecting everything
3. Enable debug stops to see intermediate products

### Broken tracks
1. Increase DBSCAN distance in ProjectionDefectSplitter
2. Check track fragment connectivity in NuTrackBuilder
3. Visualize with track builder debug output

### Missing shower energy
1. Verify shower/track classification threshold
2. Check shower cone parameters
3. Look for gaps in shower reconstruction

## Performance Notes

- Full reconstruction takes ~1-10 seconds per event
- Memory usage scales with number of spacepoints
- Clustering algorithms are main computational bottleneck

## Future Improvements

1. **Shower Reconstruction**: Implement cascade modeling
2. **Track Building**: Add Kalman filter fitting
3. **Vertex Optimization**: Implement chi-square minimization
4. **Particle ID**: Integrate CNN-based PID
5. **Code Structure**: Refactor to plugin architecture

## References

- LArMatch network paper: [arXiv:XXXX.XXXXX]
- SSNet paper: [arXiv:1808.07269]
- WireCell reconstruction: [arXiv:1802.08709]

## Contact

For questions about the reconstruction code:
- Create an issue on GitHub
- Contact the LArFlow development team