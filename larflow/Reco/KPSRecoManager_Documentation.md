# KPSRecoManager Documentation

## Overview

The `KPSRecoManager` class is the main orchestrator for the LArFlow reconstruction pipeline. It manages the complete reconstruction flow from raw LArMatch 3D spacepoints to fully reconstructed neutrino interaction candidates with tracks and showers.

## Class Purpose

This class coordinates multiple reconstruction algorithms in a specific sequence to:
1. Process raw spacepoints from the LArMatch network
2. Identify keypoints (vertices, track ends, shower starts)
3. Create particle fragments (track and shower clusters)
4. Build neutrino interaction candidates
5. Refine and characterize the reconstructed particles
6. Calculate selection variables for physics analysis

## Reconstruction Flow

### Stage 1: Spacepoint Preparation (`prepSpacepoints`)

**Purpose**: Filter and classify spacepoints from LArMatch network output

**Active Algorithms**:
- `SplitHitsBySSNet` - Labels hits with SSNet shower scores
- `KeypointFilterByWCTagger` - Filters hits using WireCell cosmic tagger
- `ChooseMaxLArFlowHit` - Reduces hits by selecting highest score per pixel

**Input**: 
- Raw LArMatch spacepoints (larflow3dhit)
- SSNet track/shower scores
- WireCell thrumu tagger images

**Output**:
- `maxtrackhit_wcfilter` - In-time track hits
- `maxshowerhit` - In-time shower hits
- `offtrigger_maxtrackhit` - Out-of-time track hits

### Stage 2: Keypoint Reconstruction (`recoKeypoints`)

**Purpose**: Identify physics keypoints from network scores

**Active Algorithms**:
- `KeypointReco` instances for:
  - Neutrino vertices (`_kpreco_nu`)
  - Track starts (`_kpreco_trackstart`)
  - Track ends (`_kpreco_trackend`)
  - Shower starts (`_kpreco_shower`)
  - Michel electrons (`_kpreco_michel`)
  - Delta rays (`_kpreco_deltas`)

**Input**: LArMatch hits with keypoint scores
**Output**: 
- `keypoint` - In-time keypoint candidates
- `keypointcosmic` - Cosmic-tagged keypoints

### Stage 3: Sub-particle Fragment Clustering (`clusterSubparticleFragments`)

**Purpose**: Create track and shower fragments for later assembly

**Active Algorithms**:
- `ProjectionDefectSplitter` - Splits track hits into straight segments
- `ShowerRecoKeypoint` - Creates shower clusters using keypoints
- `ShortProtonClusterReco` - Identifies short proton tracks

**Output**:
- `trackprojsplit_wcfilter` - In-time track fragments
- `trackprojsplit_offtrigger` - Cosmic track fragments
- `showerkp` - Shower clusters with keypoints

### Stage 4: Multi-prong Reconstruction (`multiProngReco`)

**Purpose**: Assemble fragments into complete neutrino interactions

**Active Algorithms**:
- `NuVertexMaker` - Creates neutrino vertex candidates
- `NuTrackBuilder` - Builds complete tracks from fragments
- `NuVertexShowerReco` - Reconstructs showers from vertex
- `PostNuCheckShowerTrunkOverlap` - Removes track-shower overlaps
- `NuVertexAddSecondaries` - Adds secondary particles
- `NuVertexRestoreKPHits` - Restores vetoed keypoint hits

**Output**: Complete `NuVertexCandidate` objects with tracks and showers

### Stage 5: Kinematics and PID (`runBasicKinematics`, `runBasicPID`)

**Purpose**: Calculate particle properties and identification

**Active Algorithms**:
- `NuTrackKinematics` - Track momentum and angles
- `NuShowerKinematics` - Shower energy and direction
- `LikelihoodProtonMuon` - Proton/muon discrimination
- `ShowerdQdx` - Shower dQ/dx calculation

### Stage 6: Selection Variables (`makeNuCandidateSelectionVariables`)

**Purpose**: Calculate variables for event selection

**Active Algorithms**:
- `NuSelProngVars` - Prong-level variables
- `NuSelVertexVars` - Vertex quality variables
- `NuSelUnrecoCharge` - Unaccounted charge analysis
- `NuSelCosmicTagger` - Cosmic rejection variables
- `TrackForwardBackwardLL` - Particle ID likelihood

## Currently Active Algorithm Classes

### Core Reconstruction (Always Used):
1. **SplitHitsBySSNet** - Hit classification
2. **KeypointFilterByWCTagger** - Cosmic filtering
3. **ChooseMaxLArFlowHit** - Hit reduction
4. **KeypointReco** (6 instances) - Keypoint finding
5. **ProjectionDefectSplitter** - Track clustering
6. **ShowerRecoKeypoint** - Shower clustering
7. **NuVertexMaker** - Vertex formation
8. **NuTrackBuilder** - Track building
9. **NuVertexShowerReco** - Shower reconstruction
10. **NuVertexAddSecondaries** - Secondary finding
11. **NuVertexRestoreKPHits** - Hit recovery

### Analysis/Refinement (Conditionally Used):
12. **ShortProtonClusterReco** - Short track finding
13. **PostNuCheckShowerTrunkOverlap** - Overlap removal
14. **NuTrackKinematics** - Track properties
15. **NuShowerKinematics** - Shower properties
16. **CompressRecoTrack** - Track compression

### Selection Variables (When Enabled):
17. **NuSelUnrecoCharge** - Missing charge analysis
18. **LikelihoodProtonMuon** - PID calculation
19. **ShowerdQdx** - Shower dE/dx

### MC Analysis (When MC Mode):
20. **PerfectTruthNuReco** - Truth-based reconstruction
21. **LArbysMC** - MC event information

## Deprecated/Commented Out Classes

These appear in the code but are not actively used:
- `NuVertexActivityReco` (commented out)
- `VetoHitClustering` (declared but not used)
- `NuTrackdQdx` (commented out)
- `NuShowerBuilder` (commented out)
- `CosmicTrackBuilder` (function exists but not called)
- `CosmicVertexBuilder` (declared but not used)
- `CosmicProtonFinder` (declared but not used in main flow)
- `NuVertexShowerTrunkCheck` (commented out)
- Various NuSel classes in header but not all used

## Configuration Options

### Debug Stop Points:
- `debug_stop_at_spacepoint_prep()` - Stop after Stage 1
- `debug_stop_at_keypoint_reco()` - Stop after Stage 2
- `debug_stop_at_subclustering()` - Stop after Stage 3
- `debug_stop_at_nutracker()` - Stop after track building

### Output Control:
- `minimze_output_size()` - Reduce intermediate data
- `saveSelectedNuVerticesOnly()` - Save only selected vertices
- `saveEventKeypoints()` - Include keypoints in output
- `runPerfectMCreco()` - Enable MC truth reconstruction

## Data Flow Summary

```
LArMatch Spacepoints → Hit Filtering → Keypoint Finding → 
Fragment Clustering → Vertex Formation → Track/Shower Building → 
Kinematics/PID → Selection Variables → NuVertexCandidate Output
```

## Key Output Products

1. **NuVertexCandidate**: Complete neutrino interaction hypothesis
   - 3D vertex position
   - Track collection with properties
   - Shower collection with properties
   - Associated hits and clusters

2. **NuSelectionVariables**: Physics selection metrics
   - Vertex quality measures
   - Particle identification scores
   - Topology characteristics

## Usage Notes

- The reconstruction version must be set (1 or 2) via constructor or `set_reco_version()`
- Input spacepoint container name can be configured via `set_spacepoint_input_container_name()`
- MC analysis mode should be enabled before processing for truth studies
- Debug stop points are useful for algorithm development and visualization