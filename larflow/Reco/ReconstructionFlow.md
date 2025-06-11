# LArFlow Reconstruction Flow Diagram

## High-Level Flow

```mermaid
graph TD
    A[LArMatch Network Output<br/>3D Spacepoints] --> B[Stage 1: Spacepoint Preparation]
    B --> C[Stage 2: Keypoint Reconstruction]
    C --> D[Stage 3: Fragment Clustering]
    D --> E[Stage 4: Vertex & Prong Building]
    E --> F[Stage 5: Kinematics & PID]
    F --> G[Stage 6: Selection Variables]
    G --> H[Final Output:<br/>NuVertexCandidate]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#9f9,stroke:#333,stroke-width:2px
```

## Detailed Algorithm Flow

```mermaid
graph TD
    %% Stage 1: Spacepoint Preparation
    A1[Raw LArMatch Hits] --> B1[SplitHitsBySSNet<br/>Label with shower scores]
    B1 --> C1[KeypointFilterByWCTagger<br/>Separate cosmic/in-time]
    C1 --> D1[In-time Hits]
    C1 --> D2[Cosmic Hits]
    
    D1 --> E1[SplitHitsBySSNet<br/>Track/Shower separation]
    E1 --> F1[Track Hits]
    E1 --> F2[Shower Hits]
    
    F1 --> G1[ChooseMaxLArFlowHit<br/>Pixel deduplication]
    F2 --> G2[ChooseMaxLArFlowHit<br/>Pixel deduplication]
    
    G1 --> H1[maxtrackhit_wcfilter]
    G2 --> H2[maxshowerhit]
    
    D2 --> E2[SplitHitsBySSNet<br/>Track/Shower separation]
    E2 --> F3[Cosmic Track Hits]
    F3 --> G3[ChooseMaxLArFlowHit]
    G3 --> H3[offtrigger_maxtrackhit]
    
    %% Stage 2: Keypoint Reconstruction
    A1 --> K1[KeypointReco x6<br/>Nu/Track/Shower KPs]
    K1 --> K2[Keypoint Filtering<br/>Thrumu projection check]
    K2 --> K3[In-time Keypoints]
    K2 --> K4[Cosmic Keypoints]
    
    %% Stage 3: Fragment Clustering
    H1 --> L1[ProjectionDefectSplitter<br/>Track fragments]
    H2 --> L2[ShowerRecoKeypoint<br/>Shower clusters]
    H3 --> L3[ProjectionDefectSplitter<br/>Cosmic fragments]
    
    L1 --> M1[trackprojsplit_wcfilter]
    L2 --> M2[showerkp clusters]
    L3 --> M3[trackprojsplit_offtrigger]
    
    %% Stage 4: Vertex Building
    K3 --> N1[NuVertexMaker]
    M1 --> N1
    M2 --> N1
    
    N1 --> O1[Vertex Candidates]
    
    O1 --> P1[NuTrackBuilder<br/>Assemble tracks]
    O1 --> P2[NuVertexShowerReco<br/>Build showers]
    
    P1 --> Q1[Track Compression]
    P2 --> Q2[Shower/Track Overlap Check]
    
    Q1 --> R1[NuVertexAddSecondaries]
    Q2 --> R1
    
    R1 --> S1[NuVertexRestoreKPHits]
    
    %% Final stages
    S1 --> T1[Kinematics Calculation]
    T1 --> U1[PID & Selection Variables]
    U1 --> V1[Final NuVertexCandidate]
    
    style A1 fill:#f9f,stroke:#333,stroke-width:2px
    style V1 fill:#9f9,stroke:#333,stroke-width:2px
```

## Data Products at Each Stage

### Input Data
- **larmatch**: Raw 3D spacepoints from neural network
- **ubspurn_plane[0,1,2]**: SSNet track/shower scores  
- **thrumu**: WireCell cosmic tagger images

### Stage 1 Output: Filtered Spacepoints
```
├── maxtrackhit_wcfilter     # In-time track hits (deduped)
├── maxshowerhit             # In-time shower hits (deduped)
├── offtrigger_maxtrackhit   # Cosmic track hits (deduped)
└── [intermediate products]
    ├── taggerfilterhit      # All in-time hits
    ├── taggerrejecthit      # All cosmic hits
    └── ssnetsplit_*         # Pre-deduplication splits
```

### Stage 2 Output: Keypoints
```
├── keypoint                 # In-time keypoint candidates
│   ├── Type 0: Nu vertex
│   ├── Type 1: Track start  
│   ├── Type 2: Track end
│   ├── Type 3: Shower start
│   ├── Type 4: Michel
│   └── Type 5: Delta
└── keypointcosmic           # Cosmic-tagged keypoints
```

### Stage 3 Output: Fragments
```
├── trackprojsplit_wcfilter  # In-time track segments
├── trackprojsplit_offtrigger # Cosmic track segments  
├── showerkp                 # Shower clusters with keypoints
└── shortproton              # Short proton candidates
```

### Stage 4 Output: Assembled Particles
```
NuVertexCandidate
├── pos[3]                   # 3D vertex position
├── track_v                  # Vector of larlite::track
├── shower_v                 # Vector of larlite::shower
├── track_len_v              # Track lengths
├── shower_plane_pixsum_vv   # Shower energies by plane
└── [cluster associations]   # Which clusters used
```

### Stage 5-6 Output: Physics Quantities
```
NuSelectionVariables
├── approx_vis_energy_MeV    # Total visible energy
├── max_proton_pid           # Best proton score
├── dist2truevtx             # MC truth matching
├── [prong variables]        # Per track/shower
└── [vertex variables]       # Vertex quality
```

## Algorithm Dependencies

### External Dependencies
- WireCell cosmic tagger (thrumu images)
- SSNet particle classification (ubspurn images)  
- LArMatch network scores (in spacepoint features)

### Internal Dependencies
1. Keypoints depend on filtered spacepoints
2. Fragment clustering uses keypoints for vetoing
3. Vertex building requires both keypoints and fragments
4. Track/shower builders need vertex seeds
5. Kinematics requires completed prongs
6. Selection variables need all reconstruction complete

## Configuration Parameters

### Key Thresholds
- Keypoint score threshold: 0.5
- LArMatch hit threshold: 0.5
- DBSCAN clustering distance: 0.7 cm (keypoints), 1.0 cm (tracks)
- Minimum cluster size: 10 hits
- Cosmic pixel sum threshold: 50.0

### Reconstruction Versions
- Version 1: Original algorithm set
- Version 2: Updated with improved shower reconstruction

## Performance Considerations

### Computational Bottlenecks
1. DBSCAN clustering in ProjectionDefectSplitter
2. Track fitting in NuTrackBuilder
3. Shower cone matching in NuVertexShowerReco

### Memory Usage
- Spacepoint containers can be large (>100k points)
- Multiple copies made during filtering stages
- Debug modes save additional intermediate products

## Debugging Tools

### Stop Points
Enable via KPSRecoManager methods:
- After spacepoint prep: See filtered hits
- After keypoint reco: Visualize keypoints
- After clustering: Check fragments
- After track building: Inspect tracks

### Visualization Products
When debug stops enabled:
- All intermediate hit containers saved
- Cluster + PCA axis pairs for 3D viz
- Keypoint locations and types
- Track trajectory points