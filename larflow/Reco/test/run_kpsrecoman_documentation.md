# run_kpsrecoman.py Documentation

## Overview

`run_kpsrecoman.py` is the main script for running the LArFlow reconstruction pipeline on data files. It processes LArMatch 3D spacepoints and network outputs to produce fully reconstructed neutrino interaction candidates.

## Usage

```bash
python3 run_kpsrecoman.py -i <dlmerged_file> -l <larmatch_file> -o <output_file> [options]
```

## Required Arguments

- **`-i, --input-dlmerged`** : Input file containing ADC images, SSNet scores, bad channel info, and other network outputs (LArCV format)
- **`-l, --input-larflow`** : Input file containing LArMatch 3D spacepoints (`larflow3dhit` objects in larlite format)
- **`-o, --output`** : Base name for output files (will create multiple output files with different suffixes)

## Optional Arguments

### Event Processing
- **`-n, --num-entries`** : Number of events to process (default: all events)
- **`-e, --start-entry`** : Starting event number (default: 0)
- **`-tb, --tickbackwards`** : Flag if input images have reversed time direction

### MC Options
- **`-mc, --ismc`** : Enable MC truth information processing and matching
- **`--run-perfect-mcreco`** : Run perfect reconstruction using MC truth (requires `--ismc`)
- **`--run-nuvertexshowerreco-mcana-mode`** : Enable detailed MC analysis for shower reconstruction

### Output Control
- **`-p, --products`** : Output product level (default: "rerun")
  - `"rerun"` : Save enough data to rerun downstream analysis
  - `"min"` : Minimal output for visualization
  - `"debug"` : Save all intermediate products
- **`-f, --event-filter`** : Only save selected neutrino vertices
- **`--save-all-keypoints`** : Save all reconstructed keypoints to analysis file

### Algorithm Control
- **`-v, --version`** : Reconstruction version (default: 2, recommended)
- **`-ll, --loglevel`** : Logging verbosity (0=debug, 1=info, 2=normal, 3=warning, 4=error)

### Debug Stop Points
- **`--stop-after-spacepointprep`** : Stop after spacepoint filtering stage
- **`--stop-after-keypointreco`** : Stop after keypoint reconstruction
- **`--stop-after-subclustering`** : Stop after fragment clustering
- **`--stop-after-nutracker`** : Stop after track building

## Output Files

The script creates three output files:

1. **`<output>_larlite.root`** : Contains reconstructed objects in larlite format
   - Tracks, showers, vertices
   - Clusters and keypoints
   - MC truth (if enabled)

2. **`<output>_larcv.root`** : Contains image data in LArCV format
   - Wire ADC images
   - Network score images
   - WireCell tagger images

3. **`<output>_kpsrecomanagerana.root`** : Analysis tree with detailed reconstruction info
   - NuVertexCandidate objects
   - Selection variables
   - Performance metrics
   - MC matching (if enabled)

## Input Data Requirements

### DL Merged File (LArCV)
Required image2d products:
- `wire` : ADC wire plane images
- `thrumu` : WireCell cosmic tagger
- `ubspurn_plane[0,1,2]` : SSNet track/shower scores
- `chstatus` : Bad channel status

Optional (for MC):
- `ancestor`, `segment`, `instance` : Truth labels
- `larflow` : Flow truth

### LArMatch File (larlite)
Required products:
- `larflow3dhit` with tree name "larmatch" : 3D spacepoints from network

Optional (for MC):
- `mctrack`, `mcshower` : MC particles
- `mctruth` : Generator information

## Examples

### Basic Data Reconstruction
```bash
python3 run_kpsrecoman.py \
    -i merged_dlreco.root \
    -l larmatch_output.root \
    -o reco_output.root
```

### MC with Truth Matching
```bash
python3 run_kpsrecoman.py \
    -i merged_dlreco_mc.root \
    -l larmatch_output_mc.root \
    -o reco_output_mc.root \
    --ismc \
    --run-perfect-mcreco
```

### Debug First 10 Events
```bash
python3 run_kpsrecoman.py \
    -i merged_dlreco.root \
    -l larmatch_output.root \
    -o debug_output.root \
    -n 10 \
    -p debug \
    -ll 0 \
    --stop-after-subclustering
```

### Production Run with Minimal Output
```bash
python3 run_kpsrecoman.py \
    -i merged_dlreco.root \
    -l larmatch_output.root \
    -o production_output.root \
    -p min \
    -f \
    -v 2
```

## Workflow

1. **Initialization**
   - Creates KPSRecoManager instance
   - Sets up I/O managers for LArCV and larlite
   - Configures output products based on `-p` flag

2. **Event Loop**
   - Loads ADC images and network scores
   - Loads LArMatch spacepoints
   - Runs reconstruction via `recoman.process()`
   - Saves results to output files

3. **Finalization**
   - Closes I/O files
   - Writes analysis tree
   - Reports processing time

## Performance Considerations

- Processing time: ~1-10 seconds per event
- Memory usage: Scales with number of spacepoints
- Debug modes save more data but are slower

## Troubleshooting

### Common Issues

1. **Missing spline file error**
   - Ensure `Proton_Muon_Range_dEdx_LAr_TSplines.root` exists in `larflow/Reco/data/`

2. **Segmentation fault**
   - Check input file compatibility
   - Verify all required trees exist
   - Try running with `-ll 0` for debug output

3. **No vertices found**
   - Check WireCell tagger images aren't empty
   - Verify LArMatch produced spacepoints
   - Use debug stop points to diagnose

### Debug Workflow

1. Start with `--stop-after-spacepointprep` to verify inputs
2. Progress to `--stop-after-keypointreco` to check keypoints
3. Use visualization scripts on output:
   ```bash
   python vis_kpsreco.py debug_output_larlite.root
   ```

## Visualization

After running reconstruction, visualize results with:

```bash
# In larflow/Reco/test/
python vis_kpsreco.py <output>_larlite.root

# For keypoints only
python vis_keypointreco.py <output>_larlite.root

# For truth comparison (MC only)
python vis_trueshowerhits.py <output>_larlite.root
```

## Advanced Usage

### Custom Spacepoint Container
If using different spacepoint tree name:
```python
# Edit line 55 in script
input_spacepoint_container_name = "your_tree_name"
```

### Adding Custom Outputs
To save additional products, add before event loop:
```python
io.set_data_to_write("larflow3dhit", "your_product")
```

## Related Scripts

- `vis_kpsreco.py` : 3D visualization of reconstruction
- `run_mcpixelpgraph.py` : MC-based analysis
- `run_perfectreco_only.py` : Truth-only reconstruction

## Notes

- Version 2 reconstruction is recommended for better shower reconstruction
- MC mode significantly increases output size
- Debug stops are useful for algorithm development but pause execution