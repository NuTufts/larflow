# LArFlow Post-processor

Goal is take larflow input and use it to provide 3D reconstruction of hits in the detector.
The 3D hits can then be used to find and tag cosmic ray muons.

## Roadmap

* opencv-based contour tools: generate contour primitives based on small, straight cluster of pixels
* segment plane matching: larflow information is used to match clusters across planes, breaking them as well, if needed.
* 3D hit generation: matched-clusters are then used to make 3D hits by associating 2D hits to the matched-clusters
* cosmic tracking: 3D hits near the edges are used as seeds to find through-going cosmics tracks, then stopping ones
* flash-matching: use to determine true-x coordinate of hits, filter out-of-spill hits
* collect in-time hits: use corresponding pixels as seed for neutrino vertex finding and DLLEE analysis
* final output: pixel-tags for cosmic and in-time hypotheses

## Contents

* TaggerContourTools. taken from larlitecv dllee analysis. Generates primitive clusters for matching.

## Temp: how to get pictures from current dev.cxx state

First, run `dev`.

    ./dev

Then you can use `flowout_vis.py`.

    python flowout_vis.py output_flowmatch_larlite.root


You will also be able to use a ROOT-based visualization script (to be implemented).

That's it.

There are some options now, hard-coded of course.  You can look for the following booleans in `dev.cxx`:

* `use_truth`: will use the truth flow to make the hits
* `use_hits`: will use `gaushit` hits. If false, will make fake hits from the individual above threshold pixels

To turn off the visualization inside `dev.cxx` by changing `kVISUALIZE` to `false`.
This debugging visualization plots flow and contour information per subimage.

## TO DO

RECO

* [done] import opencv-based contour clustering
* [done] segment matching
* [done] 3d hit generation
* [done] Both Y->U and Y->V
* [done] deciding which flow prediction to use for final set of 3d hits
* [done] integrate endpoint+SSNet
* integrate infill
* 3d clustering
* marking endpoint of clusters (extrema of pixels along the 1st PCA axis?)
* ID boundary charge, i.e. (y,z) edges
* ID endpoints of clusters consistent with anode/cathode crossing, i.e. (x edges)
* thru-mu reco: astar algo with cylinder selection
* stopmu reco: stitch clusters into tracks
* flash-match: mark highly-consistent thrumu/stopmu tracks with out-of-time flash, veto
* cluster remaining, non-tracked charge
* optimize combinations of clusters and provide ranked list of those that are consistent with the in-time flash

ANA

* repurpose old-tagger's analysis scripts to get locations of true endpoints
* analyze eff. and purity of end-point ID
* analyze number of missing true hits and number of ghost hits
* first goal? >80% cosmic tagged with >80% 1e1p neutrino pixel eff
* purity targets?

INFILL MERGER

* prepare Katie's output into whole view image
* feed this "filled" image into the contouring tool in hopes of building better contours
* this also is given to the flowcontourmatch algo
* would be good to mark the hits where we've paired with the dead region

TWO FLOW MERGE

* choose y2u or y2v prediction based on the quality. default to y2u(?).
* also should choose the one that doesn't go into a dead-region

ENDPOINT MERGER

* prepare Josh's output into whole view image
* use Josh's output (will have false positive end-points)

## VISUALIZATION UTILITY

Some scripts are available to look at the output of `dev`.

### Pyqtgraph-based tool

Uses python package pyqtgraph which provides a GUI tool along with some opengl functions to display 3D objects.
Requires a number of dependencies. On Ubuntu/Linux, it's pretty easy to install (at least on Bionic Beaver, 18.04 LTS).

    sudo apt get python-pyqtgraph

On the Mac, it can get a bunch more involved.

Refer to the `pylard` [readme](https://github.com/twongjirad/pylard) and install the dependencies.
No need to install `pylard` itself.


### ROOT-based tool

Coming

### To Do

* ROOT-based tool (no need to installing pyqtgraph which can be difficut)
* Plot truth tracks
* colorbytruthtrack -- mark pixels close to a true track