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

It's hacky.  You run dev.cxx and make sure to pipe the standard out to text.

    ./dev > out.data

At the end of dev.cxx, it will dump out the 3D hits to standard out. You can isolate them from `out.data` by

    cat out.data | grep hitidx > hits.dat

Then you can use visdev.py to make a 3D plot. Go into the file and change the name of the text file it reads in.

    python visdev.py

That's it.

There are some options now, hard-coded of course.  You can look for the following booleans in `dev.cxx`:

* `use_truth`: will use the truth flow to make the hits
* `use_hits`: will use `gaushit` hits. If false, will make fake hits from the individual above threshold pixels

To turn off the visualization change `kVISUALIZE` to `false`.

## TO DO

RECO

* [done] import opencv-based contour clustering
* [done] segment matching
* [done] 3d hit generation
* Both Y->U and Y->V
* deciding which flow prediction to use for final set of 3d hits
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
* use Katie's output. how?

ENDPOINT MERGER

* prepare Josh's output into whole view image
* use Josh's output (will have false positive end-points)
