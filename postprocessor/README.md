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
