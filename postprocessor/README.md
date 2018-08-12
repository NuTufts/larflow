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