# LArFlow Deploy scripts

Scripts to deploy and process images go here.

* run_larflow_wholeview.py: Process larcv files containing whole images. Uses larcv processes, UBSplitDetector, and, UBLArFlowStitcher,
  to first divide image into 832x512 subimages and then remerge them.
* run_larflow_precropped.py: Process larcv files containeing pre-cropped images. Useful for per-subimage analysis.

