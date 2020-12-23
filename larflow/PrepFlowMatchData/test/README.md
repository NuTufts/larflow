# larmatch spacepoint scripts

Scripts to create/test larmatch spacepoint-creation code.

## `run_prepmatchtriplets_wfulltruth.py`

Creates larmatch spacepoints along with truth.
Uses the `TripletTruthFixer` class to repair ground truth as best as it can.
Not expected to be perfect as only 2D-projection truth is saved.
We do our best to provide good labels to the 3D points.

## `vis_prepmatchtriplets.py`

Look at the output of `run_prepmatchtriplets_wfulltruth.py`.

Example output

### larmatch truth: real or ghost point label

![larmatch truth](pngs/corsika_event_larmatch_truth.png?raw=true)

### particle ID label: for semantic segmentation

![larmatch truth](pngs/corsika_event_class_labels.png?raw=true)

### instance ID label: for clustering

![larmatch truth](pngs/corsika_event_instancelabels.png?raw=true)

