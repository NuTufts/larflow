# Test folder

Contains development scripts for the Larmatch reconstruction chain.

Inputs of the reconstruction:
* larmatch spacepoints with associated scores for true vs. ghost points (larflow3dhit_larmatch_tree)
* larmatch keypoint scores for each spacepoint for 6 classes: nu, track start, track end, shower start, delta, michel (also in larflow3dhit_larmatch_tree)
* 2D ssnet scores 
* Wirecell in-time versus out-of-time pixel image (image2d_thrumu_tree)

Stages of the reconstruction:
* keypoint formation: algos include `KeypointReco`
* splitting of keypoints using in-time/out-of-time and track/shower
* track sub-cluster splitting into 'straight-line' segments