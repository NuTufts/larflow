# larflow/Ana

This module is dedicated to:
* measuring quantities useful for decisions, heuristics for the larflow reconstruction, and
* measuring the performance of different aspects of the reconstruction.

# Analysis Routines

## `kpsreco_vertexana`

Assembles statistics to evaluate the performance of the vertexer using Keypoint network outputs.

## `dedx_larmatch_from_true_tracks`

Assembles statistics to plot the dQ/dx vs. residual range for muons and protons.
Uses the true path of primary muon and protons after space charge corrections.
The larmatch points are projected onto this path to give us the residual range.
The true direction of the path is used to estimate the dx for a given larmatch hit.
To get the charge, larmatch hits are projected into the image and a 5-pixel wide window
in the tick direction and centered around the projected pixel is summed.






