# LArMatch

Classification of matched pixels across wire planes.

To make data, the script to run is:

`run_lardata2hdf5.py`

This converts input larcv and larlite data classes into numpy arrays stored into an hdf5 file.

To view the output of the training data, use the Jupyter notebook

`view_larmatch_hdf5_data.ipynb`

# LArMatch input/output

LArMatch takes in three wireplane images from the MicroBooNE TPC (expandable to SBND and ICARUS TPCs).
We also need a "BadChannel" Image. This labels the pixels where the wires are considered unresponsive or noisy.

These input images are used to create a set of possible spacepoints.

The LArMatch network then produces the following outputs per spacepoint:

1. True/Ghost spacepoint classification score
2. Keypoint scores for 6 different classes: track start, track end, shower start, delta start, michel electron start
3. 5 particle-type classification score: electron, gamma, muon, proton, pion(+other meson)
4. Unit vector for the momentum of the particle at that location

# Truth labels

The script `run_lardata2hdf5.py` provides 2D numpy arrays.
Each numpy array provides information for the N spacepoints in the event.





