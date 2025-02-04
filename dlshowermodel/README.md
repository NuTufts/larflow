# DLShowerModel

This folder contains training and deployment scripts for development
of a transformer-based shower assembly model.

Initial sketch of model:

We use larmatch spacepoints. These are selected based on the larmatch ghost/real score.
It contains a 48-dimension feature vector from the 3x16 wire plane feature vector from
the wire plane pixels the spacepoint derives from.

We also use the ssnet shower/track labels to isolate clusters of shower-label points.
We use symmatric dynamic sampling method to isolate some M-number of spacepoint
feature vectors (out of N>M total spacepoints in the cluster) to build a representation
of the cluster. This involves passing the M vectors into a self-attention transformer
with a sum aggregator to generate a vector representation for a given cluster.

The clusters are then passed into another self-attention transformer to learn the
pair-wise affinity for each cluster. The output of the last layer is

softmax_{Nc}( QK^T )

with each row being a vector of length Nc+1.  We use the argmax of this vector to indicate if the cluster
should be assigned as a daughter cluster to a shower trunk.

## Data preparation

The source of the training data are MicroBooNE simulation files.
These are referred internally as the "dlmerged" files.
Each file contains data for several simulated events where the
detector captures an image in-time with a neutrino interaction
that occurs somewhere in the cryostat or TPC.
Each event contains the captured wire-plane images along with meta-data from the simulation
that gives us information as to the true trajectories of the particles made
by the neutrino interaction. Most of the files will have cosmic particle trajectories
taken from real data (recorded when the beam was not on).

We use the LArMatch network and tools for parsing the simulation meta-data to
provide us with spacepoints and ground-truth information for training the network.

To process the input "dlmerged" files, use the program:

`dataprep/deploy_larmatchme.py`

### Spacepoint pre-processing

The goal of our model is to cluster together EM showers.
Therefore, we must take the low-level spacepoint data and perform some pre-processing
steps to make shower clusters that the model will be tasked with grouping together properly.



We run larmatch and then run preprocessing similar to lantern reco:

1. pick out shower points
2. mask around keypoints
3. run DBScan

Then we have to save:

1. larmatch spacepoints with feature vector and plane charge info
2. shower start keypoints
3. cluster assignment for each point
4. pc-axis for each cluster


