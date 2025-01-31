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

We run larmatch and then run preprocessing similar to lantern reco:

1. pick out shower points
2. mask around keypoints
3. run DBScan

Then we have to save:

1. larmatch spacepoints with feature vector and plane charge info
2. shower start keypoints
3. cluster assignment for each point
4. pc-axis for each cluster


