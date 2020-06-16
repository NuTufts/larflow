# Voxelizer

Module containing utilities to voxelize feature vector output of the larmatch network.

This is to feed data into 3D convolutional neural networks.

We have two ways to do this:

1) pooling method
2) graphSAGE style pooling

First we do the simple pooling method (ave. or max).

Either way, we can have a pre-processing module that produces a "voxelization" plan
which tells what entries in the output match proposal matrix to merge.