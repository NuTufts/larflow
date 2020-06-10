# LArMatch Network

Folder contains python modules and scripts for defining, training, deploying
the larmatch and keypoint networks.

## Network modules

### `larmatch.py`

File contains larmatch module.

The larmatch network's purpose is to classify possible 3D spacepoints as either good or bad.
A candidate spacepoint consists of:
* a 3D location in the detector and
* the 3 pixels in the wire-plane image that this 3D point projects onto, where
* the pixels occur at the same row (i.e. time), but different columns corresponding to wires on each plane.

We classify spacepoints as good or bad using feature vectors for the pixel in each plane.
These feature vectors will embed the information in the area of each pixel.
This means we must
* generate feature vectors for each (above threshold) pixel on each image
* extract the 3 feature vectors, one for each pixel on each plane, for a given space point
* concat the feature vector
* make a score, between [0,1], as to whether the space point is good (1) or bad (0).

The larmatch module contains the following layers to generate the feature vectors.
* stem: 3x3 sparse-submanifold convolution layer (sparseconv layer)
* unet: u-net composed of sparseconv residual layers + skip connections
* resnet: additional sparseconv resnet layers
* feature: 1x1 sparseconv layer taking number of resnet features to number of output features
The method to generate features from image data is `forward_features`.
This takes in sparsified image data and outputs feature vectors for each pixel.

To extract the feature vectors from each plane and concact them for each space point,
we use the `extract_features` method.
This method requires having data that tells us what pixels to group together for each space point.
This data is made outside of the network code, using the methods in the `larflow::PrepMatchTriplet` class
found in the `larflow/larflow/PrepFlowMatchData` folder of this repository.

To producer `larmatch` scores, which classifies spacepoints as being good or bad,
use the `classify_triplets` methods.
We expect to move this method into it's own module.

## Loading data

* `load_larmatch_kps.py`: the current data loader used in the `train_kps_larmatch.py` script.
  Loads data for larmatch, keypoint, and ssnet training purposes.

## Loss Calculations

* `loss_kps_larmatch.py`: the current loss calculation module. used to calculate larmatch+keypoint loss.
  Can handle SSNet loss, but not used at this time.

## Training scripts

* `train_kps_larmatch.py`: this is the latest training script and the one that should be used for training or as the starting point for extensions.
* `train_larmatch.py`: this is a deprecated training script. kept around as we used this version to train the network used for Neutrino 2020 resuts.
   Will eventually archive this script.

## Deploy (inference) scripts

* `deploy_kps_larmatch.py`: use this script to generate spacepoints and larmatch+keypoint scores for these spacepoints.

## Grid scripts

The `grid_deply_scripts` folder contains scripts to run `deploy_kps_larmatch.py` on the Tufts cluster.

## Making training data

The `dataprep` folder containers scripts to prepare training+validation data files.

