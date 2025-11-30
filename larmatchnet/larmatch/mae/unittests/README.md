# UNIT TESTS

The scripts in this folder are meant to provide ways to test/verify the behavior of different parts of the lartpc spacepoint MAE training.

## `load_mae_backbone.py`

This is meant to test/debug the loading of the MAE Encoder backbone.
We load, as a test, the MicroBooNE larmatch 2D sparse submanifold convolutional UNet.

The script assumes that the model file, `larmatch_ubprod_ckpt78k_slimmed.pt`,
and config file, `config_test_mae_backbone.yaml`, is in the folder.