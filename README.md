# LArFlow: prediction pixel correspondence between LArTPC wireplane images

This repository contains the code for developing the larflow network.

## Dependencies

### Not included in repo

* ROOT (6.12/04 known to work)
* pytorch (0.4)
* numpy (1.14.03 known to work)
* tensorboardX (from [here](https://github.com/lanpa/tensorboard-pytorch))
* tensorboard
* cuda (9.1 known to work)
* (to do: add missing)

### Included as submodules

* LArCV2 (tufts_ub branch): library for representing LArTPC data as images along with meta-data. Also, provides IO.
* larlite: classes for meta-data. Also provides access to constants for the UB detector geometry and LAr physics
* larcvdataset: wrapper class providing interface to images stored in the larcv format. converts data into numpy arrays for use in pytorch

## Setup

* clone this repository
* from the top repository directory (where this README is located), run configure.sh
* when setting up for the first time, run build.sh
* for each time using the code, 
