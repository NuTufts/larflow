# LArFlow: prediction pixel correspondence between LArTPC wireplane images

This repository contains the code for developing a full 3D neutrino interaction
reconstruction centered using outputs of a convolutional neural network.

The convolutional neural network aims to provide information
that seeds the reconstruction of particles and interactions.
This includes:

* good 3D space points representing the locations where charge particle tracks passed
through the detector
* associations between the 3D space points and the spatial patterns in
the TPC images where the points project into.
* scores indicating which 3D points are near important, key-points:
  track ends, shower starts, and neutrino interaction vertices.

## Contents

* larmatchnet: definition of network, scripts to train and deploy network
* larflow: c++ libraries providing methods to prepare data, perform downstream reconstruction

Within the `larflow` folder, are the following c++ modules:
* `PrepFlowMatchData`: classes/methods for preparing spacepoint data from TPC images
* `KeyPoints`: classes/methods for preparing keypoint training info using spacepoint data from `PrepFlowMatchData`
* `SpatialEmbed`: classes/methods for preparing spatial embedding training info using spacepoint data from `PrepFlowMatchData`
* `Reco`: downstream reconstruction using output of networks to form candidate neutrino interactions
* `CRTMatch`: tools to combine CRT data with spacepoints and TPC data in order to provide tagged cosmic muon tracks
* `Voxelizer`: voxelize larmatch spacepoints, not finished. intended to help spacepoint output connect to 3D convolutional networks.
* `LArFlowConstants`: constants, enumerations used in the other modules
* `FlowContourMatch`:deprecated tools

Other folders are considered deprecated and need to be cleaned up and archived.

* models: different version of models
* dataprep: scripts to make larflow input and truth images from larsoft files and then prepare crops for training
* training: training scripts
* deploy: take trained models and process test files
* ana: analysis scripts for processed test files
* weights: default location for weights
* testdata: default location for testdata used for development
* utils: utility scripts
* container: script to build Singularity container that will work on tufts grid

## To Do list:

* input arguments to deploy scripts need to be sane. right now they make no sense. [DONE?]
* add script to grab files and store in testdata to be used for deploy and postprocessor [DONE]
* provide filtered weight files [DONE]
* add folder and script to grab development weights [DONE]. scripts should point to these by default [DONE]
* post-processor needs cluster matching to 3D-hit formation [DONE]
* post-processor needs Y->U + Y->V support [Ralitsa/DONE]
* deploy script for the larflow/infill/endpoint [DONE]
* should post-processor merge subimages before after cluster-matching? (ignoring for now)
* post-processor reads larflow[DONE]/infill[DONE]/endpoint[DONE]
* post-processor dbscan clustering [DONE]
* post-processor pca calculation to ID ends [DONE]
* post-processor spectral clustering [Ralitsa in progress]
* post-processor break non-straigt clusters 
* post-processor labels endpoint hits: both by endpoint-net and by 3d-location
* post-processor uses end-points to find thrumus using existing astar+3d-hit-neighborhood mask
* post-processor uses end-points + adrien tracker to find stops
* post-processor flash-matching [Taritree in progress, performing toy truth study]
* post-processor cluster selection [Ralitsa in progress]
* post-processor CROI definition

## Dependencies

### Not included in repo

* ROOT (6.12/04 known to work)
* opencv (3.2.0 known to work)
* pytorch (1.3, 1.4 known to work)
* numpy (1.14.03 known to work)
* tensorboardX (from [here](https://github.com/lanpa/tensorboard-pytorch))
* tensorboard
* cuda (currently using 10.1)
* Eigen 3

UBDL dependencies
* larlite: following X branch
* Geo2D:
* LArOpenCV:
* larcv:
* ublarcvapp:
* cilantros:
* nlohmann/json
* (to do: add missing)

### Included as submodules

* larcvdataset: wrapper class providing interface to images stored in the larcv format. converts data into numpy arrays for use in pytorch (deprecated)
* Cilantro: a library with various Clustering routines w/ C++ API (deprecated)
* Pangolin: a OpenGL viewer package, used by Cilantro (deprecated)

## Setup

### First-time setup

* clone this repository: `git clone https://github.com/NuTufts/larflow larflow`
* setup the submodules, configure environment variables, and build: `source first_setup.sh`
* if you plan to modify any of the submodules, you will need go to the head branch for each submodule. use: `source goto_head_of_submodules.sh`

### Issues building Pangolin (deprecated)

Pangolin depends on GLEW and X11. These can be provided by a package manager.
However, especially for GLEW other versions can be on the system from other libraries like CUDA and/or ROOT.
This can cause compilation errors.

If there are issues you can try the following:

* go into CMakeCache.txt and check the include and library entries for GLEW (search for GLEW).
  Change them to point to the system GLEW. On Ubuntu this will be something like:


      /usr/lib/x86_64-linux-gnu/libGLEW.so for the LIB dir and /usr/include for the INC dir


  If you do this, remove the directory CMakeFiles and run `make clean`. Then run `cmake .` and finally `make`.
  
* go into `Pangolin/include/pangolin/gl/glplatform.h` and change `<GL/glew.h>` to `/usr/include/GL/glew.h` to hack it
  to not rely on the include directories passed to the compiler. Note: the above path is for Ubuntu 16.04/18.4.

### Each time you start a new shell and want to use the code
* setup environment variables via `source configure.sh`
* if you made a modification to the submodules and want to build, you can use the build script: `source build.sh`. (of course you can porbably just type make in the top directory of the submodule as well.

### Pushing back changes

If you made changes to a submodule, you need to check in that code and then check in the new commit hash of the submodule to this repo.

Say you made a change to larcv. (Same instructions basically for all submodules).

* First make sure you are not in a DEATCHED_HEAD state)

      git branch
        develop
	`* tufts_ub`
	
* If it says detached head, go back to head of larflow repo and run `source goto_head_of_submodules.sh` and come back
* stage your commits and then push

      git add [[some file you edited]]
      git commit -m "[[short description of change]]"
      git push
* go back to head of larflow and commit the updated submodule (in this example `larcv`) to this repo
      cd ..
      git add larcv
      git commit -m "[[which submodule you updated]]"
      git push

## On the Tufts Grid

For working on the Tufts grid, you can run the code in two ways.

* Build the entire stack in the container and run (for production)
* Build the repository in your local directory (for development)


### Production

Not yet finished production container.

### Development

First, you'll need to get a copy and get it compiled

* make a directory for the larflow repository in your location on the Tufts Grid
* clone the repository: `git clone https://github.com/nutufts/larflow`
* edit `submit_compilation_to_grid.sh` to point to where your copy of the repository is located.
  also point to where your container is. (To avoid wasting space, try using an existing container if you can.)
* submit the batch script: `sbatch submit_compilation_to_grid.sh`

Then to run,

* (to come)
