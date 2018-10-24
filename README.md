# LArFlow: prediction pixel correspondence between LArTPC wireplane images

This repository contains the code for developing the larflow network.

## Contents

(not including submodule dependencies)

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
* pytorch (0.4.1 known to work)
* numpy (1.14.03 known to work)
* tensorboardX (from [here](https://github.com/lanpa/tensorboard-pytorch))
* tensorboard
* cuda (8.0,9.0,9.1 known to work)
* Eigen 3
* GLEW (opengl library)
* (to do: add missing)

### Included as submodules

* LArCV2 (tufts_ub branch): library for representing LArTPC data as images along with meta-data. Also, provides IO.
* larlite: classes for meta-data. Also provides access to constants for the UB detector geometry and LAr physics
* Geo2D: a library of 2D geometry tools to help calculate things like intersections of objects
* LArOpenCV: a library of algorithms using the OpenCV library. built as libraries that are a part of larlite
* larlitecv: a library to open larcv and larlite files in a coordinated fashion
* larcvdataset: wrapper class providing interface to images stored in the larcv format. converts data into numpy arrays for use in pytorch
* Cilantro: a library with various Clustering routines w/ C++ API
* Pangolin: a OpenGL viewer package, used by Cilantro

## Setup

### First-time setup

* clone this repository: `git clone https://github.com/NuTufts/larflow larflow`
* setup the submodules, configure environment variables, and build: `source first_setup.sh`
* if you plan to modify any of the submodules, you will need go to the head branch for each submodule. use: `source goto_head_of_submodules.sh`

### Issues building Pangolin

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
