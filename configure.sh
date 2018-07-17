#!/bin/bash

export LARFLOW_BASEDIR=$PWD
export PYTORCH_LARFLOW_BASEDIR=$(LARFLOW_BASEDIR)/pytorch-larflow

# OPENCV
export OPENCV_LIBDIR=/usr/local/lib
export OPENCV_INCDIR=/usr/local/include
export USE_OPENCV=1

git submodule update

# setup larlite environment variables
cd larlite
source config/setup.sh

# setup geo2d tool library
cd ../Geo2D
source config/setup.sh

# setup LArOpenCV algorithms
cd ../LArOpenCV
source setup_laropencv.sh

# setup larlite environment variabls
cd ../larcv
source configure.sh

# setup larlitecv
cd ../larlitecv
source configure.sh

# add larcvdataset folder to pythonpath
cd ../larcvdataset
source setenv.sh

# setup post-processor
export LARFLOW_POST_LIBDIR=${LARFLOW_BASEDIR}/postprocessor/lib
[[ ":$LD_LIBRARY_PATH:" != *":${LARFLOW_POST_LIBDIR}:"* ]] && LD_LIBRARY_PATH="${LARFLOW_POST_LIBDIR}:${LD_LIBRARY_PATH}"

# return to top-level directory
cd ../
