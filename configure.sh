#!/bin/bash

export LARFLOW_BASEDIR=$PWD
export PYTORCH_LARFLOW_BASEDIR=${LARFLOW_BASEDIR}/pytorch-larflow
export LARFLOW_MODELDIR=${LARFLOW_BASEDIR}/models
export LARFLOW_UTILSDIR=${LARFLOW_BASEDIR}/utils

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

# add larcvdataset folder to pythonpath
cd ../serverfeed
source setenv.sh

# return to top-level directory
cd ../

# setup post-processor
export LARFLOW_POST_LIBDIR=${LARFLOW_BASEDIR}/postprocessor/lib
[[ ":$LD_LIBRARY_PATH:" != *":${LARFLOW_POST_LIBDIR}:"* ]] && LD_LIBRARY_PATH="${LARFLOW_POST_LIBDIR}:${LD_LIBRARY_PATH}"


# add model folder to python path
[[ ":$PYTHONPATH:" != *":${LARFLOW_MODELDIR}:"* ]] && PYTHONPATH="${LARFLOW_MODELDIR}:${PYTHONPATH}"

# add utils folder to python path
[[ ":$PYTHONPATH:" != *":${LARFLOW_UTILSDIR}:"* ]] && PYTHONPATH="${LARFLOW_UTILSDIR}:${PYTHONPATH}"


# SETUP CILANTRO
export CILANTRO_INC_DIR=${LARFLOW_BASEDIR}/cilantro/include
export CILANTRO_LIB_DIR=${LARFLOW_BASEDIR}/cilantro/build
[[ ":$LD_LIBRARY_PATH:" != *":${CILANTRO_LIB_DIR}:"* ]] && LD_LIBRARY_PATH="${CILANTRO_LIB_DIR}:${LD_LIBRARY_PATH}"

# SETUP EIGEN
export EIGEN_INC_DIR=/usr/include/eigen3
export EIGEN_LIB_DIR=
