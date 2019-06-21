#!/bin/bash

export LARFLOW_BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export LARFLOW_BUILDDIR=${LARFLOW_BASEDIR}/build/
export LARFLOW_LIBDIR=${LARFLOW_BASEDIR}/build/lib
export PYTORCH_LARFLOW_BASEDIR=${LARFLOW_BASEDIR}/pytorch-larflow
export LARFLOW_MODELDIR=${LARFLOW_BASEDIR}/models
export LARFLOW_SPARSEDIR=${LARFLOW_BASEDIR}/sparse_larflow
export LARFLOW_UTILSDIR=${LARFLOW_BASEDIR}/utils

# OPENCV
export OPENCV_LIBDIR=/usr/local/lib
export OPENCV_INCDIR=/usr/local/include
export USE_OPENCV=1


#git submodule update

# setup post-processor
export LARFLOW_POST_DIR=${LARFLOW_BASEDIR}/postprocessor
export LARFLOW_POST_LIBDIR=${LARFLOW_BASEDIR}/postprocessor/lib
[[ ":$LD_LIBRARY_PATH:" != *":${LARFLOW_POST_LIBDIR}:"* ]] && LD_LIBRARY_PATH="${LARFLOW_POST_LIBDIR}:${LD_LIBRARY_PATH}"
[[ ":$PATH:" != *":${LARFLOW_POST_DIR}:"* ]] && PATH="${LARFLOW_POST_DIR}:${PATH}"
[[ ":$PATH:" != *":${LARFLOW_POST_DIR}/cluster:"* ]] && PATH="${LARFLOW_POST_DIR}/cluster:${PATH}"
[[ ":$LD_LIBRARY_PATH:" != *":${LARFLOW_LIBDIR}:"* ]] && LD_LIBRARY_PATH="${LARFLOW_LIBDIR}:${LD_LIBRARY_PATH}"


# add model folder to python path
[[ ":$PYTHONPATH:" != *":${LARFLOW_MODELDIR}:"* ]] && PYTHONPATH="${LARFLOW_MODELDIR}:${PYTHONPATH}"

# add sparse folder to python path
[[ ":$PYTHONPATH:" != *":${LARFLOW_SPARSEDIR}:"* ]] && PYTHONPATH="${LARFLOW_SPARSEDIR}:${PYTHONPATH}"

# add utils folder to python path
[[ ":$PYTHONPATH:" != *":${LARFLOW_UTILSDIR}:"* ]] && PYTHONPATH="${LARFLOW_UTILSDIR}:${PYTHONPATH}"

# add loader for c++ library to python path
[[ ":$PYTHONPATH:" != *":${LARFLOW_BASEDIR}/python:"* ]] && PYTHONPATH="${LARFLOW_BASEDIR}/python:${PYTHONPATH}"


# SETUP CILANTRO
export CILANTRO_INC_DIR=${LARFLOW_BASEDIR}/cilantro/include
export CILANTRO_LIB_DIR=${LARFLOW_BASEDIR}/cilantro/build
[[ ":$LD_LIBRARY_PATH:" != *":${CILANTRO_LIB_DIR}:"* ]] && LD_LIBRARY_PATH="${CILANTRO_LIB_DIR}:${LD_LIBRARY_PATH}"

# SETUP EIGEN
export EIGEN_INC_DIR=/usr/include/eigen3
export EIGEN_LIB_DIR=

# SETUP LARCVDATASET
cd larcvdataset
source setenv.sh
cd ../
