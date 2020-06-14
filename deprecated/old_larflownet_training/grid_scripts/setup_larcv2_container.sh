#!/bin/bash

home=$PWD
LARFLOW_DIR=$1

# setup CUDA
export PATH=/usr/local/nvidia:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia:${LD_LIBRARY_PATH}

# setup ROOT
source /usr/local/root/release/bin/thisroot.sh

cd $LARFLOW_DIR
source configure.sh

export ROOT_INCLUDE_PATH=${LARLITE_BASEDIR}/core:${LARLITE_BASEDIR}/larcv/core/DataFormat:${LARCV_BASEDIR}/larcv/core/Base
export ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}:${LARCV_BASEDIR}/larcv/core/PyUtil
export ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}:${LARCV_BASEDIR}/larcv/core/CPPUtil
export ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}:${LARCV_BASEDIR}/larcv/core/Processor
export ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}:${LARCV_BASEDIR}/larcv/core/ROOTUtil

cd $home
