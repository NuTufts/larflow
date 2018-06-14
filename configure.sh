#!/bin/bash

# OPENCV
export OPENCV_LIBDIR=/usr/local/lib
export OPENCV_INCDIR=/usr/local/include
export USE_OPENCV=1

# setup larlite environment variables
cd larlite
source config/setup.sh

# setup larlite environment variabls
cd ../larcv
source configure.sh

# setup larlitecv
cd ../larlitecv
source configure.sh

# add larcvdataset folder to pythonpath
cd ../larcvdataset
source setenv.sh


# return to top-level directory
cd ../
