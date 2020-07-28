#!/bin/bash

# SETUP ROOT
source /usr/local/root/root-6.16.00/bin/thisroot.sh

export OPENCV_LIBDIR=/usr/local/lib
export OPENCV_INCDIR=/usr/local/include

# uncomment this if you want to include headers using #include "larcv/PackageName/Header.h"
#export LARCV_PREFIX_INCDIR=1

suffix=so
if [[ `uname` == 'Darwin' ]]; then
    suffix=dylib
    # setting opencv paths to use location used by homebrew.
    # change this here, if you used another location
    export OPENCV_LIBDIR=/usr/local/opt/opencv3/lib
    export OPENCV_INCDIR=/usr/local/opt/opencv3/include    
fi

if [ ! -f $OPENCV_LIBDIR/libopencv_core.${suffix} ]; then
    echo "OPENCV LIBRARY NOT FOUND. Set env variable OPENCV_LIBDIR to where libopencv_core.${suffix} lives"
    return 1
fi

if [ ! -f $OPENCV_INCDIR/opencv/cv.h ]; then
    echo "OPENCV INCLUDES NOT FOUND. Please set env variable OPENCV_INCDIR to where opencv/cv.h lives"
    return 1
fi


