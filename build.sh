#!/bin/bash

workdir=$PWD

# BUILD LARLITE
cd $workdir
cd larlite
make
cd UserDev/BasicTool
make
cd ../SelectionTool/OpT0Finder
make


cd ../Geo2D
make clean
make

cd ../LArOpenCV
make clean
make -j4

cd ../larcv
make clean
make -j4

cd ../larlitecv
make -j4

cd $workdir

# cilantro/pangolin
#mkdir Pangolin/build
#cd Pangolin/build
#cmake ..
#cmake --build .

#cd ../..
#mkdir cilantro/build
#cd cilantro/build
#cmake ..
#cmake --build .

cd $workdir/postprocessor
make

cd $workdir



