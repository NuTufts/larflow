#!/bin/bash

workdir=$PWD

cd larlite
make
cd UserDev/BasicTool
make
cd ../SelectionTool/OpT0Finder
make
cd ../../..

cd ../Geo2D
make

cd ../LArOpenCV
make -j4

cd ../larcv
make

cd ../larlitecv
make

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

# return home
#cd ../..

cd $workdir/postprocessor
make clean
make

cd $workdir



