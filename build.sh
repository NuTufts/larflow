#!/bin/bash

cd larlite
make
cd UserDev/BasicTool
make
cd ../..

cd ../Geo2D
make

cd ../LArOpenCV
make -j4

cd ../larcv
make

cd ../larlitecv
make

# cilantro/pangolin
mkdir Pangolin/build
cd Pangolin/build
cmake ..
cmake --build .

cd ../..
mkdir cilantro/build
cd cilantro/build
cmake ..
cmake --build .

# return home
cd ..

