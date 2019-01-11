#!/bin/bash

workdir=$PWD

git submodule init
git submodule update

# first build pangoline and cilantro before we load root
mkdir Pangolin/build
cd Pangolin/build
cmake ..
cmake --build .

cd $workdir
mkdir cilantro/build
cd cilantro/build
cmake ..
cmake --build .

# now setup ROOT
source /usr/local/root/release/bin/thisroot.sh

# configure environment
cd $workdir
source configure.sh

# BUILD LARLITE
cd $workdir/larlite
make
cd UserDev/BasicTool
make
cd ../SelectionTool/OpT0Finder
make

cd $workdir/Geo2D
make -j4

cd $workdir/LArOpenCV
make -j4

cd $workdir/larcv
make -j4

cd $workdir/larlitecv
make -j4

cd $workdir/postprocessor
make

cd $workdir



