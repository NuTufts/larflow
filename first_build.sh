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

# BUILD POST PROCESSOR
cd $workdir/postprocessor
make

cd $workdir



