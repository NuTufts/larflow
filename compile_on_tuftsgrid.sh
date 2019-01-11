#!/bin/bash

# workdir
workdir=$1
cd $workdir

# get right branch
#git checkout cilantro_test

# setup submodules
#git submodule init
#git submodule update

# build Pangoline first, before ROOT
#cd Pangolin/
#mkdir -p build
#cd build
#cmake ../
#make

# build spectra
#cd $workdir/cilantro
#mkdir -p build
#cd build
#cmake ../
#make

cd $workdir

# setup root
source /usr/local/root/release/bin/thisroot.sh

source configure.sh
#source clean.sh
source build.sh

# post-processor
cd $workdir/postprocessor
make

# return home
cd $workdir

echo "FIN"
