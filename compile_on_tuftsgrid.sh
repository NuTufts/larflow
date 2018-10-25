#!/bin/bash

# workdir
workdir=$1
#workdir=/cluster/kappa/wongjiradlab/twongj01/larflow/

cd $workdir

# get right branch
git checkout cilantro_test

# setup submodules
git submodule init
git submodule update

# build Pangoline first, before ROOT
cd Pangolin/
mkdir -p build
cd build
cmake ../
make

# build spectra
cd $workdir/cilantro
mkdir -p build
cd build
cmake ../
make

cd $workdir

# setup root
source /usr/local/root/release/bin/thisroot.sh

source configure.sh
#source clean.sh
source build.sh

# build photolib tools
cd larlite/UserDev/SelectionTool/OpT0Finder/
make

# post-processor
cd $workdir
cd postprocessor
make clean
make

cd flashmatch
make

# return home
cd $workdir
