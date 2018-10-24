#!/bin/bash

# workdir
workdir=/cluster/kappa/wongjiradlab/twongj01/larflow/

cd $workdir

# get right branch
#git checkout flashmatch_refactor

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
source build.sh
