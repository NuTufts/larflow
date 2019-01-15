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


cd $workdir/Geo2D
make

cd $workdir/LArOpenCV
make -j4

cd $workdir/larcv
make -j4

cd $workdir/larlitecv
make -j4

cd $workdir/postprocessor
make

cd $workdir



