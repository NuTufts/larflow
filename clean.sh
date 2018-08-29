#!/bin/bash

cd larlite
make clean
cd UserDev/BasicTool
make clean
cd ../..

cd ../Geo2D
make clean

cd ../LArOpenCV
make clean -j4

cd ../larcv
make clean

cd ../larlitecv
make clean

cd ..

