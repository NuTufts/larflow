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

cd ..

