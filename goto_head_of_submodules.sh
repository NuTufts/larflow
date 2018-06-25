#!/bin/bash

# Assumes we are in larflow repo
# goes to the dev branches

cd larlite
git checkout trunk

cd ../Geo2D
git checkout develop

cd ../LArOpenCV
git checkout fmwk_update

cd ../larcv
git checkout tufts_ub

cd ../larlitecv
git checkout dev_larcv2

cd ..
