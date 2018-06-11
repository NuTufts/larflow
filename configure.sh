#!/bin/bash

# setup larlite environment variables
cd larlite
source config/setup.sh

# setup larlite environment variabls
cd ../larcv
source configure.sh

# add larcvdataset folder to pythonpath
cd ../larcvdataset
source setenv.sh

# return to top-level directory
cd ../
