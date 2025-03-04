#!/bin/bash

UBDL_DIR=$1

cd $UBDL_DIR
source setenv_noroot.sh
cd ${UBDL_DIR}/larflow/dlshowermodel/
source setenv.sh


python3 main.py
