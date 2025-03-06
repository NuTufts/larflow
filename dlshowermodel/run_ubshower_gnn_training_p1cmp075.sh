#!/bin/bash

UBDL_DIR=$1
DLSHOWER_DIR=${UBDL_DIR}/larflow/dlshowermodel/

cd $UBDL_DIR
source setenv_noroot.sh
cd ${DLSHOWER_DIR}
source setenv.sh

workdir=${DLSHOWER_DIR}/workdir/run_${SLURM_JOB_ID}x/
mkdir -p $workdir

cd $workdir

python3 ${DLSHOWER_DIR}/main.py
