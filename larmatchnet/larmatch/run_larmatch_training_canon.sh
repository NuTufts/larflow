#!/bin/bash

slurmtag="slurm${SLURM_JOBID}"

NGPUS=$1
UBDL_DIR=/n/home01/twongjirad/larmatch_retrain/ubdl/
CONFIG=${UBDL_DIR}/larflow/larmatchnet/larmatch/config/config_larmatchme_canon.yaml

cd $UBDL_DIR
source setenv_canon.sh
source configure.sh
cd ${UBDL_DIR}/larflow/larmatchnet/
source set_pythonpath.sh
cd ${UBDL_DIR}/larflow/larmatchnet/larmatch

rm -f /tmp/sharedfile
echo "CONFIG: ${CONFIG}"
#python3 train_dist_larmatchme.py --config ${CONFIG} -n 1 --gpus 1 --no-parallel > /tmp/larmatch_training_out.${slurmtag}.log
python3 train_dist_larmatchme.py --config ${CONFIG} -n 1 --gpus $NGPUS > /tmp/larmatch_training_out.${slurmtag}.log
cp /tmp/larmatch_training_out.${slurmtag}.log ${UBDL_DIR}/larflow/larmatchnet/larmatch/
