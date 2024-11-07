#!/bin/bash

slurmtag="slurm${SLURM_JOBID}"

NGPUS=$1
#CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larmatch/config/config_larmatchme_p100.yaml
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/
CONFIG=${UBDL_DIR}/larflow/larmatchnet/larmatch/config/config_larmatchme.yaml

cd $UBDL_DIR
source setenv_py3.sh
source configure.sh
cd ${UBDL_DIR}/larflow/larmatchnet/
source set_pythonpath.sh
cd ${UBDL_DIR}/larflow/larmatchnet/larmatch

export CUDA_LAUNCH_BLOCKING=1
rm -f /tmp/sharedfile
echo "CONFIG: ${CONFIG}"
#python3 train_dist_larmatchme.py --config ${CONFIG} --gpus 1 --no-parallel > /tmp/larmatch_training_out.${slurmtag}.noparallel.log
python3 train_dist_larmatchme.py --config ${CONFIG} -n 1 --gpus $NGPUS > /tmp/larmatch_training_out.${slurmtag}.log
cp /tmp/larmatch_training_out.${slurmtag}.log ${UBDL_DIR}/larflow/larmatchnet/larmatch/
