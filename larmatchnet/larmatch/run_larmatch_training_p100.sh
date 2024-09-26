#!/bin/bash

slurmtag="slurm${SLURM_JOBID}"

NGPUS=$1
#CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larmatch/config/config_larmatchme_p100.yaml
UBDL_DIR=/cluster/tufts/wongjiradlabnu/ndahle01/ubdl/
CONFIG=${UBDL_DIR}/larflow/larmatchnet/larmatch/config/config_larmatchme.yaml

cd $UBDL_DIR
source setenv_py3.sh
source configure.sh
cd ${UBDL_DIR}/larflow/larmatchnet/
source set_pythonpath.sh
cd ${UBDL_DIR}/larflow/larmatchnet/larmatch

rm -f /tmp/sharedfile
echo "CONFIG: ${CONFIG}"

python3 train_dist_larmatchme.py --config ${CONFIG} -n 1 --gpus 1 --no-parallel > /tmp/larmatch_training_out.log
cp /tmp/larmatch_training_out_fhsjs.log ${UBDL_DIR}/larflow/larmatchnet/larmatch/
