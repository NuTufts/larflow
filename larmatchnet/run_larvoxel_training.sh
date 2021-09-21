#!/bin/bash

NGPUS=$1

cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/
source setenv_py3.sh
source configure.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/

rm -f /tmp/sharedfile
python3 train_dist_larmatchvoxel.py --gpus ${NGPUS} > /tmp/larvoxel_training_out.log
cp /tmp/larvoxel_training_out.log /cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/
