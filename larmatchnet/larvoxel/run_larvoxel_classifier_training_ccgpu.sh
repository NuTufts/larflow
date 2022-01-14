#!/bin/bash

NGPUS=$1
CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel/config/larvoxel_classifier_ccgpu.yaml

cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/
source setenv_py3.sh
source configure.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/
source set_pythonpath.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel


rm -f /tmp/sharedfile
python3 train_dist_larvoxel_classify.py --config ${CONFIG} --gpus ${NGPUS} > /tmp/larvoxel_classify_training_out_ccgpu.log
cp /tmp/larvoxel_classify_training_out_ccgpu.log /cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel/
