#!/bin/bash

NGPUS=$1
CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/gen2/icdl/larflow/larmatchnet/larmatch/config/config_larmatchme_p100.yaml

cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/icdl/
source setenv_py3.sh
source configure.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/icdl/larflow/larmatchnet/
source set_pythonpath.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/icdl/larflow/larmatchnet/larmatch


rm -f /tmp/sharedfile
python3 train_dist_larmatchme.py --config ${CONFIG} --gpus ${NGPUS} > /tmp/larmatch_training_out.log
cp /tmp/larmatch_training_out.log /cluster/tufts/wongjiradlabnu/twongj01/gen2/icdl/larflow/larmatchnet/larmatch/
