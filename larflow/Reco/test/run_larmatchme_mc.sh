#!/bin/bash

JOBSTARTDATE=$(date)

INPUT_DLMERGED=$1
OUTPUT_LM=$2
NEVENTS=$3
INPUTSTEM=merged_dlreco

# we assume we are already in the container
# Common parameters
export OMP_NUM_THREADS=16

# Parameters for production version reco
#RECOVER=v2_me_06_03_prodtest
#UBDL_DIR=/cluster/home/ubdl/
#LARMATCH_DIR=${UBDL_DIR}/larflow/larmatchnet/larmatch/
#WEIGHTS_DIR=${LARMATCH_DIR}
#WEIGHT_FILE=larmatch_ckpt78k.pt
#CONFIG_FILE=/cluster/home/lantern_scripts/config_larmatchme_deploycpu.yaml
#LARMATCHME_SCRIPT=${LARMATCH_DIR}/deploy_larmatchme.py

# Parameters for shower-keypoint update version
RECOVER=v3dev_lm_showerkp_retraining
LARMATCH_DIR=${UBDL_BASEDIR}/larflow/larmatchnet/larmatch/

#WEIGHTS_DIR=${LARMATCH_DIR}/checkpoints/sparkling-sunset-78/
#WEIGHT_FILE=checkpoint.44000th.tar
WEIGHTS_DIR=${LARMATCH_DIR}/
WEIGHT_FILE=checkpoint.easy-wave-79.93000th.tar

CONFIG_FILE=${LARMATCH_DIR}/standard_deploy_larmatchme_cpu.yaml
LARMATCHME_SCRIPT=${LARMATCH_DIR}/deploy_larmatchme_v2.py

# More common parameters dependent on version-specific variables
RECO_TEST_DIR=${UBDL_DIR}/larflow/larflow/Reco/test/
baseinput=$(basename $INPUT_DLMERGED )
cp $INPUT_DLMERGED $baseinput
echo "inputfile path: $INPUT_DLMERGED"
echo "baseinput: $baseinput"

baselm=${OUTPUT_LM}
echo "larmatch outfile : ${baselm}"

# larmatch v1
#CMD="python3 $LARMATCH_DIR/deploy_larmatchme.py --config-file ${CONFIG_FILE} --supera $baseinput --weights ${WEIGHTS_DIR}/${WEIGHT_FILE} --output $lm_outfile --min-score 0.3 --adc-name wire --chstatus-name wire --device-name cpu -tb"
#CMD="python3 ${LARMATCHME_SCRIPT} --config-file ${CONFIG_FILE} --supera $baseinput --weights ${WEIGHTS_DIR}/${WEIGHT_FILE} --output ${lm_outfile} --min-score 0.5 --adc-name wire --device-name cpu -tb"

# larmatch v2 (shower keypoint version)
CMD="python3 ${LARMATCHME_SCRIPT} --config-file ${CONFIG_FILE} --input-larcv ${baseinput} --input-larlite ${baseinput} --weights ${WEIGHTS_DIR}/${WEIGHT_FILE} --output ${baselm} --min-score 0.3 --adc-name wire --device-name cpu --use-skip-limit --allow-output-overwrite -n ${NEVENTS}"
echo $CMD
$CMD
