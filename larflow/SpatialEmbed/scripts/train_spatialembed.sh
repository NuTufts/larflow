#!/bin/bash

SCRIPTDIR=/cluster/home/jhwang11/ubdl/larflow/larflow/SpatialEmbed
DATA_DIR=/cluster/home/jhwang11/ubdl/larflow/larflow/SpatialEmbed/output/prepped_shower_data
UBDL_DIR=/cluster/home/jhwang11/ubdl
NOW=$(date +"%m_%d") 
SAVE_DIR=/cluster/home/jhwang11/ubdl/larflow/larflow/trained_models/spatialembed_model_${NOW}.pt

cd $UBDL_DIR
source setenv.sh
source configure.sh
cd $SCRIPTDIR

SCRIPT="python ${UBDL_DIR}/larflow/larflow/SpatialEmbed/train_SpatialEmbed_net.py"
echo "SCRIPT: ${SCRIPT}"
touch train_log_jobid${jobid}.txt

COMMAND="${SCRIPT} -d ${DATA_DIR} -s ${SAVE_DIR}"
echo $COMMAND
$COMMAND >> train_log_jobid${jobid}.txt 2>&1

