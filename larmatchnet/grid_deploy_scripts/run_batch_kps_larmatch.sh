#!/bin/bash

OFFSET=$1
STRIDE=$2
SAMPLE_NAME=$3

# we assume we are already in the container

UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/grid_deploy_scripts/inputlists/${SAMPLE_NAME}.list
LARMATCH_DIR=${UBDL_DIR}/larflow/larmatchnet/
GRID_SCRIPTS_DIR=${UBDL_DIR}/larflow/larmatchnet/grid_deploy_scripts/
OUTPUT_DIR=${GRID_SCRIPTS_DIR}/outputdir/${SAMPLE_NAME}
OUTPUT_LOGDIR=${GRID_SCRIPTS_DIR}/logdir/${SAMPLE_NAME}

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_LOGDIR

# WE WANT TO RUN MULTIPLE FILES PER JOB IN ORDER TO BE GRID EFFICIENT
start_jobid=$(( ${OFFSET} + ${SLURM_ARRAY_TASK_ID}*${STRIDE}  ))

# LOCAL JOBDIR
local_jobdir=`printf /tmp/larmatch_kps_jobid%04d_${SAMPLE_NAME} ${SLURM_ARRAY_TASK_ID}`
echo "local jobdir: $local_jobdir"
rm -rf $local_jobdir
mkdir -p $local_jobdir

# local log file
local_logfile=`printf larmatch_kps_${SAMPLE_NAME}_jobid%04d.log ${SLURM_ARRAY_TASK_ID}`
echo "output logfile: "$local_logfile

echo "SETUP CONTAINER/ENVIRONMENT"
cd ${UBDL_DIR}
source setenv.sh
source configure.sh
export PYTHONPATH=${LARMATCH_DIR}:${PYTHONPATH}

echo "GO TO JOBDIR"
cd $local_jobdir

echo "STARTING TASK ARRAY ${SLURM_ARRAY_TASK_ID} for ${SAMPLE_NAME}" > ${local_logfile}

# run a loop
for ((i=0;i<${STRIDE};i++)); do

    jobid=$(( ${start_jobid} + ${i} ))
    echo "JOBID ${jobid}"
  
    # GET INPUT FILENAME
    let lineno=${jobid}+1
    inputfile=`sed -n ${lineno}p ${INPUTLIST}`
    baseinput=$(basename $inputfile )
    echo "inputfile path: $inputfile"

    # local outfile
    jobname=`printf jobid%04d ${jobid}`
    local_outfile=$(echo $baseinput | sed 's|merged_dlreco|larmatch_kps|g' | sed 's|.root||g' | xargs -I{} echo {}"-${jobname}.root")
    local_basename=$(echo $baseinput | sed 's|merged_dlreco|larmatch_kps|g' | sed 's|.root||g' | xargs -I{} echo {}"-${jobname}")
    echo "outfile : "$local_outfile
    
    CMD="python $LARMATCH_DIR/deploy_kps_larmatch.py --supera $inputfile --weights ${GRID_SCRIPTS_DIR}/larmatch_kps_weights/checkpoint.92000th.tar --output $local_outfile --min-score 0.2 --adc-name wire --chstatus-name wire --device-name cpu --use-unet -tb"
    echo $CMD
    $CMD >> ${local_logfile} 2>&1

    # subfolder dir
    let nsubdir=${jobid}/100
    subdir=`printf %03d ${nsubdir}`

    # copy to subdir in order to keep number of files per folder less than 100. better for file system.
    echo "COPY output to "${OUTPUT_DIR}/${subdir}/
    mkdir -p $OUTPUT_DIR/${subdir}/
    cp ${local_basename}* $OUTPUT_DIR/${subdir}/
done

# copy log to logdir
cp $local_logfile $OUTPUT_LOGDIR/

# clean-up
cd /tmp
rm -r $local_jobdir
