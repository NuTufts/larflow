#!/bin/bash

# this script is to run train_wlarcv2.py on a grid node
# this script runs from inside the singularity container

# path to workdir. place where we will write files
WORKDIR=$1

# path to larflow repo
LARFLOW_REPO=$2

# Path to where data files are stored
DATADIR=$3

# LArCVDataset config gile
DATALOADER_TRAIN_CFG=$4
DATALOADER_VALID_CFG=$5

# to to work directory
cd $WORKDIR

# setup container environment
source ${LARFLOW_REPO}/training/grid_scripts/setup_larcv2_container.sh ${LARFLOW_REPO}

# copy data to tmp (another job will do this)
#rsync -av --progress $DATADIR/*.root /tmp/

# create job folder
jobdir=`printf slurm_training_%d ${SLURMJOBID}`
mkdir $jobdir
cp ${LARFLOW_REPO}/training/*.py ${jobdir}/
cp ${DATALOADER_TRAIN_CFG} ${jobdir}/
cp ${DATALOADER_VALID_CFG} ${jobdir}/

# go into job dir
cd $jobdir

# make folder for tensorboard data
mkdir runs

# run the job
logfile=`printf log_jobid%d ${SLURMJOBID}`
python train_wlarcv2.py >& ${logfile}
