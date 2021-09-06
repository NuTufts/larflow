#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep/workdir
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl
#INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika_training.txt
#OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/dataprep/outdir_mcc9_v13_bnbnue_corsika_training/
TAG=bnbnue_valid
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika_valid.txt
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/dataprep/outdir_mcc9_v13_bnbnue_corsika_valid/

#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

stride=5
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/${TAG}_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/larmatch_kps_dataprep_${TAG}_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir
cd $local_jobdir
touch log_${TAG}_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_${TAG}_jobid${jobid}.txt`

cd $UBDL_DIR
source setenv_py3.sh >> ${local_logfile} 2>&1
source configure.sh >>	${local_logfile} 2>&1
cd $local_jobdir

SCRIPT="python ${UBDL_DIR}/larflow/larflow/KeyPoints/test/run_prepalldata.py"
echo "SCRIPT: ${SCRIPT}" >> ${local_logfile} 2>&1
echo "startline: ${startline}" >> ${local_logfile} 2>&1

for i in {1..5}
do
    let lineno=$startline+$i
    larcvtruth=`sed -n ${lineno}p $INPUTLIST`
    larcvtruth_base=`basename ${larcvtruth}`
    larcvtruth_dir=`dirname $larcvtruth`

    mcinfo_path=`echo $larcvtruth | sed 's|larcvtruth-|mcinfo-|g'`
    mcinfo_base=`basename $mcinfo_path`
    mcinfo_dir=`dirname $mcinfo_path`
    ls -lh $mcinfo_path >> ${local_logfile} 2>&1
    COMMAND="${SCRIPT} --out out_${i}.root --input-larcv $larcvtruth --input-larlite ${mcinfo_path} -adc wiremc -tb -tri"
    echo $COMMAND
    $COMMAND >> ${local_logfile} 2>&1
done

ls -lh out_*.root >> ${local_logfile} 2>&1

ana_outputfile=`printf "larmatchtriplet_ana_trainingdata_%04d.root" ${jobid}`
hadd -f $ana_outputfile out_*.root >> ${local_logfile} 2>&1
cp $ana_outputfile ${OUTPUT_DIR}/
cp log_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
