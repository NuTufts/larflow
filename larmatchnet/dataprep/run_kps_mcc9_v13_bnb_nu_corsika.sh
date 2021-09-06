#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep/workdir
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl
#INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnb_nu_corsika_training.txt
#OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/dataprep/outdir_mcc9_v13_bnb_nu_corsika_training/
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnb_nu_corsika_valid.txt
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/dataprep/outdir_mcc9_v13_bnb_nu_corsika_valid/
tag=bnb_nu_validdata

#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

stride=5
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/${tag}_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/larmatch_kps_dataprep_${tag}_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

cd $local_jobdir
touch log_bnb_nu_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_${tag}_jobid${jobid}.txt`

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

    mcinfo_base=`python ${WORKDIR}/../find_mcinfo_pair.py ${larcvtruth}`
    mcinfo_dir=`echo $larcvtruth_dir | sed 's|larcv_mctruth|larlite_mcinfo|g'`
    mcinfo_path=${mcinfo_dir}/${mcinfo_base}
    ls -lh $mcinfo_path
    COMMAND="${SCRIPT} --out out_${i}.root --input-larcv $larcvtruth --input-larlite ${mcinfo_path} -adc wiremc -tb -tri"
    echo $COMMAND
    $COMMAND >> ${local_logfile} 2>&1
done

ls -lh out_*.root >>	${local_logfile} 2>&1

ana_outputfile=`printf "larmatchtriplet_${tag}_%04d.root" ${jobid}`
hadd -f $ana_outputfile out_*.root >>	${local_logfile} 2>&1
cp $ana_outputfile ${OUTPUT_DIR}/
cp log_bnb_nu_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
