#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep/workdir
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnb_nu_corsika_training.txt
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/dataprep/outdir_mcc9_v13_bnb_nu_corsika_training/

#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

stride=5
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/bnb_nu_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/larmatch_kps_dataprep_bnb_nu_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

cd $local_jobdir
touch log_bnb_nu_jobid${jobid}.txt

cd $UBDL_DIR
source setenv_py3.sh >> log_bnb_nu_jobid${jobid}.txt 2>&1
source configure.sh >>	log_bnb_nu_jobid${jobid}.txt 2>&1
cd $local_jobdir

SCRIPT="python ${UBDL_DIR}/larflow/larflow/KeyPoints/test/run_prepalldata.py"
echo "SCRIPT: ${SCRIPT}" >>	log_bnb_nu_jobid${jobid}.txt 2>&1
echo "startline: ${startline}" >>	log_bnb_nu_jobid${jobid}.txt 2>&1

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
    $COMMAND >> log_bnb_nu_jobid${jobid}.txt 2>&1
done

ls -lh out_*.root >>	log_bnb_nu_jobid${jobid}.txt 2>&1

ana_outputfile=`printf "larmatchtriplet_ana_trainingdata_%04d.root" ${jobid}`
hadd -f $ana_outputfile out_*.root >>	log_bnb_nu_jobid${jobid}.txt 2>&1
cp $ana_outputfile ${OUTPUT_DIR}/
cp log_bnb_nu_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
