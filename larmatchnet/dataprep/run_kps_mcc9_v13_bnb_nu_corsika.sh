#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/larmatchnet/dataprep/workdir
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnb_nu_corsika_training.list
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/dataprep/outdir_mcc9_v13_bnb_nu_corsika_training/

#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

stride=10
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/larmatch_kps_dataprep_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

cd $UBDL_DIR
source setenv.sh
source configure.sh
cd $local_jobdir

SCRIPT="python ${UBDL_DIR}/larflow/larflow/KeyPoints/test/run_prepkeypointdata.py"
echo "SCRIPT: ${SCRIPT}"
echo "startline: ${startline}"
touch log_jobid${jobid}.txt

for i in {1..10}
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
    $COMMAND >> log_jobid${jobid}.txt 2>&1
done

ls -lh out_*.root

ana_outputfile=`printf "larmatchtriplet_ana_trainingdata_%04d.root" ${jobid}`
hadd -f $ana_outputfile out_*.root
cp $ana_outputfile ${OUTPUT_DIR}/
cp log_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir