#!/bin/bash

WORKDIR=/cluster/home/jhwang11/ubdl/larflow/larflow/SpatialEmbed/workdir
UBDL_DIR=/cluster/home/jhwang11/ubdl
INPUTLIST=/cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/goodlist.txt
OUTPUT_DIR=${UBDL_DIR}/larflow/larflow/SpatialEmbed/output/prepped_shower_data

stride=10
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/spatialembed_dataprep_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

cd $UBDL_DIR
source setenv.sh
source configure.sh
cd $local_jobdir

SCRIPT="python ${UBDL_DIR}/larflow/larflow/SpatialEmbed/prepspatialembeddata.py"
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

    COMMAND="${SCRIPT} -o out_${i}.root -ilcv $larcvtruth -ill ${mcinfo_path}"
    echo $COMMAND
    $COMMAND >> log_jobid${jobid}.txt 2>&1
done

ls -lh out_*.root

rm $local_jobdir/delete*
cp *.root ${OUTPUT_DIR}
cp log_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir