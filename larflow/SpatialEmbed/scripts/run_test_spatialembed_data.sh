#!/bin/bash

WORKDIR=/cluster/home/jhwang11/ubdl/larflow/larflow/SpatialEmbed/workdir
UBDL_DIR=/cluster/home/jhwang11/ubdl
INPUTLIST=/cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/goodlist.txt
OUTPUT_DIR=${UBDL_DIR}/larflow/larflow/SpatialEmbed/output/prepped_shower_data

stride=15
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

cd $UBDL_DIR
source setenv.sh
source configure.sh

echo "startline: ${startline}"

for i in {1..15}
do
    let lineno=$startline+$i
    larcvtruth=`sed -n ${lineno}p $INPUTLIST`

    # COMMAND="${SCRIPT} -o out_${i}.root -ilcv $larcvtruth -ill ${mcinfo_path}"
    # echo $COMMAND
    echo "++ $larcvtruth ++"
done
