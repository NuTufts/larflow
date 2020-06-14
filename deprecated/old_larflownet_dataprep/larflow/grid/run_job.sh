#!/bin/sh

# REMEMBER, WE ARE IN THE CONTAINER RIGHT NOW
# This means we access the next work drives through some mounted folders

LARFLOW_DEP_DIR=/cluster/kappa/wongjiradlab/twongj01/llf/larflow
LARFLOW_DIR=/cluster/kappa/wongjiradlab/twongj01/llf/larflow/larflowcropper
export PATH=${LARFLOW_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${LARFLOW_DIR}/obj:${LD_LIBRARY_PATH}

workdir=${LARFLOW_DIR}/grid

inputlist_dir=$1
jobid_list=$2
output_dir=$3

echo "WORKDIR: $1"

# setup shell
source /usr/local/bin/thisroot.sh
cd $LARFLOW_DEP_DIR/larlite
source config/setup.sh
cd $LARFLOW_DEP_DIR/larcv
source configure.sh
cd $LARFLOW_DEP_DIR/larlitecv
source configure.sh

cd $workdir

let NUM_PROCS=`cat ${jobid_list} | wc -l`
echo "number of processes: $NUM_PROCS"
if [ "$NUM_PROCS" -lt "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "No Procces ID to run."
    return
fi

let "proc_line=${SLURM_ARRAY_TASK_ID}+1"
echo "sed -n ${proc_line}p ${jobid_list}"
let jobid=`sed -n ${proc_line}p ${jobid_list}`
echo "JOBID ${jobid}"

slurm_folder=`printf slurm_fullres_larflowcrop_job%04d ${jobid}`
mkdir -p ${slurm_folder}
cd ${slurm_folder}/

# copy over input list
inputlist=`printf ${inputlist_dir}/input_%d.txt ${jobid}`

cp $inputlist input.txt
larcv_larflow=`sed -n 1p input.txt`

logfile=`printf log_%04d.txt ${jobid}`
echo "crop_wlarcv2 $supera ${LARFLOW_DIR}/larflowcrop.cfg baka.root"
crop_wlarcv2 $larcv_larflow ${LARFLOW_DIR}/larflowcrop.cfg baka.root >& $logfile || exit

outfile=`printf ${output_dir}/larflowcrop_%06d.root ${jobid}`
cp output_larflow_cropper.root $outfile

# clean up
cd ../
rm -r $slurm_folder
