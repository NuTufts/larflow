#!/bin/bash

tag=bnb_nu
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/icdl/larflow/larmatchnet/larmatch/prep/workdir/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/icdl
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/larmatch/prep/inputlists/mcc9_v13_bnb_nu_corsika.triplettruth.list
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnb_nu_corsika/
PYSCRIPT=${UBDL_DIR}/larflow/larmatchnet/larmatch/prep/make_larmatch_training_data_from_tripletfile.py


#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

stride=5
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/larmatch_${tag}_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/larmatch_dataprep_${tag}_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

cd $local_jobdir
touch log_${tag}_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_${tag}_jobid${jobid}.txt`

cd $UBDL_DIR
source setenv_py3.sh >> ${local_logfile} 2>&1
source configure.sh >>	${local_logfile} 2>&1
cd $local_jobdir

CMD="python3 ${PYSCRIPT}"
echo "SCRIPT: ${PYSCRIPT}" >> ${local_logfile} 2>&1
echo "startline: ${startline}" >> ${local_logfile} 2>&1

for i in {1..5}
do
    let lineno=$startline+$i
    larmatchdata=`sed -n ${lineno}p $INPUTLIST`
    larmatchdata_base=`basename ${larmatchdata}`
    larmatchdata_dir=`dirname ${larmatchdata}`
    voxel_filesuffix=`echo ${larmatchdata_base} | sed 's|larmatchtriplet\_||g'`

    COMMAND="python3 ${PYSCRIPT} -d uboone --output larmatchdata_${tag}_${voxel_filesuffix} --single ${larmatchdata}"
    echo $COMMAND
    echo $COMMAND >> ${local_logfile} 2>&1
    #$COMMAND >> ${local_logfile} 2>&1
    $COMMAND >> ${local_logfile}
    cp larmatchdata_${tag}_${voxel_filesuffix}* ${OUTPUT_DIR}/
    rm larmatchdata_${tag}_${voxel_filesuffix}*
    #break
done

cp log_${tag}_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
