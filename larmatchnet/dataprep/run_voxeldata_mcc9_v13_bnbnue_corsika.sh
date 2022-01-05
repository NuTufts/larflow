#!/bin/bash

dataset=train
tag=bnbnue_${dataset}data_3mm_RSE
WORKDIR=/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/larmatchnet/dataprep/workdir/voxeldata/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/pabrat01/ubdl
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika.triplettruth.list
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/dataprep/voxel_outdir_mcc9_v13_bnbnue_corsika_${dataset}/${tag}


#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

stride=5
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/larvoxel_${tag}_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/larvoxel_dataprep_${tag}_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

cd $local_jobdir
touch log_${tag}_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_${tag}_jobid${jobid}.txt`

cd $UBDL_DIR
source setenv_py3.sh >> ${local_logfile} 2>&1
source configure.sh >>	${local_logfile} 2>&1
cd $local_jobdir

SCRIPT="python3 ${UBDL_DIR}/larflow/larmatchnet/dataprep/make_voxeltraindata.py"
echo "SCRIPT: ${SCRIPT}" >> ${local_logfile} 2>&1
echo "startline: ${startline}" >> ${local_logfile} 2>&1

for i in {1..5}
do
    let lineno=$startline+$i
    larmatchdata=`sed -n ${lineno}p $INPUTLIST`
    larmatchdata_base=`basename ${larmatchdata}`
    larmatchdata_dir=`dirname ${larmatchdata}`
    voxel_filesuffix=`echo ${larmatchdata_base} | sed 's|larmatchtriplet\_||g'`
    flist=`printf inputlist_${larmatch_data}_%01d.txt $i`
    echo $larmatchdata > $flist

    COMMAND="${SCRIPT} --output larvoxeldata_${tag}_${voxel_filesuffix} ${flist}"
    echo $COMMAND
    $COMMAND >> ${local_logfile} 2>&1
    cp larvoxeldata_${tag}_${voxel_filesuffix}* ${OUTPUT_DIR}/
    rm larvoxeldata_${tag}_${voxel_filesuffix}*
done

#ana_outputfile=`printf "larvoxeldata_${tag}_%04d.root" ${jobid}`
#hadd -f $ana_outputfile larvoxeldata_*.root >>	${local_logfile} 2>&1
#cp $ana_outputfile ${OUTPUT_DIR}/
cp log_${tag}_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
