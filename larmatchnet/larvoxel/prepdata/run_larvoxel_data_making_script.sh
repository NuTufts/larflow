#!/bin/bash

PYSCRIPT=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel/prepdata/make_larvoxel_classdata_from_tripletfile.py
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl
INPUT_JSON_FILE=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel/prepdata/filelist.json
OUTFOLDER=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel/prepdata/larvoxel_singleparticle_bnbnue/
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel/prepdata/workdir/

tag=bnbnue
jobid=${SLURM_ARRAY_TASK_ID}

mkdir -p $WORKDIR
jobworkdir=`printf "%s/larvoxelclass_${tag}_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir

mkdir -p $OUTFOLDER
mkdir -p $OUTFOLDER/electron
mkdir -p $OUTFOLDER/gamma
mkdir -p $OUTFOLDER/muon
mkdir -p $OUTFOLDER/pion
mkdir -p $OUTFOLDER/proton
mkdir -p $OUTFOLDER/kaon

local_jobdir=`printf /tmp/larvoxelclass_dataprep_${tag}_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

cd $local_jobdir
touch log_${tag}_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_${tag}_jobid${jobid}.txt`

cd $UBDL_DIR
source setenv_py3.sh >> ${local_logfile} 2>&1
source configure.sh >>	${local_logfile} 2>&1


cd $local_jobdir

outputname=`printf larvoxel_singleparticle_%04d.root ${jobid}`

CMD="python3 ${PYSCRIPT} --output ${outputname} --fileid ${jobid} ${INPUT_JSON_FILE}"
echo "SCRIPT: ${PYSCRIPT}" >> ${local_logfile} 2>&1
$CMD >> ${local_logfile} 2>&1
cp ${local_logfile} ${jobworkdir}/

# output data
cp larvoxel_singleparticle_*_electron.root ${OUTFOLDER}/electron/
cp larvoxel_singleparticle_*_gamma.root ${OUTFOLDER}/gamma/
cp larvoxel_singleparticle_*_muon.root ${OUTFOLDER}/muon/
cp larvoxel_singleparticle_*_pion.root ${OUTFOLDER}/pion/
cp larvoxel_singleparticle_*_proton.root ${OUTFOLDER}/proton/
cp larvoxel_singleparticle_*_kaon.root ${OUTFOLDER}/kaon/

cd /tmp
rm -r ${local_jobdir}

