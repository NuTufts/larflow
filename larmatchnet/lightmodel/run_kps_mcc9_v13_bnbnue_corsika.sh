#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/larmatchnet/lightmodel/workdir
UBDL_DIR=/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/lightmodel/inputlists/mcc9_v13_bnbnue_corsika.paired.list
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/lightmodel/outdir_flashmatchdata_mcc9_v13_bnbnue_corsika/
TAG=bnbnue

#FOR DEBUG
#SLURM_ARRAY_TASK_ID=0

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

touch paths.txt
pathfile=`echo paths.txt`

for i in {1..5}
do
    let lineno=$startline+$i
    larcvtruth=`sed -n ${lineno}p $INPUTLIST | awk '{print $1}'`
    mcinfo_path=`sed -n ${lineno}p $INPUTLIST | awk '{print $2}'`
    echo "${mcinfo_path}" >> ${pathfile} 2>&1
    COMMAND="${SCRIPT} --out out_${i}.root --input-larcv $larcvtruth --input-larlite ${mcinfo_path} -adc wiremc -tb -tri"
    echo $COMMAND
    $COMMAND >> ${local_logfile} 2>&1
done

ls -lh out_*.root >> ${local_logfile} 2>&1

ana_outputfile=`printf "larmatchtriplet_${TAG}_%04d.root" ${jobid}`
hadd -f $ana_outputfile out_*.root >> ${local_logfile} 2>&1

SCRIPT="python3 ${UBDL_DIR}/larflow/larmatchnet/lightmodel/makeJson.py"
COMMAND="${SCRIPT} --t ${ana_outputfile}"
echo $COMMAND
$COMMAND >> ${local_logfile} 2>&1

out_json=`printf "tempJson.json"`

SCRIPT="python3 ${UBDL_DIR}/larflow/larmatchnet/lightmodel/make_flashmatchdata_from_tripletfile.py"
COMMAND="${SCRIPT} -o out_062023 -i 0 ${out_json}"
echo $COMMAND
$COMMAND >> ${local_logfile} 2>&1

cp *out_062023* ${OUTPUT_DIR}/
cp ${local_logfile} ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
