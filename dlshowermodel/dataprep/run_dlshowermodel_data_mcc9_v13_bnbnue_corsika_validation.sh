#!/bin/bash

tag=bnbnue
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl
DLSHOWERMODEL_DIR=${UBDL_DIR}/larflow/dlshowermodel/
WORKDIR=${DLSHOWERMODEL_DIR}/dataprep/workdir/
PYSCRIPT=${DLSHOWERMODEL_DIR}/dataprep/deploy_larmatchme.py
LARMATCH_CONFIG=${DLSHOWERMODEL_DIR}/dataprep/config_larmatchme.yaml
LARMATCH_WEIGHTS=${DLSHOWERMODEL_DIR}/dataprep/checkpoint.wise-capybara-107.101000.tar

# TRAINING DATA
OUTPUT_DIR=${DLSHOWERMODEL_DIR}/dataprep/output/outdir_mcc9_v13_bnbnue_corsika_validation/
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika_validation.paired.list

#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

stride=5
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/dlshowermodel_dataprep_${tag}_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/dlshowermodel_dataprep_${tag}_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir

cd $local_jobdir
touch log_${tag}_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_${tag}_jobid${jobid}.txt`

cd $UBDL_DIR
source setenv_py3.sh >> ${local_logfile} 2>&1
source configure.sh >>	${local_logfile} 2>&1
cd ${DLSHOWERMODEL_DIR}
source setenv.sh
cd $local_jobdir

CMD="python3 ${PYSCRIPT}"
echo "SCRIPT: ${PYSCRIPT}" >> ${local_logfile} 2>&1
echo "startline: ${startline}" >> ${local_logfile} 2>&1

for i in {1..5}
do
    let lineno=$startline+$i
    larcv_input=`sed -n ${lineno}p $INPUTLIST | awk '{ print $1 }'`
    larlite_input=`sed -n ${lineno}p $INPUTLIST | awk '{ print $2 }'`    
    larcv_input_base=`basename ${larcv_input}`
    larcv_input_dir=`dirname ${larcv_input}`
    output_base=`echo ${larcv_input_base} | sed 's|larcvtruth|dlshowermodel\_trainingdata|' | sed 's|root|h5|'`
    echo "Make ${output_base}"
    COMMAND="python3 ${PYSCRIPT} --config-file ${LARMATCH_CONFIG} --weights ${LARMATCH_WEIGHTS} --input-larcv ${larcv_input} --input-larlite ${larlite_input}  -trueedges --adc wiremc -o ./${output_base} -d cpu -p 0.3"
    echo $COMMAND
    $COMMAND >> ${local_logfile} 2>&1
    cp ${output_base}* ${OUTPUT_DIR}/
    rm ${output_base}*
    #break
done

cp log_${tag}_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
