#!/bin/bash

tag=bnb_nu
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/workdir/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl
PYSCRIPT=${UBDL_DIR}/larflow/larmatchnet/larmatch/run_lardata2hdf5.py

# VALIDATION DATA
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnb_nu_corsika_validation/
#INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnb_nu_corsika_validation.paired.list
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/larmatch/prep/makeuplist.mcc9_v13_bnb_nu_corsika_validation.paired.txt

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
cd $UBDL_DIR/larflow/larmatchnet/
source set_pythonpath.sh
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
    output_base=`echo ${larcv_input_base} | sed 's|larcv\_mctruth|larmatch\_trainingdata|' | sed 's|root|h5|'`

    COMMAND="python3 ${PYSCRIPT} --input-larlite ${larlite_input} --input-larcv ${larcv_input} -tb -tri --adc wiremc -o ./${output_base}"
    echo $COMMAND
    echo $COMMAND >> ${local_logfile} 2>&1
    #$COMMAND >> ${local_logfile} 2>&1
    $COMMAND >> ${local_logfile}
    #$COMMAND
    cp ${output_base}* ${OUTPUT_DIR}/
    rm ${output_base}*
    #break
done

cp log_${tag}_jobid* ${jobworkdir}/

cd /tmp
rm -r $local_jobdir
