#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlab/twongj01/dev/ubdl/larflow/larmatchnet/dataprep/workdir
INPUTLIST=/cluster/tufts/wongjiradlab/twongj01/dev/ubdl/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika.txt
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/dev/ubdl
OUTPUT_DIR=/cluster/tufts/wongjiradlab/twongj01/dev/ubdl/larflow/larmatchnet/dataprep/outdir/
CONFIG=/cluster/tufts/wongjiradlab/twongj01/dev/ubdl/larflow/larflow/PrepFlowMatchData/test/prepflowmatchdata.cfg

#SLURM_ARRAY_TASK_ID=1

stride=10
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

echo "startline: ${startline}"
infiles=""
for i in {1..10}
do
    let lineno=$startline+$i
    infile=`sed -n ${lineno}p $INPUTLIST`
    infiles=`printf "%s %s" "$infiles" $infile`
done
echo "INPUTFILES:"
echo $infiles | sed 's| |\n|g'

jobworkdir=`printf "%s/jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

larcv_outputfile=`printf "${OUTPUT_DIR}/larmatch_larcv_trainingdata_%04d.root" ${jobid}`
ana_outputfile=`printf "${OUTPUT_DIR}/larmatch_ana_trainingdata_%04d.root" ${jobid}`

cd $UBDL_DIR
source setenv.sh
source configure.sh

cd $jobworkdir

echo "python $UBDL_DIR/larflow/larflow/PrepFlowMatchData/test/run_prepflowmatchdata.py -olcv out.root -c $CONFIG $infiles"
python $UBDL_DIR/larflow/larflow/PrepFlowMatchData/test/run_prepflowmatchdata.py -olcv out.root -c $CONFIG $infiles >& log_jobid${jobid}.txt
cp out.root $larcv_outputfile
cp ana_flowmatch_data.root $ana_outputfile

