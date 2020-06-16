#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/larmatchnet/dataprep/workdir
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl
INPUTLIST=${UBDL_DIR}/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika.txt
OUTPUT_DIR=${UBDL_DIR}/larflow/larmatchnet/dataprep/outdir/
CONFIG=${UBDL_DIR}/larflow/larmatchnet/dataprep/prepmatchtriplet.cfg

#FOR DEBUG
#SLURM_ARRAY_TASK_ID=5

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

mkdir -p $WORKDIR
jobworkdir=`printf "%s/jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

larcv_outputfile=`printf "${OUTPUT_DIR}/larmatchtriplet_larcv_trainingdata_%04d.root" ${jobid}`
ana_outputfile=`printf "${OUTPUT_DIR}/larmatchtriplet_ana_trainingdata_%04d.root" ${jobid}`

cd $UBDL_DIR
source setenv.sh
source configure.sh

cd $jobworkdir

COMMAND="python $UBDL_DIR/larflow/larflow/PrepFlowMatchData/test/process_matchtriplet_data.py --out out.root -mc -c $CONFIG $infiles"
echo $COMMAND
$COMMAND >& log_jobid${jobid}.txt
cp out.root $ana_outputfile
rm out.root