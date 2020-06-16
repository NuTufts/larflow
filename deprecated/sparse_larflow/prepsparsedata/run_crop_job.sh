#!/bin/bash

#echo "RUN SPARSE CROPPED LARFLOW"

# we assume this is being called from within a container
prepdir=$PWD

# configuration
sparsedir=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/
sparselist=${sparsedir}/prepsparsedata/mcc9mar_bnbcorsika.list
rerunlist=${sparsedir}/prepsparsedata/rerun_processing.list

# setup root
#echo "SETUP ROOT"
source /usr/local/root/build/bin/thisroot.sh

# set up environment: ubdl
#echo "SETUP UBDL"
cd /cluster/tufts/wongjiradlab/twongj01/ubdl
source setenv.sh
source configure.sh
cd $prepdir

# set up environment: larflow
#echo "SETUP LARFLOW"
source /cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/configure.sh

#echo "LARCV_BASEDIR: "$LARCV_BASEDIR
#echo "SLURM_ARRAY_TASK_ID: "${SLURM_ARRAY_TASK_ID}

# get the jobid and input file
# get the filepath
let lineno=${SLURM_ARRAY_TASK_ID}+1
jobid=`sed -n ${lineno}p ${rerunlist} | awk '{ print $1 }'`
larcvpath=`sed -n ${lineno}p ${rerunlist} | awk '{print $2 }'`
#echo "JOBID: ${jobid}"
#echo "LARCV INPUT: ${larcvpath}"

# extract the filename
larcvfilename=$(basename -- $larcvpath)
#echo $larcvfilename

# create the output file name
sparsefilename=`echo ${larcvfilename} | sed 's/larcvtruth/sparsecroplarflow/'`
#echo $sparsefilename

# prep the work directory
workdir=`printf ${sparsedir}/workdir/cropped_sparsifyjobid%04d ${jobid}`
#echo $workdir
mkdir -p $workdir
cp ${sparsedir}/sparsify_cropped_data.py ${workdir}/

cd ${workdir}
echo "python sparsify_cropped_data.py ${larcvpath} ${sparsefilename}"
#python sparsify_cropped_data.py ${larcvpath} ${sparsefilename} >& out.log
python sparsify_cropped_data.py ${larcvpath} ${sparsefilename} >& /dev/null
