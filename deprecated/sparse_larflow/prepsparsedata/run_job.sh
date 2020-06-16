#!/bin/bash

# we assume this is being called from within a container

# configuration
sparsedir=/cluster/kappa/90-days-archive/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/
sparselist=${sparsedir}/prepsparsedata/mcc9mar_bnbcorsika.list
rerunlist=${sparsedir}/prepsparsedata/rerun_processing.list

# set up environment
cd /cluster/kappa/90-days-archive/wongjiradlab/twongj01/ubdl
source setenv.sh
source configure.sh

# get the jobid and input file
# get the filepath
let lineno=${SLURM_ARRAY_TASK_ID}+1
jobid=`sed -n ${lineno}p ${rerunlist} | awk '{ print $1 }'`
larcvpath=`sed -n ${lineno}p ${rerunlist} | awk '{print $2 }'`
echo "JOBID: ${jobid}"
echo "LARCV INPUT: ${larcvpath}"

# extract the filename
larcvfilename=$(basename -- $larcvpath)
echo $larcvfilename

# create the output file name
sparsefilename=`echo ${larcvfilename} | sed 's/larcvtruth/sparselarflowy2u/'`
echo $sparsefilename

# prep the work directory
workdir=`printf ${sparsedir}/workdir/sparsifyjobid%04d ${jobid}`
echo $workdir
mkdir -p $workdir
cp ${sparsedir}/sparsify_data.py ${workdir}/


cd ${workdir}
echo "python sparsify_data.py ${larcvpath} ${sparsefilename}"
python sparsify_data.py ${larcvpath} ${sparsefilename}
