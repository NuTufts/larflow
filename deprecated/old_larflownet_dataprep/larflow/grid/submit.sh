#!/bin/bash
#
#SBATCH --job-name=croplf_fullres
#SBATCH --output=log_croplf_fullres.txt
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --array=0-168

# GRID LOCATIONS
CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/dllee_unified/singularity-dllee-unified-taggerv2beta-20171121.img
# INSIDE CONTAINER
WORKDIR=/cluster/kappa/wongjiradlab/twongj01/llf/larflow/larflowcropper/grid/
OUTPUTDIR=/cluster/kappa/wongjiradlab/twongj01/llf/larflow/output_fullres/
JOBIDLIST=${WORKDIR}/rerunlist.txt
INPUTLISTDIR=${WORKDIR}/inputlists

module load singularity
srun singularity exec  ${CONTAINER} bash -c "cd ${WORKDIR} && source run_job.sh ${INPUTLISTDIR} ${JOBIDLIST} ${OUTPUTDIR}"

