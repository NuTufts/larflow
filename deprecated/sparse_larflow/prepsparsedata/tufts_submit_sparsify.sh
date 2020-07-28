#!/bin/bash

# slurm submission script to build ubdl

#SBATCH --job-name=larflow_sparsify
#SBATCH --output=larflow_sparsify.log
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1:00:00
#SBATCH --array=0-244

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_singularity_031219.img

# get dir where we called script
workdir=/cluster/kappa/90-days-archive/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/prepsparsedata/

module load singularity
singularity exec ${container} bash -c "cd ${workdir} && source run_job.sh"

