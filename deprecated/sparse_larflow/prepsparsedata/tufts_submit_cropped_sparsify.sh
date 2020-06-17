#!/bin/bash

# slurm submission script to build ubdl

#SBATCH --job-name=larflow_cropped_sparsify
#SBATCH --output=larflow_cropped_sparsify.log
#SBATCH --mem-per-cpu=2000
#SBATCH --time=10:00
#SBATCH --array=0-999

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_051919.img

# get dir where we called script
workdir=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/prepsparsedata/

module load singularity
singularity exec ${container} bash -c "cd ${workdir} && source run_crop_job.sh"
