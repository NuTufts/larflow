#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=larmatchdata
#SBATCH --output=larmatchdata_2.log
#SBATCH --mem-per-cpu=2000
#SBATCH --time=30:00
#SBATCH --array=200-246


#container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_10022019.simg
container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_091319.simg

DATA_PREP_DIR=/cluster/tufts/wongjiradlab/twongj01/dev/ubdl/larflow/larmatchnet/dataprep

module load singularity
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run.sh"

