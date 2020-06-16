#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=lmkpsdata
#SBATCH --output=lmkpsdata.log
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1-00:00:00
#SBATCH --array=2-245

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_10022019.simg
#container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_091319.simg

DATA_PREP_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/larmatchnet/dataprep

module load singularity
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_kps.sh"

