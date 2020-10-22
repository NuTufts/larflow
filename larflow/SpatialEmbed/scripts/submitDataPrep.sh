#!/bin/bash

# Slurm submission script for making Spatial Embed Instance Segmentation Data

#SBATCH --job-name=spisdata
#SBATCH --output=spisdata.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1-00:00:00
#SBATCH --array=0-48

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_pytorch1.3cpu.simg
DATA_PREP_DIR=/cluster/home/jhwang11/ubdl/larflow/larflow/SpatialEmbed/scripts

module load singularity
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_prep_spatialembed_data.sh"