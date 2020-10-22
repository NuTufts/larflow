#!/bin/bash

# Slurm submission script for training Spatial Embed Instance Segmentation Network

#SBATCH --job-name=spistrain
#SBATCH --output=spistrain.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1-00:00:00

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_pytorch1.3cpu.simg
DATA_PREP_DIR=/cluster/home/jhwang11/ubdl/larflow/larflow/SpatialEmbed/scripts

module load singularity
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source train_spatialembed.sh"