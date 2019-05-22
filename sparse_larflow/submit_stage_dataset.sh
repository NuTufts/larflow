#!/bin/bash

#SBATCH --job-name=stage_sparse_data
#SBATCH --output=log_stage_sparse_data
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --mem=2000
#SBATCH --partition=gpu
#SBATCH --nodelist=pgpu01

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_051919.img
DATADIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/prepsparsedata
VERSION=v5

module load singularity
singularity exec  --nv ${CONTAINER} bash -c "rsync -av --progress ${DATADIR}/*_${VERSION}.root /tmp/"
