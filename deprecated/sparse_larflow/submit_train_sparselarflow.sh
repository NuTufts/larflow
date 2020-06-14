#!/bin/bash

#SBATCH --job-name=train_sparselarflow
#SBATCH --output=log_train_sparselarflow.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=8000
#SBATCH --partition=gpu
#SBATCH --nodelist=pgpu01

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_051919.img
WORKDIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow

module load singularity
singularity exec  --nv ${CONTAINER} bash -c "cd ${WORKDIR} && source setup_env_tufts_container.sh && python train_sparse_larflow.py >& out.train.log"
