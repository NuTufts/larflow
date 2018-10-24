#!/bin/bash
#
#SBATCH --job-name=compile_larflow
#SBATCH --output=log_compile_larflow.log
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --partition batch

LARFLOW_REPO_DIR=/cluster/tufts/wongjiradlab/twongj01/larflow/
LARFLOW_REPO_DIR_INCONTAINER=/cluster/kappa/wongjiradlab/twongj01/larflow/
CONTAINER=${LARFLOW_REPO_DIR}/container/singularity-larbys-pytorch-0.4.1-nv384.66.img

module load singularity
singularity exec --nv ${CONTAINER} bash -c "cd ${LARFLOW_REPO_DIR_INCONTAINER} && source compile_on_tuftsgrid.sh"