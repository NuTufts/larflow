#!/bin/bash
#
#SBATCH --job-name=larflow_deploy_6
#SBATCH --output=log_larflow_deploy_6.txt
#SBATCH --mem-per-cpu=8000
#SBATCH --time=3-00:00:00
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03
#SBATCH --array=6-11

LARFLOW_REPO_DIR=/cluster/tufts/wongjiradlab/twongj01/larflow/
LARFLOW_REPO_DIR_IC=/cluster/kappa/wongjiradlab/twongj01/larflow/
FILELIST_DIR=${LARFLOW_REPO_DIR_IC}/grid/filelists/
OUTPUT_DIR=${LARFLOW_REPO_DIR_IC}/grid/output/
CONTAINER=${LARFLOW_REPO_DIR}/container/singularity-larbys-pytorch-0.4.1-nv384.66.img

module load singularity
singularity exec --nv ${CONTAINER} bash -c "cd ${LARFLOW_REPO_DIR_IC}/grid && source deploy_larflow.sh ${LARFLOW_REPO_DIR_IC} ${FILELIST_DIR} ${OUTPUT_DIR}"
