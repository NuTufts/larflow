#!/bin/bash
#
#SBATCH --job-name=larflow
#SBATCH --output=larflow.log
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03

WORKDIR_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/larflow/training
LARFLOW_REPO_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/larflow
DATADIR_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/llf/larflow/datasets
DATALOADER_CFG=${WORKDIR_IN_CONTAINER}/tufts_flowloader_832x512_y2u_train.cfg
DATALOADER_CFG=${WORKDIR_IN_CONTAINER}/tufts_flowloader_832x512_y2u_valid.cfg
CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/singularity-larbys-pytorch/singularity-pytorch-0.3-larcv2-nvidia384.66.img

module load singularity
singularity exec --nv ${CONTAINER} bash -c "cd ${LARFLOW_REPO_IN_CONTAINER}/grid_scripts && source run_train_wlarcv.sh ${WORKDIR_IN_CONTAINER} ${LARFLOW_REPO_IN_CONTAINER} ${DATADIR_IN_CONTAINER} ${DATALOADER_CFG}"
