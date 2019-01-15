#!/bin/bash
#
#SBATCH --job-name=compile_larflow
#SBATCH --output=log_compile_larflow.log
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --partition batch

LARFLOW_REPO_DIR=/cluster/tufts/wongjiradlab/twongj01/dev/larflow/
LARFLOW_REPO_DIR_INCONTAINER=/cluster/kappa/wongjiradlab/twongj01/dev/larflow/
CONTAINER=/cluster/tufts/wongjiradlab/larbys/images/singularity-larflow/singularity-larflow-v2.img

module load singularity

#  clean
#singularity exec --nv ${CONTAINER} bash -c "cd ${LARFLOW_REPO_DIR_INCONTAINER} && source /usr/local/root/release/bin/thisroot.sh && source configure.sh && source clean.sh"

# compile
srun singularity exec --nv ${CONTAINER} bash -c "cd ${LARFLOW_REPO_DIR_INCONTAINER} && source compile_on_tuftsgrid.sh ${LARFLOW_REPO_DIR_INCONTAINER}"

# first build
#srun singularity exec --nv ${CONTAINER} bash -c "cd ${LARFLOW_REPO_DIR_INCONTAINER} && source first_build.sh"