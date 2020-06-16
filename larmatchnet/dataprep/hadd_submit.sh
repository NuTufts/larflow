#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=haddlarmatch
#SBATCH --output=log_hadd_larmatchdata_pv0.txt
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1-00:00:00
#SBATCH --array=6-9

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_10022019.simg
DATA_PREP_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/larmatchnet/dataprep
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl

SETUP="cd ${UBDL_DIR} && source setenv.sh && source configure.sh && cd ${DATA_PREP_DIR} && mkdir -p outhadd_kps"

module load singularity
srun singularity exec ${container} bash -c "${SETUP} && hadd -f /tmp/larmatch_kps_train_p0${SLURM_ARRAY_TASK_ID}.root @kps_trainlist_p0${SLURM_ARRAY_TASK_ID}.list && scp /tmp/larmatch_kps_train_p0${SLURM_ARRAY_TASK_ID}.root ${DATA_PREP_DIR}/outhadd_kps/"
#srun singularity exec ${container} bash -c "${SETUP} && hadd -f /tmp/larmatch_valid_p0${SLURM_ARRAY_TASK_ID}.root @validlist_p0${SLURM_ARRAY_TASK_ID}.txt && scp /tmp/larmatch_valid_p0${SLURM_ARRAY_TASK_ID}.root ${DATA_PREP_DIR}/outhadd_kps/"
#srun singularity exec ${container} bash -c "${SETUP} && hadd -f /tmp/larmatch_valid_p00.root @validlist_p00.txt && scp /tmp/larmatch_valid_p00.root ${DATA_PREP_DIR}/outhadd/"
#srun singularity exec ${container} bash -c "${SETUP} && hadd -f /tmp/larmatch_valid_p01.root @validlist_p01.txt && scp /tmp/larmatch_valid_p01.root ${DATA_PREP_DIR}/outhadd/"

