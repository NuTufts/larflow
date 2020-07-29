#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=hadd_kps
#SBATCH --output=log_hadd_larmatchdata_kps.txt
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_pytorch1.3cpu.simg
DATA_PREP_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/larmatchnet/dataprep
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/ubdl

module load singularity
#srun singularity exec ${container} bash -c "cd ${UBDL_DIR} && source setenv.sh && source configure.sh && cd ${DATA_PREP_DIR} && hadd -f larmatch_train_p00.root @trainlist_p00.txt"
srun singularity exec ${container} bash -c "cd ${UBDL_DIR} && source setenv.sh && source configure.sh && cd ${DATA_PREP_DIR} && rm -f /tmp/larmatch_train_p0${SLURM_ARRAY_TASK_ID}.root && hadd -f /tmp/larmatch_train_p0${SLURM_ARRAY_TASK_ID}.root @kps_trainlist_p0${SLURM_ARRAY_TASK_ID}.list && scp /tmp/larmatch_train_p0${SLURM_ARRAY_TASK_ID}.root ${DATA_PREP_DIR}/outhadd_kps/"
#srun singularity exec ${container} bash -c "cd ${UBDL_DIR} && source setenv.sh && source configure.sh && cd ${DATA_PREP_DIR} && hadd -f /tmp/larmatch_valid.root @validlist.txt && scp /tmp/larmatch_valid.root ${DATA_PREP_DIR}/outhadd/"


