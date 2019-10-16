#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=haddlarmatch
#SBATCH --output=log_hadd_larmatchdata_p0v.txt
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1-00:00:00

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_091319.simg
DATA_PREP_DIR=/cluster/tufts/wongjiradlab/twongj01/dev/ubdl/larflow/larmatchnet/dataprep
UBDL_DIR=/cluster/tufts/wongjiradlab/twongj01/dev/ubdl

module load singularity
#srun singularity exec ${container} bash -c "cd ${UBDL_DIR} && source setenv.sh && source configure.sh && cd ${DATA_PREP_DIR} && hadd -f larmatch_train_p00.root @trainlist_p00.txt"
#srun singularity exec ${container} bash -c "cd ${UBDL_DIR} && source setenv.sh && source configure.sh && cd ${DATA_PREP_DIR} && hadd -f /tmp/larmatch_train_p01.root @trainlist_p01.txt && scp /tmp/larmatch_train_p01.root ${DATA_PREP_DIR}/outhadd/"
srun singularity exec ${container} bash -c "cd ${UBDL_DIR} && source setenv.sh && source configure.sh && cd ${DATA_PREP_DIR} && hadd -f /tmp/larmatch_valid.root @validlist.txt && scp /tmp/larmatch_valid.root ${DATA_PREP_DIR}/outhadd/"

