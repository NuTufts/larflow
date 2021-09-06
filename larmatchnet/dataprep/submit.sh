#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=lmkpsdata
#SBATCH --output=lmkpsdata_bnb_nu_valid.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=8:00:00
#SBATCH --array=0-79

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdldeps_u20.02_pytorch1.9_py3.simg
DATA_PREP_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnbnue_corsika_training: 2000 files
# mcc9_v13_bnbnue_corsika_valid: 464 files
#srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_kps_mcc9_v13_bnbnue_corsika.sh"


# mcc9_v13_bnb_nu_corsika_training: 2000 files
# mcc9_v13_bnb_nu_corsika_valid: 484 files
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_kps_mcc9_v13_bnb_nu_corsika.sh"


