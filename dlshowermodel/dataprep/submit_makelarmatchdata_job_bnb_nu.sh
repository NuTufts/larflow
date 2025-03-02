#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=lmdata
#SBATCH --output=lmdata_bnb_nu_val_sub0.log
#SBATCH --mem-per-cpu=8000
#SBATCH --time=8:00:00
#SBATCH --array=0-172
##SBATCH --partition=preempt
#SBATCH --partition=batch
#SBATCH --error=gridlog_dlshowermodel_bnb_nu_val.%N.%j.%A.%a.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_torch1.9.0_minkowski.sif
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl
DATA_PREP_DIR=${UBDL_DIR}/larflow/dlshowermodel/dataprep/

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnb_nu_corsika: 2863 files
# 2000 for training: 0-399 jobs
# running 5 files per job: 573 jobs needed total
#srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_dlshowermodel_data_mcc9_v13_bnb_nu_corsika_training.sh"

# 863  for validation: 0-172 jobs
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_dlshowermodel_data_mcc9_v13_bnb_nu_corsika_validation.sh"

