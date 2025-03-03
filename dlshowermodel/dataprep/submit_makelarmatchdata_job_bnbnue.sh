#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=dlshwrdata
#SBATCH --output=dlshwrdata_valid_sub0.log
#SBATCH --mem-per-cpu=8000
#SBATCH --time=8:00:00
#SBATCH --array=0-92
##SBATCH --partition=preempt
#SBATCH --partition=batch
##SBATCH --partition=wongjiradlab
#SBATCH --error=gridlog_make_dlshowermodeldata_bnb_nue.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_torch1.9.0_minkowski.sif
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl
DATA_PREP_DIR=${UBDL_DIR}/larflow/dlshowermodel/dataprep/

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnbnue_corsika: 2461 files
# training split: 2000 files / 5 per job  = 0-399
# validation split: 461 files / 5 per job = 0-92
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_dlshowermodel_data_mcc9_v13_bnbnue_corsika_training.sh"
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_dlshowermodel_data_mcc9_v13_bnbnue_corsika_validation.sh"


