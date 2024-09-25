#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=lmdata
#SBATCH --output=lmdata_bnb_nu_validation_submission_makeup02.log
#SBATCH --mem-per-cpu=8000
#SBATCH --time=8:00:00
#SBATCH --array=0-6
##SBATCH --partition=preempt
#SBATCH --partition=batch
#SBATCH --error=gridlog_makelarmatchdata_bnb_nu.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_compute8_wjupyternotebook.sif
DATA_PREP_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnb_nu_corsika: 2863 files
# 2000 for training: 0-399 jobs
# 863  for validation: 0-172 jobs
# running 5 files per job: 573 jobs needed total
#srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_larmatchdata_mcc9_v13_bnb_nu_corsika.sh"
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_larmatchdata_mcc9_v13_bnb_nu_corsika_validation.sh"


