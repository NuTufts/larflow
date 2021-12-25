#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=lmdata
#SBATCH --output=lmdata_bnb_nu.log
#SBATCH --mem-per-cpu=8000
#SBATCH --time=2:00:00
#SBATCH --array=68,69,70,71,175,176,177,178
##SBATCH --partition=preempt
#SBATCH --partition=batch
#SBATCH --error=gridlog_makelarmatchdata_bnb_nu.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
DATA_PREP_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larmatch/prep/

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnb_nu_corsika: 573 files
# running 5 files per job: 114 jobs needed
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_larmatchdata_mcc9_v13_bnb_nu_corsika.sh"


