#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=lvclassdata
#SBATCH --output=lvclass_data_bnb_nu.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=2:00:00
#SBATCH --array=0
#SBATCH --partition=preempt
##SBATCH --partition=wongjiradlab
#SBATCH --error=gridlog_lvclassdata_bnb_nu.%j.%N.err

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
DATA_PREP_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel/prepdata/

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnbnue_corsika: 493 files
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_larvoxel_data_making_script_bnb_nu.sh"
