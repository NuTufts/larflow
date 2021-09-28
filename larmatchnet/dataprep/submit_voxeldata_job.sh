#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=voxeldata
#SBATCH --output=larvoxeldata_bnb_nu_train.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=4:00:00
#SBATCH --array=0-47
#SBATCH --partition=preempt

container=/cluster/tufts/wongjiradlabnu//larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0.sif
#container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdldeps_u20.02_pytorch1.9_py3.simg
DATA_PREP_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnbnue_corsika_training: 2000 files
# mcc9_v13_bnbnue_corsika_valid: 464 files
#srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_kps_mcc9_v13_bnbnue_corsika.sh"


# mcc9_v13_bnb_nu_corsika_training: 240 files/stride 5 = 48 jobs
# mcc9_v13_bnb_nu_corsika_valid: 74 files/stride 5 = 15 jobs
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_voxeldata_mcc9_v13_bnb_nu_corsika.sh"


