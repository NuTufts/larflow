#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=lmkpsdata
#SBATCH --output=lmkpsdata_bnb_nue.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=8:00:00
#SBATCH --array=0
#SBATCH --partition=preempt
#SBATCH --error=griderr_mktriplettruth.%j.%N.err

container=/cluster/tufts/wongjiradlabnu//larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
DATA_PREP_DIR=/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/larmatchnet/lightmodel

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnbnue_corsika: 2461, 493 jobs (0-492)
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_kps_mcc9_v13_bnbnue_corsika.sh"


# mcc9_v13_bnb_nu_corsika: 2863 files, 573 jobs
#srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_kps_mcc9_v13_bnb_nu_corsika.sh"


