#!/bin/bash

#SBATCH --job-name=train_larmatch
#SBATCH --output=gridlog_train_lvmultidecoder_pass0.log
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=5
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=ccgpu
#SBATCH --error=gridlog_train_lvmultidecoder.%j.%N.err

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0.sif
module load singularity/3.5.3

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container bash -c "source ${WORKDIR}/run_larvoxel_training.sh 4"
#echo "TEST"
