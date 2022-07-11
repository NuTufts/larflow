#!/bin/bash

#SBATCH --job-name=larmatch
#SBATCH --output=gridlog_train_larmatch_nossnet_ccgpu_jan5.log
#SBATCH --mem-per-cpu=40000
#SBATCH --cpus-per-task=5
#SBATCH --time=8-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=ccgpu
#SBATCH --error=gridlog_train_larmatch_ccgpu_jan5.%j.%N.err

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larmatch/
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
module load singularity/3.5.3

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container bash -c "source ${WORKDIR}/run_larmatch_training.sh 4"
#echo "TEST"
