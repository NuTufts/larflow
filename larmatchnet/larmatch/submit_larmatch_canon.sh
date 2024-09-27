#!/bin/bash

#SBATCH --job-name=larmatch
#SBATCH --output=gridlog_train_larmatch_wpaf_fullruntest_canon.log
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128g
#SBATCH --partition=iaifi_gpu
#SBATCH --error=gridlog_train_larmatch.%j.%N.err

# Change WORKDIR to be the folder where this script lives
WORKDIR=/n/home01/twongjirad/larmatch_retrain/ubdl/larflow/larmatchnet/larmatch/
container=/n/home01/twongjirad/containers/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_compute8_wjupyternotebook.sif
#module load singularity/3.5.3
NGPUS=2

singularity exec --nv $container bash -c "source ${WORKDIR}/run_larmatch_training_canon.sh ${NGPUS}"
#echo "TEST"
