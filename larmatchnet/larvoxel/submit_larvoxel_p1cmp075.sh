#!/bin/bash

#SBATCH --job-name=larmatch
#SBATCH --output=gridlog_train_larvoxel_p1cmp075.log
#SBATCH --mem-per-cpu=16000
#SBATCH --cpus-per-task=5
#SBATCH --time=3-00:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:4
#SBATCH --error=gridlog_train_larvoxel.%j.%N.err

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/larvoxel
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
module load singularity/3.5.3

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container bash -c "source ${WORKDIR}/run_larvoxel_classifier_training_p100.sh 4"
#echo "TEST"
