#!/bin/bash

#SBATCH --job-name=larmatch
#SBATCH --output=gridlog_train_larmatch_nossnet_p1cmp075.log
#SBATCH --mem-per-cpu=4g
#SBATCH --cpus-per-task=32
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:p100:6
#SBATCH --partition=wongjiradlab
#SBATCH --error=gridlog_train_larmatch.%j.%N.err

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/icdl/larflow/larmatchnet/larmatch/
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
module load singularity/3.5.3

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container bash -c "source ${WORKDIR}/run_larmatch_training_p100.sh 6"
#echo "TEST"
