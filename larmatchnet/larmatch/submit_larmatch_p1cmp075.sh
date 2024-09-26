#!/bin/bash

#SBATCH --job-name=larmatch
#SBATCH --output=gridlog_train_larmatch_wpaf_fullruntest_p1cmp075.log
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=30
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:p100:4
#SBATCH --partition=wongjiradlab
#SBATCH --error=gridlog_train_larmatch.%j.%N.err

WORKDIR=/cluster/tufts/wongjiradlabnu/ndahle01/ubdl/larflow/larmatchnet/larmatch

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
module load singularity/3.5.3
NGPUS=4

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container bash -c "source ${WORKDIR}/run_larmatch_training_p100.sh ${NGPUS}"
#echo "TEST"
