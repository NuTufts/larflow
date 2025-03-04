#!/bin/bash

#SBATCH --job-name=ubshowergnn
#SBATCH --output=gridlog_train_ubshower_gnn_%N.%j.txt
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=wongjiradlab
#SBATCH --error=griderr_train_ubshower_gnn.%N.%j.err

# Change WORKDIR to be the folder where this script lives
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/
WORKDIR=${UBDL_DIR}/larflow/dlshowermodel/
#container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/u22.04_cu11.8_torch2.2.2_torchgeometric.sif
module load singularity/3.5.3
NGPUS=1

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container bash -c "source ${WORKDIR}/run_ubshower_gnn_training_p1cmp075.sh ${UBDL_DIR}"
