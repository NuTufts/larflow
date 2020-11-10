#!/bin/bash

#SBATCH --job-name=spisminmax
#SBATCH --output=logs/minmax/spisminmax.%A.log
#SBATCH --error=logs/minmax/spisminmax_error.%A.log
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16g
#SBATCH --time=01:00:00

#container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_pytorch1.3cpu.simg
container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_dldependencies_pytorch1.3.sing

UBDL_DIR=/cluster/tufts/wongjiradlab/jhwang11/ubdl
SCRIPTDIR=${UBDL_DIR}/larflow/larflow/SpatialEmbed
MODEL=${UBDL_DIR}/larflow/larflow/SpatialEmbed/trained_models/spatialembed_model_11_07_short.pt
RESULTS=${UBDL_DIR}/larflow/larflow/SpatialEmbed/trained_models/spatialembed_model_11_07_short_test_evaluations.pickle

SCRIPT="python ${UBDL_DIR}/larflow/larflow/SpatialEmbed/processEvaluations.py"
COMMAND="${SCRIPT} -m ${MODEL} -r ${RESULTS}"

module load singularity

srun singularity exec --nv ${container} bash -c "cd $UBDL_DIR && source setenv.sh && source configure.sh && cd $SCRIPTDIR && $COMMAND"
#srun -p gpu --gres=gpu:p100:1 singularity exec --nv ${container} bash -c "nvidia-smi &&  cd $SCRIPTDIR && $COMMAND"
#srun -p gpu --gres=gpu:1 singularity exec --nv ${container} bash -c "cd $UBDL_DIR && source setenv.sh && source configure.sh && cd $SCRIPTDIR && $COMMAND"