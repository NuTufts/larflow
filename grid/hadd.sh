#!/bin/bash

CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/twongj01/larflow/container/singularity-larbys-pytorch-0.4.1-nv384.66-np1.14.img
WORKDIR_IC=/cluster/kappa/wongjiradlab/twongj01/larflow/

sample="mcc8_bnbcosmic_trueflow"
#sample="mcc9_extbnb_beta1"
#sample="mcc8v11_bnbcosmic_detsys_cv"

singularity exec ${CONTAINER} bash -c "cd ${WORKDIR_IC} && source /usr/local/root/release/bin/thisroot.sh && source configure.sh && cd grid  && hadd -f output_${sample}_larlite.root output/truthflow/*larflowhits*.root"
singularity exec ${CONTAINER} bash -c "cd ${WORKDIR_IC} && source /usr/local/root/release/bin/thisroot.sh && source configure.sh && cd grid  && hadd -f output_${sample}_larcv.root output/truthflow/*larflowlarcv*.root"