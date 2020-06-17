#!/bin/bash

#SBATCH --job-name=hadd_cropped
#SBATCH --output=log_hadd_cropped.txt
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4000

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_singularity_031219.img
WORKDIR=/cluster/kappa/90-days-archive/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/prepsparsedata

VER=v5

OUTROOT1=${WORKDIR}/larflow_sparsify_cropped_train1_${VER}.root
OUTROOT2=${WORKDIR}/larflow_sparsify_cropped_train2_${VER}.root
OUTROOT3=${WORKDIR}/larflow_sparsify_cropped_train3_${VER}.root
OUTROOT4=${WORKDIR}/larflow_sparsify_cropped_valid_${VER}.root

module load singularity

singularity exec  ${CONTAINER} bash -c "source /usr/local/root/build/bin/thisroot.sh && cd ${WORKDIR} && hadd -f ${OUTROOT1} @trainlist1.txt"
singularity exec  ${CONTAINER} bash -c "source /usr/local/root/build/bin/thisroot.sh && cd ${WORKDIR} && hadd -f ${OUTROOT2} @trainlist2.txt"
singularity exec  ${CONTAINER} bash -c "source /usr/local/root/build/bin/thisroot.sh && cd ${WORKDIR} && hadd -f ${OUTROOT3} @trainlist3.txt"
singularity exec  ${CONTAINER} bash -c "source /usr/local/root/build/bin/thisroot.sh && cd ${WORKDIR} && hadd -f ${OUTROOT4} @validlist.txt"

