#!/bin/bash
#SBATCH --job-name=hadd_valid
#SBATCH --output=log_hadd_valid.txt
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4000

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_singularity_031219.img
WORKDIR=/cluster/kappa/90-days-archive/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/prepsparsedata
OUTROOT=${WORKDIR}/larflow_sparsify_valid.root

module load singularity
srun singularity exec  ${CONTAINER} bash -c "source /usr/local/root/build/bin/thisroot.sh && cd ${WORKDIR} && hadd -f ${OUTROOT} @validlist.txt"
