#!/bin/bash

module load singularity

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_singularity_031219.img
workdir=/cluster/kappa/90-days-archive/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/prepsparsedata/

module load singularity
singularity exec ${container} bash -c "source /usr/local/root/build/bin/thisroot.sh && cd ${workdir} && python check_cropped_jobs.py"
