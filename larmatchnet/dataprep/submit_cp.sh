#!/bin/bash
#SBATCH --job-name=xferlarmatchdata
#SBATCH --output=log_xfer_larmatchdata.log
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1-00:00:00

OUTDIR=/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/larmatchnet/dataprep/outhadd_kps/
FNAME="larmatch_*.root"
srun rsync -av --progress $OUTDIR/$FNAME twongj01@goeppert.lns.mit.edu:/home/twongj01/data/larmatch_kpsa_data/
