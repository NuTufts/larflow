#!/bin/bash

# args
repodir=$1
filelistdir=$2
outputdir=$3

# jobid
jobid=${SLURM_ARRAY_TASK_ID}
let line=${jobid}+1

# get input files from filelists
supera=`sed -n ${line}p ${filelistdir}/supera_list.txt`
opreco=`sed -n ${line}p ${filelistdir}/opreco_list.txt`
reco2d=`sed -n ${line}p ${filelistdir}/reco2d_list.txt`
mcinfo=`sed -n ${line}p ${filelistdir}/mcinfo_list.txt`

# gpuid
let gpuid=${jobid}%6

# setup container
source /usr/local/root/release/bin/thisroot.sh

# setup larflow repo
cd $repodir
source configure.sh

# add postprocessor folder to PATH
export PATH=${LARFLOW_BASEDIR}/postprocessor:${PATH}

# go to deploy dir
cd ${repodir}/deploy

# deploy on gpu
outfile_y2u=`printf /tmp/larflow_y2u_jobid%02d.root ${jobid}`
outfile_y2v=`printf /tmp/larflow_y2v_jobid%02d.root ${jobid}`

./run_larflow_wholeview.py -i $supera -o $outfile_y2u -c ${repodir}/weights/dev_filtered/devfiltered_larflow_y2u_832x512_32inplanes.tar -f Y2U -g ${gpuid} -b 2 -n 20 --saveadc --ismc
./run_larflow_wholeview.py -i $supera -o $outfile_y2v -c ${repodir}/weights/dev_filtered/devfiltered_larflow_y2v_832x512_32inplanes.tar -f Y2V -g ${gpuid} -b 2 -n 20

outfile_hadd=`printf /tmp/larflow_jobid%02d.root ${jobid}`

hadd -f ${outfile_hadd} ${outfile_y2u} ${outfile_y2v}

# postprocessor
outputpost=`printf /tmp/output_larflowhits_jobid%02d.root ${jobid}`

cd ${repodir}/postprocessor
echo "run: ./dev ${outfile_hadd} ${supera} ${reco2d} ${opreco} ${mcinfo} ${outputpost}"
rm ./pylardcache/*
./dev ${outfile_hadd} ${supera} ${reco2d} ${opreco} ${mcinfo} ${outputpost}

mv ${outputpost} ${outputdir}/

echo "DONE"