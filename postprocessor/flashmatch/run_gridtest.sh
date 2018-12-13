#!/bin/bash

#clusterfile=truthcluster_gridtest_larlite.root
#superafile=convertout.root
#larlitefile=opreco-Run000001-SubRun002000.root
#mcinfofile=mcinfo-Run000001-SubRun002000.root

folder=../../testdata/griddebug/
clusterfile=truthcluster-larlite-Run000001-SubRun006914.root
superafile=supera-larcv2-Run000001-SubRun006914.root
larlitefile=opreco-Run000001-SubRun006914.root
mcinfofile=reco2d-Run000001-SubRun006914.root

lcvout=gridtest_larcv.root
llout=gridtest_larlite.root
ana=gridtest_ana.root

gdb --args ./dev_flashmatch ${folder}/${clusterfile} ${folder}/${larlitefile} ${folder}/${superafile} ${folder}/${mcinfofile} ${llout} ${lcvout} ${ana}

#valgrind --num-callers=30 --suppressions=$ROOTSYS/etc/valgrind-root.supp ./dev_flashmatch ${folder}/${clusterfile} ${folder}/${larlitefile} ${folder}/${superafile} ${folder}/${mcinfofile} ${llout} ${lcvout} ${ana} >& debug_valgrind.out


