#!/bin/bash

#clusterfile=truthcluster_gridtest_larlite.root
#superafile=convertout.root
#larlitefile=opreco-Run000001-SubRun002000.root
#mcinfofile=mcinfo-Run000001-SubRun002000.root

folder=../../testdata/griddebug/
clusterfile=truthcluster_larlite.root
superafile=supera_convertout.root
larlitefile=opreco-Run000001-SubRun006922.root
mcinfofile=reco2d-Run000001-SubRun006922.root


gdb --args ./dev_flashmatch ${folder}/${clusterfile} ${folder}/${larlitefile} ${folder}/${superafile} ${folder}/${mcinfofile} gridtestout_larlite.root gridtestout_larcv.root

#valgrind  ./dev_flashmatch ${folder}/${clusterfile} ${folder}/${larlitefile} ${folder}/${superafile} ${folder}/${mcinfofile} gridtestout_larlite.root >& out.valgrind.txt


