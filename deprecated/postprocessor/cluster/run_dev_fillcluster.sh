#!/bin/bash

FLASHMATCH=../../testdata/griddebug/flashmatch-larlite-Run000001-SubRun006867.root
SUPERA_LARCV2=../../testdata/griddebug/supera-larcv2-Run000001-SubRun006867.root
OUTPUT=output_fillcluster.root

gdb --args ./dev_fillcluster $FLASHMATCH $SUPERA_LARCV2 $OUTPUT

#valgrind --num-callers=30 --suppressions=$ROOTSYS/etc/valgrind-root.supp ./dev_fillcluster $FLASHMATCH $SUPERA_LARCV2 >& debug_valgrind.out
