#!/bin/bash

TESTDIR=../../testdata/mcc9v12_intrinsicoverlay/
LARFLOWOUT=${TESTDIR}/larflow-noinfill-larcv-mcc9v12_intrinsicoverlay-Run004955-SubRun000079.root
SUPERA_FILE=${TESTDIR}/supera-Run004955-SubRun000079.root
RECO2D_FILE=${TESTDIR}/reco2d-Run004955-SubRun000079.root
OPRECO_FILE=${TESTDIR}/opreco-Run004955-SubRun000079.root
SSNET_FILE=${TESTDIR}/ssnetserveroutv2-larcv-Run004955-SubRun000079.root

#gdb --args ./dev -c $LARFLOWOUT -su $SUPERA_FILE -op $OPRECO_FILE -wss $SSNET_FILE -oll output_pp_test_larlite.root -olc output_pp_test_larcv.root -n 1
./dev -c $LARFLOWOUT -su $SUPERA_FILE -op $OPRECO_FILE -wss $SSNET_FILE -oll output_pp_test_larlite.root -olc output_pp_test_larcv.root
