#!/bin/bash

DLCOSMICTAG_FILE="../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root"
SUPERA_FILE="../testdata/larcv_5482426_95.root"
RECO2D_FILE="../testdata/larlite_reco2d_5482426_95.root"
OPRECO_FILE="../testdata/larlite_opreco_5482426_95.root"
MCINFO_FILE="../testdata/larlite_mcinfo_5482426_95.root"

# GDB
#echo "./dev $DLCOSMICTAG_FILE $SUPERA_FILE $RECO2D_FILE $OPRECO_FILE $MCINFO_FILE output_truthpixmatch_larlite.root 0"
#gdb ./dev


./dev --use-truth -c $DLCOSMICTAG_FILE -su $SUPERA_FILE -re $RECO2D_FILE -op $OPRECO_FILE -mc $MCINFO_FILE -oll output_truthpixmatch_larlite.root -olc output_truthpixmatch_larcv.root -n 1
