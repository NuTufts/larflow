#!/bin/bash

DLCOSMICTAG="../../testdata/griddebug/dlcosmictag-larcv2-Run000001-SubRun006867.root"
SUPERA="../../testdata/griddebug/supera-larcv2-Run000001-SubRun006867.root"

gdb --args ./stitch_dlcosmic_images ${DLCOSMICTAG} ${SUPERA} out_stitched_larlite.root
