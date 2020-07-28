#!/bin/bash

# run normally
#./dev_flashmatch ../cluster/output_dev_truthcluster_truepixflow.root ../../testdata/larlite_opreco_5482426_95.root ../../testdata/larcv_5482426_95.root ../../testdata/larlite_mcinfo_5482426_95.root
#./dev_flashmatch ../cluster/output_dev_truthcluster_recopixflow.root ../../testdata/larlite_opreco_5482426_95.root ../../testdata/larcv_5482426_95.root ../../testdata/larlite_mcinfo_5482426_95.root

# for GDB
args="../cluster/outputdev_trutcluster.root ../../testdata/larlite_opreco_5482426_95.root ../../testdata/larcv_5482426_95.root ../../testdata/larlite_mcinfo_5482426_95.root"
echo "args: $args"
gdb dev_flashmatch


