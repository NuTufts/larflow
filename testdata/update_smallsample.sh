#!/bin/bash

user=$1

rsync -av --progress smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root ${user}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/twongj01/larflow/testdata/smallsample/
