#!/bin/bash

user=$1

mkdir -p smallsample
rsync -av --progress ${user}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/twongj01/larflow/testdata/smallsample/*.root smallsample/
