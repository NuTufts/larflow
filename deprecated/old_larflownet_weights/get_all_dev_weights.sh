#!/bin/bash

user=$1

rsync -av --progress ${user}@xfer.cluster.tufts.edu:/cluster/tufts/wongjiradlab/twongj01/larflow/weights/dev_filtered .
