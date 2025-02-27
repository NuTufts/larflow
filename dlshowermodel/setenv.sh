#!/bin/bash

export JUPYTER_CONFIG_DIR=~/.local/etc/jupyter
export PYTHONNOUSERSITE=1

# add model folder to python path
[[ ":$PYTHONPATH:" != *":${LARFLOW_BASEDIR}:"* ]] && PYTHONPATH="${LARFLOW_BASEDIR}:${PYTHONPATH}"


