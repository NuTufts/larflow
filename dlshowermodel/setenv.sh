#!/bin/bash

export JUPYTER_CONFIG_DIR=~/.local/etc/jupyter

# add model folder to python path
[[ ":$PYTHONPATH:" != *":${LARFLOW_BASEDIR}:"* ]] && PYTHONPATH="${LARFLOW_BASEDIR}:${PYTHONPATH}"


