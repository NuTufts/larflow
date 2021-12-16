#!/bin/bash

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# add to python path
[[ ":$PYTHONPATH:" != *":${HERE}:"* ]] && export PYTHONPATH="${HERE}:${PYTHONPATH}"
