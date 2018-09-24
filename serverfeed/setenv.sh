#!/bin/sh

# setups environment variables for larcvdataset, an additional wrapper to larcv::ThreadFiller
# assumes script called in the folder code sits

# clean up previously set env
if [[ -z $FORCE_SERVERFEED_BASEDIR ]]; then
    where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export SERVERFEED_BASEDIR=${where}
else
    export SERVERFEED_BASEDIR=$FORCE_SERVERFEED_BASEDIR
fi

# Add to python path
[[ ":$PYTHONPATH:" != *":${SERVERFEED_BASEDIR}:"* ]] && PYTHONPATH="${SERVERFEED_BASEDIR}:${PYTHONPATH}"
# Add to ld library path
[[ ":$LD_LIBRARY_PATH:" != *":${SERVERFEED_BASEDIR}/lib:"* ]] && LD_LIBRARY_PATH="${SERVERFEED_BASEDIR}/lib:${LD_LIBRARY_PATH}"
