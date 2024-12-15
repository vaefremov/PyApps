#!/bin/bash
if [ -d ../venv ] 
then
  . ../venv/bin/activate
else 
  . ./venv/bin/activate
fi

# Limit number of threads numpy uses:
export OPENBLAS_NUM_THREADS=4
export GOTO_NUM_THREADS=4
export OMP_NUM_THREADS=4

DIR=$(dirname $0)
export PYTHONPATH=.
exec python ${DIR}/main.py $@
