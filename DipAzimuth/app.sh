#!/bin/bash -x
if [ -d ../venv ] 
then
  . ../venv/bin/activate
else 
  . ./venv/bin/activate
fi
DIR=$(dirname $0)
export PYTHONPATH=.
exec python ${DIR}/main.py $@
