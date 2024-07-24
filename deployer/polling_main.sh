#!/bin/bash
# Script that runs deployer_main.sh periodically.

BIN=$(readlink -f $(dirname $0))
INTERVAL_SEC=30
echo "Start $INTERVAL_SEC sec " $(date) 
while true
do
  sleep $INTERVAL_SEC
  $BIN/deployer_main.sh
done
