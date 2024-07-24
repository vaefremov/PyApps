#!/bin/bash -x
# Main script of deployer

BIN=$(readlink -f $(dirname $0))

export REPO="/home/efremov/Projects/Repo1"
export DEPLOY_TO="/tmp/efremov/JobApps"
export ATTIC="$DEPLOY_TO/../Attic"


APPS=$($BIN/pull_check_updates.sh)

for a in $APPS
do
  if [ "$a" == "di_lib" ]
  then
    $BIN/deploy_lib.sh
    echo $(date) "di_lib deployed"
  else
    $BIN/deploy_app.sh $a
    echo $(date) "App $a deployed"
  fi
done
