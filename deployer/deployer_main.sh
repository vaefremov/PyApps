#!/bin/bash -x
# Main script of deployer

BIN=$(readlink -f $(dirname $0))

export REPO="/home/efremov/Projects/Repo1"
export DEPLOY_TO="/tmp/efremov/JobApps"
export ATTIC="$DEPLOY_TO/../Attic"


pull_check_updates()
{
    cd $REPO

    PREV_VER=$(git rev-parse HEAD)
    git pull >/dev/null

    FILES=$(git diff --name-only $PREV_VER origin/master)

    APPS=$(for f in $FILES ;do echo $(dirname $f); done | sort | uniq)
    echo $APPS
}

# APPS=$($BIN/pull_check_updates.sh)
APPS=$(pull_check_updates)

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
