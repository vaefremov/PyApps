#!/bin/bash
# Main script of deployer

BIN=$(readlink -f $(dirname $0))

# export REPO="/tmp/efremov/Repo1"
export REPO="/data/DI_apps/DbHelper/Tmp/PyApps"
# export DEPLOY_TO="/tmp/efremov/JobApps"
export DEPLOY_TO="/data/DI_apps/DbHelper/JobApps"
export ATTIC="$DEPLOY_TO/../Attic"


pull_check_updates()
{
    cd $REPO

    PREV_VER=$(git rev-parse HEAD)
    git pull >/dev/null

    CUR_VER=$(git rev-parse HEAD)

    [ "$PREV_VER" == "$CUR_VER" ] && exit 0

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
    if [ $? -eq 0 ]; then
      echo $(date) "App $a deployed"
    else
      echo "Error in deployment of $a"
    fi
  fi
done
