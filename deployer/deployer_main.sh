#!/bin/bash
# Main script of deployer

BIN=$(readlink -f $(dirname $0))

# export REPO="/data/DI_apps/DbHelper/Tmp/PyApps"
export REPO="/hdd1/DI_apps/DbHelper/Tmp/PyApps"
# export DEPLOY_TO="/data/DI_apps/DbHelper/JobApps"
export DEPLOY_TO="/hdd1/DI_apps/DbHelper/JobApps"
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
  case $a in
    Old_programs/*) 
      echo "Skipped deployment: $a" 
      ;;

    di_lib) 
      $BIN/deploy_lib.sh
      echo $(date) "di_lib deployed"
      ;;
      
    *) echo "Try deployment: $a"
      $BIN/deploy_app.sh $a
      if [ $? -eq 0 ]; then
        echo $(date) "App $a deployed"
      else
        echo "Error in deployment of $a"
      fi
  esac
done

$BIN/deploy_helper.sh
