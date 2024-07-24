#!/bin/bash -x
# 
#
BIN=$(readlink -f $(dirname $0))
REPO="/home/efremov/Projects/Repo1"

cd $REPO

PREV_VER=$(git rev-parse HEAD)
git pull >/dev/null

FILES=$(git diff --name-only $PREV_VER origin/master)

APPS=$(for f in $FILES ;do echo $(dirname $f); done | sort | uniq)
echo $APPS
