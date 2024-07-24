#!/bin/bash -x

APP=${1:?'App argument not set'}

REPO=${REPO:?"/home/efremov/Projects/Repo1"}
DEPLOY_TO=${DEPLOY_TO:?"/tmp/efremov/JobApps"}
ATTIC=$DEPLOY_TO/../Attic

SUFFIX=$(date +"%Y%m%d%H%M%S")

cd $DEPLOY_TO
if [ -d $APP ]
then
  mv $APP $ATTIC/${APP}_$SUFFIX
fi

cp -r $REPO/$APP $DEPLOY_TO/
