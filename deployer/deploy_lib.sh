#!/bin/bash
# Deploy di_lib library

APP="di_lib"

REPO=${REPO:?"Repo not set"}
DEPLOY_TO=${DEPLOY_TO:?"Deployment target not set"}/..
ATTIC="$DEPLOY_TO/Attic"

SUFFIX=$(date +"%Y%m%d%H%M%S")

cd $DEPLOY_TO
if [ -d $APP ]
then
  mv $APP $ATTIC/${APP}_$SUFFIX
fi

cp -r $REPO/$APP $DEPLOY_TO/
