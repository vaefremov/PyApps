#!/bin/bash 

APP=${1:?'App argument not set'}

REPO=${REPO:?"No repository location specified"}
DEPLOY_TO=${DEPLOY_TO:?"No deployment target"}
ATTIC=$DEPLOY_TO/../Attic

SUFFIX=$(date +"%Y%m%d%H%M%S")

# Check if this is a correct app
if [ !  -x $REPO/$APP/app.sh ];  then  echo "Warning: app.sh not executable, will fix" 1>&2 ; fi
if [ ! -f $REPO/$APP/config.json ]; then echo "Config missing" 1>&2 && exit 2; fi
jsonlint-3 $REPO/$APP/config.json 1>&2
if [ $? -ne 0 ]; then echo "Bad config" 1>&2 && exit 3; fi

cd $DEPLOY_TO
if [ -d $APP ]
then
  mv $APP $ATTIC/${APP}_$SUFFIX
fi

cp -r $REPO/$APP $DEPLOY_TO/
chmod +x $DEPLOY_TO/$(basename $APP)/app.sh
