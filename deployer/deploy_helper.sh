#!/bin/bash
# Deploy DbHelper and DiPropagation from the last build

DEPLOY_TO=${DEPLOY_TO:?"No deployment target"}/..

DICI="/hdd1/DiCi"
LAST_BUILD_STAMP="$DEPLOY_TO/dbhelper_last_build.txt"

current_build=$(ls -tr $DICI | tail -1)
last_build=$(cat $LAST_BUILD_STAMP)

deploy_helper()
{
    echo $(date) "Deploy helper from $DICI/$current_build/Linux/bin/ to $DEPLOY_TO/DbHelper"
    pkill DbHelper
    sleep 1
    cp $DICI/$1/Linux/bin/DbHelper $DEPLOY_TO/DbHelper
    cd $DEPLOY_TO
    nohup ./start_helper.sh &
}

deploy_propagator()
{
    echo $(date) "Deploy propagator from $DICI/$current_build/Linux/bin/JobApps/DiPropagation/DiPropagation to $DEPLOY_TO/JobApps/DiPropagation/"
    cp $DICI/$1/Linux/bin/JobApps/DiPropagation/DiPropagation $DEPLOY_TO/JobApps/DiPropagation/DiPropagation

}

update_stamp()
{
    echo $1 > $LAST_BUILD_STAMP
}

if [ "$DICI/$current_build" != "$DICI/$last_build" ]
then
    deploy_helper $current_build
    deploy_propagator $current_build
    update_stamp $current_build
fi