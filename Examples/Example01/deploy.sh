#!/bin/bash -x
WHERE=${1:?"No destination"}
DIR=$(dirname $0)
#SCRIPT_DIR=$(basename $(dirname $0))
#SCRIPT_NAME=${SCRIPT_DIR::${#SCRIPT_DIR}-5}
cp -r $DIR $WHERE
#cp $WHERE/$DIR/$SCRIPT_NAME $WHERE
