#!/bin/bash

#replace with your endpoint name in format https://<<endpoint-name>>
export ENDPOINT_NAME=https://$1

export REGION=$2
export CONTENT_TYPE=text/csv
# export PAYLOAD='{"inputs": "I am super happy right now."}'
export USERS=50
export WORKERS=5
export RUN_TIME=5m
export LOCUST_UI=false # Use Locust UI

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"

#replace with the locust script that you are testing, this is the locust_script that will be used to make the InvokeEndpoint API calls. 
export SCRIPT=locust_script.py

#make sure you are in a virtual environment
#. ./venv/bin/activate

if $LOCUST_UI ; then
    locust -f $SCRIPT -H $ENDPOINT_NAME --master --expect-workers $WORKERS -u $USERS -t $RUN_TIME --csv results_$1 &
else
    locust -f $SCRIPT -H $ENDPOINT_NAME --master --expect-workers $WORKERS -u $USERS -t $RUN_TIME --csv results_$1 --headless &
fi

for (( c=1; c<=$WORKERS; c++ ))
do 
    locust -f $SCRIPT -H $ENDPOINT_NAME --worker --master-host=localhost &
done