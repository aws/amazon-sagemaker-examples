#!/bin/bash

echo "starting locust"
locust -f $SCRIPT -H $ENDPOINT_NAME --master \
    --expect-workers $WORKERS -u $USERS -t $RUN_TIME -r $SPAWN_RATE --csv $RESULTS_PREFIX --headless &


for (( c=1; c<=$WORKERS; c++ ))
do 
    echo "starting locust worker $c"
    locust -f $SCRIPT -H $ENDPOINT_NAME --worker --master-host=localhost &
done


echo "waiting for locust testing to complete"
wait