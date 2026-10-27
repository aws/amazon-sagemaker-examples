#!/bin/sh -x
#source ./export_locally.sh
echo "Starting the game server auto-pilot"

# This process runs as daemon set per game server deployment. It attempt to get the required number of game-servers within a region.
# It gets this number from an ML model. For testing purpose, it avoids the ML endpoint and uses random numbers instead. 
# In the case of scale down, individual game server pods should protect itself with a termination grace period to avoid player interruption 
# In such cases, an extra x min will be given where terminated pods status will turn to 'terminating'. 
# Game servers can listen to SIGKILL or SIGTERM and trigger a game-server drain (this is not k8s thing. it is a game-server logic)

if [ "${DEMAND_URL}" == "" ]; then
  echo '[ERROR] Environment variable `DEMAND_URL` has no value set.' 1>&2
  exit 1
fi

if [ "${MODEL_URL}" == "" ]; then
  echo '[ERROR] Environment variable `MODEL_URL` has no value set.' 1>&2
  exit 1
fi

if [ "${QUEUE_URL}" == "" ]; then
  echo '[ERROR] Environment variable `QUEUE_URL` has no value set.' 1>&2
  exit 1
fi

if [ "${NAMESPACE}" == "" ]; then
  echo '[ERROR] Environment variable `NAMESPACE` has no value set.' 1>&2
  exit 1
fi

if [ "${POD_NAME}" == "" ]; then
  echo '[ERROR] Environment variable `POD_NAME` has no value set.' 1>&2
  exit 1
fi

if [ "${DEPLOY_NAME}" == "" ]; then
  echo '[ERROR] Environment variable `DEPLOY_NAME` has no value set.' 1>&2
  exit 1
fi

if [ "${CURRENT_SIZE_METRIC_NAME}" == "" ]; then
  echo '[ERROR] Environment variable `CURRENT_SIZE_METRIC_NAME` has no value set.' 1>&2
  exit 1
fi

if [ "${NEW_SIZE_METRIC_NAME}" == "" ]; then
  echo '[ERROR] Environment variable `NEW_SIZE_METRIC_NAME` has no value set.' 1>&2
  exit 1
fi

if [ "${FREQUENCY}" == "" ]; then
  echo '[ERROR] Environment variable `FREQUENCY` has no value set.' 1>&2
  exit 1
fi

if [ "${MIN_GS_NUM}" == "" ]; then
  echo '[ERROR] Environment variable `MIN_GS_NUM` has no value set.' 1>&2
  exit 1
fi

if [ "${AWS_DEFAULT_REGION}" == "" ]; then
  echo '[ERROR] Environment variable `AWS_DEFAULT_REGION` has no value set.' 1>&2
  exit 1
fi

while true
do

  #Testing only - generating random size for replica set. Realworld solution will get this number from an AI/ML endpoint
  #NEW_RS_SIZE=$RANDOM
  #RANGE=30
  #let "NEW_RS_SIZE %= $RANGE"
  #echo NEW_RS_SIZE=${NEW_RS_SIZE}
  #End testing section

  #Getting predictions from a trained model
  TMP_PREDICTION_JSON="/tmp/prediction.json"
  DEMAND_URL=${DEMAND_URL}
  MODEL_URL=${MODEL_URL}
  MODEL_URL_PARAM="/"
  `curl -w "\n" $MODEL_URL/$MODEL_URL_PARAM > $TMP_PREDICTION_JSON`
  NEW_RS_SIZE_FLOAT=`cat $TMP_PREDICTION_JSON | jq '.Prediction.num_of_gameservers'`
  IS_FALSE_POSITIVE=`cat $TMP_PREDICTION_JSON | jq '.Prediction.is_false_positive'`
  if [ -n "${NEW_RS_SIZE_FLOAT}" ]; then
    NEW_RS_SIZE=${NEW_RS_SIZE_FLOAT%.*}
  else
    echo "api returned null, setting rs default="${MIN_GS_NUM}
    NEW_RS_SIZE=${MIN_GS_NUM}
  fi
  echo NEW_RS_SIZE=${NEW_RS_SIZE}
  CURRENT_RS_SIZE=`kubectl get deploy ${DEPLOY_NAME} -n ${NAMESPACE} -o=jsonpath='{.status.availableReplicas}'`
  echo CURRENT_RS_SIZE=${CURRENT_RS_SIZE}

  kubectl scale deploy/${DEPLOY_NAME} --replicas=${NEW_RS_SIZE} -n ${NAMESPACE}
  echo "sleeping for ${SLEEP_TIME_B4_NEXT_READ} to allow the scale operations"

  NUM_NODES=`kubectl get nodes -o json | jq '.items[].metadata.labels'| grep ${NODE_GROUP} | wc -l`
  echo NUM_NODES=${NUM_NODES}

  TIME_ST="$(date '+%Y-%m-%d %H:%M:%S')"
  echo TIMESTAMP=$TIME_ST

  MESSAGE="[{'type': 'autopilot','timestamp':${TIME_ST},'deployment':${DEPLOY_NAME},'current_rs_size': ${CURRENT_RS_SIZE},'new_rs_size':${NEW_RS_SIZE},'region':${AWS_DEFAULT_REGION}}]"
  CURRENT_DEMAND=`curl -w "\n" $DEMAND_URL | jq '.Prediction.num_of_gameservers'`
  echo CURRENT_DEMAND=${CURRENT_DEMAND}
  #aws sqs send-message --queue-url ${QUEUE_URL} --message-body "${MESSAGE}" 
  aws cloudwatch put-metric-data --metric-name ${CURRENT_SIZE_METRIC_NAME} --namespace ${DEPLOY_NAME} --value ${CURRENT_RS_SIZE}
  #aws cloudwatch put-metric-data --metric-name ${NEW_SIZE_METRIC_NAME} --namespace ${DEPLOY_NAME} --value ${NEW_RS_SIZE}
  aws cloudwatch put-metric-data --metric-name current_gs_demand --namespace ${DEPLOY_NAME} --value ${CURRENT_DEMAND}
  aws cloudwatch put-metric-data --metric-name num_of_nodes --namespace ${DEPLOY_NAME} --value ${NUM_NODES}
  aws cloudwatch put-metric-data --metric-name false_positive --namespace ${DEPLOY_NAME} --value ${IS_FALSE_POSITIVE}

  sleep ${FREQUENCY}

done
