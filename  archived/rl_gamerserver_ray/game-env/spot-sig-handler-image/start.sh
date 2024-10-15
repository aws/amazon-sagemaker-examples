#!/bin/sh

echo "Starting the spot int handler"


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

echo "Getting the node name"
NODE_NAME=$(kubectl --namespace ${NAMESPACE} get pod ${POD_NAME} --output jsonpath="{.spec.nodeName}")
echo NODE_NAME=${NODE_NAME}

if [ "${NODE_NAME}" == "" ]; then
  echo "[ERROR] Unable to fetch the name of the node running the pod \"${POD_NAME}\" in the namespace \"${NAMESPACE}\"." 1>&2
  exit 1
fi

echo "Gather some information"
INSTANCE_ID_URL=${INSTANCE_ID_URL:-http://169.254.169.254/latest/meta-data/instance-id}
INSTANCE_ID=$(curl -s ${INSTANCE_ID_URL})
echo INSTANCE_ID_URL=${INSTANCE_ID_URL}
echo INSTANCE_ID=${INSTANCE_ID}

INSTANCE_TYPE=$(kubectl get nodes ${NODE_NAME} -L beta.kubernetes.io/instance-type| grep -v INSTANCE-TYPE| awk '{print $NF}')
echo INSTANCE_TYPE=$INSTANCE_TYPE

INSTANCE_AZ=$(kubectl get nodes ${NODE_NAME} -L failure-domain.beta.kubernetes.io/zone| grep -v ZONE| awk '{print $NF}')
echo INSTANCE_AZ=$INSTANCE_AZ

INSTANCE_GROUP=$(kubectl get nodes ${NODE_NAME} -L alpha.eksctl.io/nodegroup-name| grep -v NODEGROUP-NAME| awk '{print $NF}')
echo INSTANCE_GROUP=$INSTANCE_GROUP

TIME_ST="$(date '+%Y-%m-%d %H:%M:%S')"
echo TIMESTAMP=$TIME_ST

echo "\`kubectl drain ${NODE_NAME}\` will be executed once a termination notice is made."


POLL_INTERVAL=${POLL_INTERVAL:-5}
NOTICE_URL=${NOTICE_URL:-http://169.254.169.254/latest/meta-data/spot/termination-time}

#echo "Send test message to queue ${QUEUE_URL} with message ${MESSAGE}"
#MESSAGE="[{'status': 'test spot termination', 'public_hostname': ${NODE_NAME}, 'public_port': NA, 'region': 'us-east'}]"
#MESSAGE_GRP_ID="gsGrp_us-east-1"
#aws sqs send-message --queue-url ${QUEUE_URL} --message-body "${MESSAGE}" --message-group-id ${MESSAGE_GRP_ID}

echo "Polling ${NOTICE_URL} every ${POLL_INTERVAL} second(s)"
while http_status=$(curl -o /dev/null -w '%{http_code}' -sL ${NOTICE_URL}); [ ${http_status} -ne 200 ]; do
  echo $(date): ${http_status}
  sleep ${POLL_INTERVAL}
done

echo $(date): ${http_status}

MESSAGE="[{'status': 'spot termination','instance_az': ${INSTANCE_AZ},'instance_group':${INSTANCE_GROUP},'time':${TIME_ST}, 'instance_type': ${INSTANCE_TYPE}, 'public_hostname': ${NODE_NAME}, 'public_port': NA, 'region': ${AWS_DEFAULT_REGION}]"
echo MESSAGE=$MESSAGE
aws sqs send-message --queue-url ${QUEUE_URL} --message-body "${MESSAGE}" 

echo "Drain the node."
kubectl drain ${NODE_NAME} --force --ignore-daemonsets

echo "Sleep for 200 seconds to prevent raise condition"
# The instance should be terminated by the end of the sleep assumming 120 sec notification time
sleep 200
