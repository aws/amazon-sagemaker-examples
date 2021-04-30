#!/bin/bash

echo "Starting sage-train.sh"

set -e

COACH_EXP_NAME=sagemaker_rl
cd /opt/amazon/

export PYTHONUNBUFFERED=1

# Start the redis server and Coach training worker
redis-server /etc/redis/redis.conf & (sleep 5 && \
python markov/training_worker.py $@ 2>&1)
