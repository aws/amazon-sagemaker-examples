#!/bin/sh

echo "Creating docker image repositories"
aws cloudformation create-stack --stack-name ecr-repos --template-body file://./ecr-repos.json

echo "Creating DynamoDB table latest-observations"
aws cloudformation create-stack --stack-name latest-observations-dynamodb-table --template-body file://./latest_observations-table.json

echo "Waiting for the DynamoDB table latest-observations to be created"
aws dynamodb wait table-exists --table-name latest_observations

echo "Initializing latest-observations"
aws dynamodb put-item --table-name latest_observations --item file://./init_observation.json
