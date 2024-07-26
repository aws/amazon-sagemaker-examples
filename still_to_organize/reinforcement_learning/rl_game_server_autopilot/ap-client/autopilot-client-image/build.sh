#!/bin/bash

source account.conf
region="us-west-2"
repo='.dkr.ecr.'$region'.amazonaws.com/autopilot:latest'
repo_url=$account$repo

$(aws ecr get-login --no-include-email --region $region)
docker build -t autopilot .
docker tag autopilot:latest $repo_url
docker push $repo_url
