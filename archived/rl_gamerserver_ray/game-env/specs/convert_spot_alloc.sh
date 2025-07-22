#!/bin/sh

asg_name=`aws autoscaling describe-auto-scaling-groups| jq '.AutoScalingGroups[].AutoScalingGroupName' | grep rl-gs-autopilot`
aws autoscaling update-auto-scaling-group --auto-scaling-group-name $asg_name --mixed-instances-policy file://./mixed-instance-policy.json
