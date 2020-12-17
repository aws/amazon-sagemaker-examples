#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <s3-bucket-name>"
    exit 1
fi

S3_BUCKET=$1
S3_PREFIX="mask-rcnn/sagemaker/input"

mounted=$(df -kh | grep "$HOME/efs" | wc -l)
if [ "$mounted" -lt "1" ]; then
   echo "EFS file-system is not mounted"
   exit 1
else
   echo "EFS file-system is mounted"
   echo $(df -kh | grep "$HOME/efs")
fi

if [ -d $HOME/efs/$S3_PREFIX ]; then
    echo "$HOME/efs/$S3_PREFIX already exists"
    exit 1
fi
sudo mkdir -p $HOME/efs/$S3_PREFIX

echo "`date`: Copying files from s3://$S3_BUCKET/$S3_PREFIX [ eta 46 minutes ]"
sudo aws s3 cp --recursive s3://$S3_BUCKET/$S3_PREFIX $HOME/efs/$S3_PREFIX | awk 'BEGIN {ORS="="} {if(NR%200==0)print "="}'
echo "Done."
