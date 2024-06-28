#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <s3-bucket-name>"
    exit 1
fi

S3_BUCKET=$1
S3_PREFIX="mask-rcnn/sagemaker/input"

mounted=$(df -kh | grep "$HOME/fsx" | wc -l)
if [ "$mounted" -lt "1" ]; then
   echo "FSx for Lustre file-system is not mounted"
   exit 1
else
   echo "FSx for Lustre file-system is mounted"
   echo $(df -kh | grep "$HOME/fsx")
fi

if [ -d $HOME/fsx/$S3_PREFIX ]; then
    echo "$HOME/fsx/$S3_PREFIX already exists"
    exit 1
fi
sudo mkdir -p $HOME/fsx/$S3_PREFIX

echo "`date`: Copying files from s3://$S3_BUCKET/$S3_PREFIX [ eta 46 minutes ]"
sudo aws s3 cp --recursive s3://$S3_BUCKET/$S3_PREFIX $HOME/fsx/$S3_PREFIX | awk 'BEGIN {ORS="="} {if(NR%200==0)print "="}'
echo "Done."
