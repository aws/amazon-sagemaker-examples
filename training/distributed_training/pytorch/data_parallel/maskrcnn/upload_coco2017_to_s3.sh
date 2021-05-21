#!/bin/bash
# credit of https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/distributed_tensorflow_mask_rcnn/prepare-s3-bucket.sh

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <s3-bucket-name> <s3-bucket-prefix>" # TODO: fix usage message to reflect that bucket prefix is also required
    exit 1
fi

S3_BUCKET=$1
S3_PREFIX=$2

# Stage directory must be on EBS volume with 100 GB available space
STAGE_DIR=$HOME/SageMaker/coco-2017-$(date +"%Y-%m-%d-%H-%M-%S")

echo "Create stage directory: $STAGE_DIR"
mkdir -p $STAGE_DIR

wget -O $STAGE_DIR/train2017.zip http://images.cocodataset.org/zips/train2017.zip
echo "Extracting $STAGE_DIR/train2017.zip"
unzip -o $STAGE_DIR/train2017.zip  -d $STAGE_DIR | awk 'BEGIN {ORS="="} {if(NR%1000==0)print "="}'
echo "Done."
rm $STAGE_DIR/train2017.zip

wget -O $STAGE_DIR/val2017.zip http://images.cocodataset.org/zips/val2017.zip
echo "Extracting $STAGE_DIR/val2017.zip"
unzip -o $STAGE_DIR/val2017.zip -d $STAGE_DIR | awk 'BEGIN {ORS="="} {if(NR%1000==0)print "="}'
echo "Done."
rm $STAGE_DIR/val2017.zip

wget -O $STAGE_DIR/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o $STAGE_DIR/annotations_trainval2017.zip -d $STAGE_DIR
rm $STAGE_DIR/annotations_trainval2017.zip

echo "`date`: Uploading extracted files to s3://$S3_BUCKET/$S3_PREFIX [ eta 12 minutes ]"
aws s3 cp --recursive $STAGE_DIR s3://$S3_BUCKET/$S3_PREFIX | awk 'BEGIN {ORS="="} {if(NR%100==0)print "="}'
echo "Done."

echo "Delete stage directory: $STAGE_DIR"
rm -rf $STAGE_DIR
echo "Success."