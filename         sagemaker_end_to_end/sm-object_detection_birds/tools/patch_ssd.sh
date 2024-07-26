#!/bin/bash
#  ./patch_ssd.sh s3://my-bucket/my-model-output 5 s3://my-bucket/my-patched-output

# This script requires you to have already cloned the incubator-mxnet project
# to obtain the deploy.py script
#git clone https://github.com/apache/incubator-mxnet

BUCKET_AND_PREFIX=$1 # should enforce that target must contain 'deeplens', else import fails
NUM_CLASSES=$2       # could be extracted from hyperparams.json for more robust implementation
TARGET_PREFIX=$3

# download and unpack the model artifacts
rm -rf tmp && mkdir tmp
aws s3 cp $BUCKET_AND_PREFIX/model.tar.gz tmp
gunzip -k -c tmp/model.tar.gz | tar -C tmp -xopf -
ls -l tmp/*

# copy your parameters file and symbol file into the SSD model directory
mv tmp/*-0000.params tmp/ssd_resnet50_512-0000.params
mv tmp/*-symbol.json tmp/ssd_resnet50_512-symbol.json

# run the deploy Python script, which creates patched versions of
# your parameters and symbol file with a new prefix of "deploy"
python incubator-mxnet/example/ssd/deploy.py --network resnet50 \
  --data-shape 512 --num-class $NUM_CLASSES --prefix tmp/ssd_

# now re-package your updated model artifacts
rm tmp/ssd_*  &&  rm tmp/model.tar.gz
tar -cvzf ./patched_model.tar.gz -C tmp \
  ./deploy_ssd_resnet50_512-0000.params \
  ./deploy_ssd_resnet50_512-symbol.json \
  ./hyperparams.json

# move the new artifacts to S3 for DeepLens model creation. Use a new
# prefix to keep it distinct from the original model artifacts.
aws s3 cp patched_model.tar.gz $TARGET_PREFIX/

# clean up
rm -rf tmp
rm *.gz
