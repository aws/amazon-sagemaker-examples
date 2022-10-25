# © 2021 Amazon Web Services, Inc. or its affiliates. All Rights Reserved.
#
# This AWS Content is provided subject to the terms of the AWS Customer Agreement
# available at http://aws.amazon.com/agreement or other written agreement between
# Customer and either Amazon Web Services, Inc. or Amazon Web Services EMEA SARL or both.

#!/bin/sh

REPO=$1
S3_BUCKET_NAME=$2

NAME=sourcedir
PUSH=true

if [ -z ${REPO} ] ;
then
    echo "Repository not specified"
    exit 1
fi

SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH/$REPO/src
tar --exclude='data' -czvf ${NAME}.tar.gz *

rm -rf ../dist/$REPO
mkdir ../dist
mkdir ../dist/$REPO

mv ${NAME}.tar.gz ../dist/$REPO

if [ -z ${S3_BUCKET_NAME} ] ;
then
  echo "S3 Bucket not specified, no upload"
  PUSH=false
fi

if $PUSH ;
then
  echo "Uploading s3://${S3_BUCKET_NAME}/artifact/${REPO}/${NAME}.tar.gz"

  aws s3 cp ../dist/${REPO}/${NAME}.tar.gz s3://${S3_BUCKET_NAME}/artifact/${REPO}/${NAME}.tar.gz
else
  exit 1
fi