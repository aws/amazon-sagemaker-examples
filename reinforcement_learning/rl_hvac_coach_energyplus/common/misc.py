# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from __future__ import absolute_import

import base64
import contextlib
import os
import time
import shlex
import shutil
import subprocess
import sys
import tempfile

import boto3
import json

    
def wait_for_s3_object(s3_bucket, key, local_dir, local_prefix='', 
                       aws_account=None, aws_region=None, timeout=1200, limit=20,
                       fetch_only=None, training_job_name=None):
    """
    Keep polling s3 object until it is generated.
    Pulling down latest data to local directory with short key

    Arguments:
        s3_bucket (string): s3 bucket name
        key (string): key for s3 object
        local_dir (string): local directory path to save s3 object
        local_prefix (string): local prefix path append to the local directory
        aws_account (string): aws account of the s3 bucket
        aws_region (string): aws region where the repo is located
        timeout (int): how long to wait for the object to appear before giving up
        limit (int): maximum number of files to download
        fetch_only (lambda): a function to decide if this object should be fetched or not
        training_job_name (string): training job name to query job status

    Returns:
        A list of all downloaded files, as local filenames
    """
    session = boto3.Session()
    aws_account = aws_account or session.client("sts").get_caller_identity()['Account']
    aws_region = aws_region or session.region_name

    s3 = session.resource('s3')
    sagemaker = session.client('sagemaker')
    bucket = s3.Bucket(s3_bucket)
    objects = []

    print("Waiting for s3://%s/%s..." % (s3_bucket, key), end='', flush=True)
    start_time = time.time()
    cnt = 0
    while len(objects) == 0:
        objects = list(bucket.objects.filter(Prefix=key))
        if fetch_only:
            objects = list(filter(fetch_only, objects))
        if objects:
            continue
        print('.', end='', flush=True)
        time.sleep(5)
        cnt += 1
        if cnt % 80 == 0:
            print("")
        if time.time() > start_time + timeout:
            raise FileNotFoundError("S3 object s3://%s/%s never appeared after %d seconds" % (s3_bucket, key, timeout))
        if training_job_name:
            training_job_status = sagemaker.describe_training_job(TrainingJobName=training_job_name)['TrainingJobStatus']
            if training_job_status == 'Failed':
                raise RuntimeError("Training job {} failed while waiting for S3 object s3://{}/{}"
                                   .format(training_job_name, s3_bucket, key))

    print('\n', end='', flush=True)

    if len(objects) > limit:
        print("Only downloading %d of %d files" % (limit, len(objects)))
        objects = objects[-limit:]

    fetched_files = []
    for obj in objects:
        print("Downloading %s" % obj.key)
        local_path = os.path.join(local_dir, local_prefix, obj.key.split('/')[-1])
        obj.Object().download_file(local_path)
        fetched_files.append(local_path)

    return fetched_files


def get_execution_role(role_name="sagemaker", aws_account=None, aws_region=None):
    """
    Create sagemaker execution role to perform sagemaker task

    Args:
        role_name (string): name of the role to be created
        aws_account (string): aws account of the ECR repo
        aws_region (string): aws region where the repo is located
    """
    session = boto3.Session()
    aws_account = aws_account or session.client("sts").get_caller_identity()['Account']
    aws_region = aws_region or session.region_name

    assume_role_policy_document = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": ["sagemaker.amazonaws.com", "robomaker.amazonaws.com"]
                },
                "Action": "sts:AssumeRole"
            }
        ]
    })

    client = session.client('iam')
    try:
        client.get_role(RoleName=role_name)
    except client.exceptions.NoSuchEntityException:
        client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=str(assume_role_policy_document)
        )

        print("Created new sagemaker execution role: %s" % role_name)

    client.attach_role_policy(
        PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
        RoleName=role_name
    )

    return client.get_role(RoleName=role_name)['Role']['Arn']


