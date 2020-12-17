# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     https://aws.amazon.com/apache-2-0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import boto3
from datetime import datetime, timedelta
import re

client = boto3.client('sagemaker')
running_jobs = client.list_training_jobs(CreationTimeAfter=datetime.utcnow() - timedelta(hours=1))

logdir = None
for job in running_jobs['TrainingJobSummaries']:
    tensorboard_job = False
    name = None
    tags = client.list_tags(ResourceArn=job['TrainingJobArn'])
    for tag in tags['Tags']:
        if tag['Key'] == 'TensorBoard':
            name = tag['Value']
        if tag['Key'] == 'Project' and tag['Value'] == 'cifar10':
            desc = client.describe_training_job(TrainingJobName=job['TrainingJobName'])
            job_name = desc['HyperParameters']['sagemaker_job_name'].replace('"', '')
            tensorboard_dir = re.sub(
                'source/sourcedir.tar.gz', 'model', desc['HyperParameters']['sagemaker_submit_directory']
            )
            tensorboard_job = True

    if tensorboard_job:
        if name is None:
            name = job['TrainingJobName']

        if logdir is None:
            logdir = '{}:{}'.format(name, tensorboard_dir)
        else:
            logdir = '{},{}:{}'.format(logdir, name, tensorboard_dir)

if logdir:
    print('AWS_REGION={} tensorboard --logdir {}'.format(boto3.session.Session().region_name, logdir))
else:
    print('No jobs are in progress')
