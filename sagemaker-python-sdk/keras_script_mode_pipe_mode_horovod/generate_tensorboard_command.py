import boto3
from datetime import datetime,timedelta
import re
client = boto3.client('sagemaker')
running_jobs = client.list_training_jobs(CreationTimeAfter=datetime.utcnow() - timedelta(hours=1))

logdir = None
for job in running_jobs['TrainingJobSummaries']:
    tensorboardjob = False
    tags = client.list_tags(
        ResourceArn=job['TrainingJobArn']
    )
    name = None
    for tag in tags['Tags']:
        if tag['Key'] == 'TensorBoard':
            name = tag['Value']
        if tag['Key'] == 'Project' and tag['Value'] == 'cifar10':
            desc = client.describe_training_job(TrainingJobName=job['TrainingJobName'])
            jobName = desc['HyperParameters']['sagemaker_job_name'].replace('"', '')
            tensorboardDir = re.sub('source/sourcedir.tar.gz', 'model',desc['HyperParameters']['sagemaker_submit_directory'])
            tensorboardjob = True
    
    if tensorboardjob:
        if name is None:
            name = job['TrainingJobName']
        if logdir is None:
            logdir = name+":"+ tensorboardDir
        else:
            logdir = logdir + ","+name+":" + tensorboardDir
            
if logdir:
    print("tensorboard --logdir " + logdir)
else:
    print("No jobs are in progress")