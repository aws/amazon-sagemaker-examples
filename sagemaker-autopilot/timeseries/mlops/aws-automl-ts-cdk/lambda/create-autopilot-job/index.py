import os
import boto3
import json
from datetime import datetime, timedelta

sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')


def handler(event, context):
    print(event)
    
    sagemaker_role = os.environ['SAGEMAKER_ROLE_ARN']
    resource_bucket = os.environ['RESOURCE_BUCKET']
    
    ts = datetime.now() # note TZ is UTC
    ts = ts.strftime("%Y%m%dT%H%M%S")
    job_name = 'automl-job-'+ts  #os.environ['JOB_NAME']
    
    csv_s3_uri = f's3://{resource_bucket}/input/training_data.csv'
    automl_config_file_key = 'config/automl_problem_config.json'
    
    
    # Define input data config for SageMaker
    input_data_config = [
        {
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': csv_s3_uri
                }
            }
        }
    ]

    # Define output data config for SageMaker
    output_data_config = {
        'S3OutputPath': f's3://{resource_bucket}/autopilot-output/'
    }
    
    # Get AutoML Problem Dynamic Config file from S3
    automl_problem_config_s3 = s3.get_object(Bucket=resource_bucket, Key=automl_config_file_key)
    
    automl_problem_config_body = automl_problem_config_s3['Body'].read().decode('utf-8')
    
    automl_problem_config = json.loads(automl_problem_config_body)
    
    # Create the AutoML job
    response = sagemaker.create_auto_ml_job_v2(
        AutoMLJobName=job_name,
        AutoMLJobInputDataConfig=input_data_config,
        OutputDataConfig=output_data_config,
        RoleArn=sagemaker_role,
        AutoMLProblemTypeConfig=automl_problem_config
    )

    return {
        'AutoMLJobResponse': response,
        'AutoMLJobName': job_name
    }