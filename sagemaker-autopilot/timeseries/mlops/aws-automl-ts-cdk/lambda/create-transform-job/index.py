import os
import boto3
import json
from datetime import datetime, timedelta

sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')


def handler(event, context):
    print(event)
    
    resource_bucket = os.environ['RESOURCE_BUCKET']
    transform_model = event['BestCandidate']['CandidateName']
    transform_job_name = transform_model + '-job'
    transform_job_config_file_key = 'config/batch_transform_job_config.json'
    
    # Get Transform Job Config file from S3
    transform_job_config_s3 = s3.get_object(Bucket=resource_bucket, Key=transform_job_config_file_key)
    
    transform_job_config_body = transform_job_config_s3['Body'].read().decode('utf-8')
    
    transform_job_config = json.loads(transform_job_config_body)
    
    # Configure the Transform Job variables from Config file
    # Input Location - key to the specific file
    input_location_key = transform_job_config['InputLocationKey']
    input_location = f's3://{resource_bucket}/{input_location_key}'
    
    # Output location - prefix only
    output_location_prefix = transform_job_config['OutputLocationPrefix']
    output_location = f's3://{resource_bucket}/{output_location_prefix}/'
    
    instance_type = transform_job_config['InstanceType']
    instance_count = transform_job_config['InstanceCount']
    max_payload_mb = transform_job_config['MaxPayloadInMB']
    
    # Start the SageMaker Batch Transform Job
    sm_response = sagemaker.create_transform_job(
        TransformJobName=transform_job_name, 
        ModelName=transform_model,
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_location
                }
            },
            'ContentType': 'text/csv' 
        },
        TransformOutput={
            'S3OutputPath': output_location,
            'Accept': 'text/csv'
        },
        TransformResources={
            'InstanceType': instance_type,
            'InstanceCount': instance_count
        },
        MaxPayloadInMB=max_payload_mb
    )
    
    response = {
        'response': sm_response,
        'TransformJobName': transform_job_name,
        'ModelName': transform_model,
        'InputLocation': input_location,
        'OutputLocation': output_location
    }
    
    return response
    