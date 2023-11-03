import boto3
import os

s3 = boto3.client('s3')

def handler(event, context):
    
    resource_bucket = os.environ['RESOURCE_BUCKET']

    files_to_check = [
        'config/automl_problem_config.json',
        'config/batch_transform_job_config.json'
    ]

    for file_key in files_to_check:
        try:
            s3.head_object(Bucket=resource_bucket, Key=file_key)
        except:
            return {
                'config_status': 'FAILED',
                'message': f'File {file_key} does not exist.'
            }
    
    # Check for .zip or .csv file in the S3 bucket with prefix raw/
    response = s3.list_objects_v2(
        Bucket=resource_bucket,
        Prefix='raw/',
        MaxKeys=100
    )
    
    # Iterate over the returned objects and look for files with the desired extensions
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.zip') or obj['Key'].endswith('.csv'):
                return {
                    'config_status': 'SUCCEEDED'
                }
        return {
            'config_status': 'FAILED',
            'message': 'No .zip or .csv files found with the raw/ prefix.'
        }
    else:
        return {
            'config_status': 'FAILED',
            'message': 'No .zip or .csv files found with the raw/ prefix.'
        }
