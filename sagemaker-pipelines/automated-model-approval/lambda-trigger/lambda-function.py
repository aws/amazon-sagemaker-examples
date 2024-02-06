import json
import os
import boto3

# Specify the name of your SageMaker pipeline
pipeline_name = os.environ['model-governance-pipeline-name']

def lambda_handler(event, context):
    # Initialize the SageMaker client
    sagemaker_client = boto3.client('sagemaker')
    

    print("Received event: " + json.dumps(event, indent=2))
    model_arn = event.get('detail', {}).get('ModelPackageArn', 'Unknown')
    model_package_group_name = event.get('detail', {}).get('ModelPackageGroupName', 'Unknown') 
    model_package_name = event.get('detail', {}).get('ModelPackageName', 'Unknown') 
    model_data_url = event.get('InferenceSpecification', {}).get('ModelDataUrl', 'Unknown')
        
    print("Model Package ARN:", model_arn)
    print("Model Package Group  Name:", model_package_group_name)
    print("Model Package Name:", model_package_name)
    print("Model Data URL:", model_data_url)
    
    # Define multiple parameters
    
    pipeline_parameters = [
        {'Name': "ModelPackageGroupName", 'Value': model_package_group_name},
        {'Name': "Bucket", 'Value': model_data_url},
   ]
    
    
    # Start the pipeline execution
    response = sagemaker_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=pipeline_name,
        PipelineParameters=pipeline_parameters
    )
    
    # Return the response
    return response