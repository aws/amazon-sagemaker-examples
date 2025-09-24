# utility script.

import boto3
import os
from pathlib import Path
import tarfile
import shutil
import sagemaker
import json

# Use the function with your model artifacts path
# model_artifacts = "s3://sagemaker-us-east-1-891376962744/meta-textgeneration-llama-3-8b-2025-03-07-19-04-01-960/output/model/"
def download_model_from_s3(s3_uri, local_dir='downloaded_model'):
    # Parse the S3 URI
    bucket_name = s3_uri.split('/')[2]
    # Get the prefix (everything after bucket name)
    prefix = '/'.join(s3_uri.split('/')[3:])
    
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # List all objects in the prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    objects = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    # Download each object
    for page in objects:
        if 'Contents' in page:
            for obj in page['Contents']:
                # Get the relative path of the file
                relative_path = obj['Key'][len(prefix):].lstrip('/')
                if relative_path:
                    # Create subdirectories if needed
                    local_file_path = os.path.join(local_dir, relative_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    print(f"Downloading: {obj['Key']} to {local_file_path}")
                    s3_client.download_file(
                        bucket_name,
                        obj['Key'],
                        local_file_path
                    )


# Create the model.tar.gz
def package_llama_model(model_dir, output_path='model.tar.gz'):
    """
    Package Llama model files into the required format for SageMaker deployment
    """
    # Create a temporary directory
    temp_dir = 'temp_model_dir'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Required files for Llama
    required_files = [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'generation_config.json'
    ]
    
    # Copy all required files
    for file in os.listdir(model_dir):
        if file in required_files or file.endswith('.safetensors'):
            shutil.copy2(
                os.path.join(model_dir, file),
                os.path.join(temp_dir, file)
            )
    
    # Create tar.gz archive
    with tarfile.open(output_path, 'w:gz') as tar:
        tar.add(temp_dir, arcname='')
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    # Verify the contents
    print("Verifying archive contents:")
    with tarfile.open(output_path, 'r:gz') as tar:
        for member in tar.getmembers():
            print(f"- {member.name}")



def upload_model_to_s3(s3_uri: str, local_model_path: str = "model.tar.gz") -> str:
    """
    Upload a model.tar.gz file to S3 using SageMaker Session.
    
    Args:
        s3_uri (str): Full S3 URI where the model will be uploaded (e.g., 's3://bucket-name/prefix/path/')
        local_model_path (str, optional): Local path to the model.tar.gz file. Defaults to "model.tar.gz"
        
    Returns:
        str: S3 URI where the model was uploaded
        
    Raises:
        ValueError: If S3 URI is invalid
        Exception: If upload fails or if SageMaker session cannot be created
    """
    try:
        # Parse S3 URI to get bucket and prefix
        if not s3_uri.startswith('s3://'):
            raise ValueError("Invalid S3 URI. Must start with 's3://'")
        
        # Remove 's3://' and split into bucket and prefix
        path_parts = s3_uri.replace('s3://', '').strip('/').split('/', 1)
        
        if len(path_parts) < 1:
            raise ValueError("Invalid S3 URI. Must contain bucket name")
            
        bucket = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        # Create SageMaker session
        session = sagemaker.Session()
        
        # Upload using SageMaker Session
        s3_uri = session.upload_data(
            path=local_model_path,
            bucket=bucket,
            key_prefix=prefix
        )
        
        print(f"Model successfully uploaded to: {s3_uri}")
        return s3_uri
        
    except Exception as e:
        print(f"Error uploading model to S3: {str(e)}")
        raise


def get_hub_content_document(
    hub_name="SageMakerPublicHub",
    hub_content_name="meta-textgeneration-llama-3-8b",
    region="us-east-1"
):
    # Create a SageMaker client
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        # Call describe_hub_content API
        response = sagemaker_client.describe_hub_content(
            HubName=hub_name,
            HubContentType='Model',
            HubContentName=hub_content_name
        )
        
        # Extract HubContentDocument
        hub_content_document = response['HubContentDocument']
        
        # Print formatted JSON
        # print(json.dumps(hub_content_document, indent=2))
        
        return hub_content_document
        
    except Exception as e:
        print(f"Error retrieving hub content: {str(e)}")
        return None


def update_hosting_uris(dictionary, new_path):
    """
    Recursively update HostingScriptUri and HostingArtifactUri in hub_content_document
    """
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if key in ['HostingScriptUri', 'HostingArtifactUri']:
                dictionary[key] = new_path
            elif isinstance(value, (dict, list)):
                update_hosting_uris(value, new_path)
    elif isinstance(dictionary, list):
        for item in dictionary:
            update_hosting_uris(item, new_path)