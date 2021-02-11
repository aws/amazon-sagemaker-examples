import time
import boto3
import argparse


# Parse argument variables passed via the DeployModel processing step
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str)
parser.add_argument('--region', type=str)
parser.add_argument('--endpoint-instance-type', type=str)
parser.add_argument('--endpoint-name', type=str)
args = parser.parse_args()

region = args.region
boto3.setup_default_session(region_name=region)
sagemaker_boto_client = boto3.client('sagemaker')

#name truncated per sagameker length requirememnts (63 char max)
endpoint_config_name=f'{args.model_name[:56]}-config'
existing_configs = sagemaker_boto_client.list_endpoint_configs(NameContains=endpoint_config_name)['EndpointConfigs']

if not existing_configs:
    create_ep_config_response = sagemaker_boto_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': args.endpoint_instance_type,
            'InitialVariantWeight': 1,
            'InitialInstanceCount': 1,
            'ModelName': args.model_name,
            'VariantName': 'AllTraffic'
        }]
    )

existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains=args.endpoint_name)['Endpoints']

if not existing_endpoints:
    create_endpoint_response = sagemaker_boto_client.create_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=endpoint_config_name)

endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
endpoint_status = endpoint_info['EndpointStatus']

while endpoint_status == 'Creating':
    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']
    print('Endpoint status:', endpoint_status)
    if endpoint_status == 'Creating':
        time.sleep(60)