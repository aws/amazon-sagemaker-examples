import time
from datetime import datetime
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

# truncate name per sagameker length requirememnts (63 char max) if necessary
endpoint_config_name = f'{args.endpoint_name}-config-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

# create new endpoint config file 
create_ep_config_response = sagemaker_boto_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        'InstanceType': args.endpoint_instance_type,
        'InitialVariantWeight': 1,
        'InitialInstanceCount': 1,
        'ModelName': args.model_name,
        'VariantName': 'AllTraffic'
        }])

print("ModelName: {}".format(args.model_name))

# create endpoint if model endpoint does not already exist, otherwise update the endpoint
try:
    create_endpoint_response = sagemaker_boto_client.create_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

except:
    # delete existing monitoring schedule before updating endpoint
    schedules = sagemaker_boto_client.list_monitoring_schedules(EndpointName=args.endpoint_name)['MonitoringScheduleSummaries']
    while len(schedules) > 0:
        for schedule in schedules:
            # can only delete schedules in states 'Scheduled', 'Failed', and 'Stopped'
            if schedule['MonitoringScheduleStatus'] == 'Pending':
                continue
            else:
                sagemaker_boto_client.delete_monitoring_schedule(MonitoringScheduleName=schedule['MonitoringScheduleName'])
        time.sleep(5)
        schedules = sagemaker_boto_client.list_monitoring_schedules(EndpointName=args.endpoint_name)['MonitoringScheduleSummaries']
    print("Updating endpoint {} and deleting existing monitoring schedule".format(args.endpoint_name))
    create_endpoint_response = sagemaker_boto_client.update_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
endpoint_status = endpoint_info['EndpointStatus']

while endpoint_status != 'InService':
    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']
    print('Endpoint status:', endpoint_status)
    if endpoint_status != 'InService':
        time.sleep(60)