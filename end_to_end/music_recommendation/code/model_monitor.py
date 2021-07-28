import boto3
import argparse
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])
import sagemaker
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.predictor import Predictor
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker import get_execution_role
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker import session
from time import gmtime, strftime


# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument('--baseline-data-uri', type=str)
parser.add_argument('--bucket-name', type=str)
parser.add_argument('--bucket-prefix', type=str)
parser.add_argument('--endpoint', type=str)
parser.add_argument('--region', type=str),
parser.add_argument('--schedule-name', type=str)
args = parser.parse_args()

bucket = args.bucket_name
prefix = args.bucket_prefix
endpoint_name = args.endpoint
region = args.region
mon_schedule_name_base = args.schedule_name

# create a sagemaker session with user's region 
boto_session = boto3.Session(region_name=region)
sagemaker_boto_client = boto_session.client('sagemaker')
sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_boto_client
)

# Enable real-time inference data capture

s3_capture_upload_path = f's3://{bucket}/{prefix}/endpoint-data-capture/' #example: s3://bucket-name/path/to/endpoint-data-capture/

# Change parameters as you would like - adjust sampling percentage, 
#  chose to capture request or response or both
data_capture_config = DataCaptureConfig(
    enable_capture = True,
    sampling_percentage=25,
    destination_s3_uri=s3_capture_upload_path,
    kms_key_id=None,
    capture_options=["REQUEST", "RESPONSE"],
    csv_content_types=["text/csv"],
    json_content_types=["application/json"]
)

# Now it is time to apply the new configuration
predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)
predictor.update_data_capture_config(data_capture_config=data_capture_config)

print('Created Predictor at endpoint {}'.format(endpoint_name))

baseline_data_uri = args.baseline_data_uri ##'s3://bucketname/path/to/baseline/data' - Where your validation data is
baseline_results_uri = f's3://{bucket}/{prefix}/baseline/results' ##'s3://bucketname/path/to/baseline/data' - Where the results are to be stored in

print('Baseline data is at {}'.format(baseline_data_uri))

my_default_monitor = DefaultModelMonitor(
    role=get_execution_role(sagemaker_session=sagemaker_session),
    sagemaker_session=sagemaker_session,
    instance_count=2,
    instance_type='ml.m5.4xlarge',
    volume_size_in_gb=60,
    max_runtime_in_seconds=1800,
)


my_default_monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri,
    dataset_format=DatasetFormat.csv(header=False),
    output_s3_uri=baseline_results_uri,
    wait=True
)

print('Model data baseline suggested at {}'.format(baseline_results_uri))

import datetime as datetime
from time import gmtime, strftime

mon_schedule_name = '{}-{}'.format(mon_schedule_name_base, datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))

s3_report_path = f's3://{bucket}/{prefix}/monitor/report'

# Setup daily Cron job schedule 
print(f"Attempting to create monitoring schedule as {mon_schedule_name} \n")

try:
    my_default_monitor.create_monitoring_schedule(
        monitor_schedule_name=mon_schedule_name,
        endpoint_input=endpoint_name,
        output_s3_uri=s3_report_path,
        statistics=my_default_monitor.baseline_statistics(),
        constraints=my_default_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=True,
    )
    desc_schedule_result = my_default_monitor.describe_schedule()
    print('Created monitoring schedule. Schedule status: {}'.format(desc_schedule_result['MonitoringScheduleStatus']))
    
except:
    my_default_monitor.update_monitoring_schedule(
        endpoint_input=endpoint_name,
        schedule_cron_expression=CronExpressionGenerator.daily()
    )
    print("Monitoring schedule already exists for endpoint. Updating schedule.")