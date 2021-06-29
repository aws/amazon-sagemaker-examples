"""Example AWS Lambda function module for scheduling an exported Data Wrangler flow.
"""
import json
import time
import uuid

import boto3
import sagemaker
from sagemaker.dataset_definition.inputs import AthenaDatasetDefinition, DatasetDefinition
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import Processor


def lambda_handler(event, context):
    """ main handler function for lambda """

    job_info = run_flow()

    api_response = {
        'statusCode': 200,
        'event': event,
        'job_info': job_info
    }

    return api_response


def generate_query():
    """Generate a dynamic query based on datetime"""
    from datetime import datetime

    # Our sample data is from year 2008
    year = 2008
    curr_date = datetime.now()
    start_date = f"{year}{curr_date.month:02}00"
    end_date = f"{year}{(curr_date.month + 1):02}00"

    query_string = f"select * from inpatient_claim where clm_from_dt between {start_date} and {end_date}"
    return query_string


def run_flow():
    runtime_query_string = generate_query()
    print(runtime_query_string)

    data_sources = []
    # this is just a placeholder, please replace with your own bucket
    source_bucket = "<PLACEHOLDER-BUCKET>"

    data_sources.append(ProcessingInput(
        input_name="Inpatient_Claim",
        dataset_definition=DatasetDefinition(
            local_path="/opt/ml/processing/Inpatient_Claim",
            data_distribution_type="FullyReplicated",
            # You can override below to point to other database or use different queries
            athena_dataset_definition=AthenaDatasetDefinition(
                catalog="AwsDataCatalog",
                database="cms",
                query_string=runtime_query_string,
                output_s3_uri=f"s3://{source_bucket}/athena/Inpatient_Claim/",
                output_format="PARQUET"
            )
        )
    ))

    data_sources.append(ProcessingInput(
        source=f"s3://{source_bucket}/DW/DE1_0_2008_Beneficiary_Summary_File_Sample_20.csv",
        # You can override this to point to other dataset on S3
        destination="/opt/ml/processing/2008_Beneficiary_Summary",
        input_name="2008_Beneficiary_Summary",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated"
    ))

    # Sagemaker session
    sess = sagemaker.Session()
    print(f"Data Wrangler sagemaker session: {sess}")

    # You can configure this with your own bucket name, e.g.
    # bucket = <my-own-storage-bucket>
    bucket = sess.default_bucket()
    print(f"Data Wrangler export storage bucket: {bucket}")

    # unique flow export ID
    flow_export_id = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())}-{str(uuid.uuid4())[:8]}"
    flow_export_name = f"flow-{flow_export_id}"

    # Output name is auto-generated from the select node's ID + output name from the flow file.
    output_name = "8b392709-d2c4-4b8e-bdda-e75b2d14f35e.default"

    s3_output_prefix = f"export-{flow_export_name}/output"
    s3_output_path = f"s3://{bucket}/{s3_output_prefix}"
    print(f"Flow S3 export result path: {s3_output_path}")

    s3_processing_output = ProcessingOutput(
        output_name=output_name,
        source="/opt/ml/processing/output",
        destination=s3_output_path,
        s3_upload_mode="EndOfJob"
    )

    # name of the flow file which should exist in the current notebook working directory
    flow_file_name = "cms.flow"

    with open(flow_file_name) as f:
        flow = json.load(f)

    # Upload flow to S3
    s3_client = boto3.client("s3")
    s3_client.upload_file(flow_file_name, bucket, f"data_wrangler_flows/{flow_export_name}.flow")

    flow_s3_uri = f"s3://{bucket}/data_wrangler_flows/{flow_export_name}.flow"

    print(f"Data Wrangler flow {flow_file_name} uploaded to {flow_s3_uri}")

    # Input - Flow: cms.flow
    flow_input = ProcessingInput(
        source=flow_s3_uri,
        destination="/opt/ml/processing/flow",
        input_name="flow",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated"
    )

    print(f"ProcessingInput defined")

    # IAM role for executing the processing job. You can set this up with your own role
    iam_role = "<IAM-ROLE-PLACEHOLDER>"

    # Unique processing job name. Please give a unique name every time you re-execute processing jobs
    processing_job_name = f"data-wrangler-flow-processing-{flow_export_id}"

    # Data Wrangler Container URL.
    container_uri = "663277389841.dkr.ecr.us-east-1.amazonaws.com/sagemaker-data-wrangler-container:1.x"

    # Processing Job Instance count and instance type.
    instance_count = 2
    instance_type = "ml.m5.4xlarge"

    # Size in GB of the EBS volume to use for storing data during processing
    volume_size_in_gb = 30

    # Content type for each output. Data Wrangler supports CSV as default and Parquet.
    output_content_type = "CSV"

    # Network Isolation mode; default is off
    enable_network_isolation = False

    # Output configuration used as processing job container arguments 
    output_config = {
        output_name: {
            "content_type": output_content_type
        }
    }

    processor = Processor(
        role=iam_role,
        image_uri=container_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        network_config=NetworkConfig(enable_network_isolation=enable_network_isolation),
        sagemaker_session=sess
    )
    print(f"Processor defined")

    # Start Job

    processor.run(
        inputs=[flow_input] + data_sources,
        outputs=[s3_processing_output],
        arguments=[f"--output-config '{json.dumps(output_config)}'"],
        wait=False,
        logs=False,
        job_name=processing_job_name
    )
    s3_job_results_path = f"s3://{bucket}/{s3_output_prefix}/{processing_job_name}"
    print(f"Job results are saved to S3 path: {s3_job_results_path}")

    job_result = sess.wait_for_processing_job(processing_job_name)
    return job_result
