import json

import boto3
import sagemaker


def update_dw_s3uri(flow_file_name):
    """
    Update the input S3 locations in a Data Wrangler (.flow) file with the default bucket
    """
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    prefix = "data_wrangler_flows"

    # upload data to S3
    s3_client = boto3.client("s3", region_name=boto3.Session().region_name)

    fleet_info_filename = "example_fleet_info.csv"
    sensor_logs_filename = "example_fleet_sensor_logs.csv"

    s3_client.upload_file(
        Filename=f"data/{fleet_info_filename}",
        Bucket=bucket,
        Key=f"{prefix}/data/{fleet_info_filename}",
    )
    s3_client.upload_file(
        Filename=f"data/{sensor_logs_filename}",
        Bucket=bucket,
        Key=f"{prefix}/data/{sensor_logs_filename}",
    )

    fleet_info_uri = f"s3://{bucket}/{prefix}/data/{fleet_info_filename}"
    sensor_logs_uri = f"s3://{bucket}/{prefix}/data/{sensor_logs_filename}"

    # read flow file and change the s3 location to our `processing_output_filename`
    with open(flow_file_name, "r") as f:
        flow = f.read()
        flow = json.loads(flow)

    # replace old s3 locations with our personal s3 location
    new_nodes = []
    for node in flow["nodes"]:
        if node["type"] == "SOURCE":
            if node["parameters"]["dataset_definition"]["name"] == fleet_info_filename:
                node["parameters"]["dataset_definition"]["s3ExecutionContext"][
                    "s3Uri"
                ] = fleet_info_uri
            elif node["parameters"]["dataset_definition"]["name"] == sensor_logs_filename:
                node["parameters"]["dataset_definition"]["s3ExecutionContext"][
                    "s3Uri"
                ] = sensor_logs_uri
        new_nodes.append(node)

    flow["nodes"] = new_nodes

    with open(flow_file_name, "w") as f:
        json.dump(flow, f)


dw_container_dict = {
    "us-east-2": "415577184552.dkr.ecr.us-east-2.amazonaws.com/sagemaker-data-wrangler-container:1.0.1"
}


def get_dw_container_for_region(region_in):
    """
    Get the Data Wrangler container based on the given region
    """
    container_uri = dw_container_dict[region_in]
    return container_uri
