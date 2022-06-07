#!/usr/bin/env python
import os
import sys
import time
import argparse
from urllib.parse import urlparse
import pathlib

import boto3

comprehend = boto3.client("comprehend", region_name=os.environ["AWS_REGION"])

print(sys.argv)


def deploy(args):
    s3 = boto3.client("s3")
    arn = (
        s3.get_object(
            Bucket=urlparse(args.arn_path).netloc,
            Key=urlparse(f"{args.arn_path}/arn.txt").path[1:],
        )["Body"]
        .read()
        .decode()
        .strip()
    )

    endpoint_response = comprehend.create_endpoint(
        EndpointName=f'DEMO-classifier-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        ModelArn=arn,
        DesiredInferenceUnits=10,
    )

    endpoint_arn = endpoint_response["EndpointArn"]

    max_time = time.time() + 15 * 60  # 15 min
    while time.time() < max_time:
        describe_endpoint = comprehend.describe_endpoint(EndpointArn=endpoint_arn)
        status = describe_endpoint["EndpointProperties"]["Status"]

        if status == "IN_ERROR":
            sys.exit(1)

        if status == "IN_SERVICE":
            endpoint_arn_output_dir = "/opt/ml/processing/endpoint_arn"
            pathlib.Path(endpoint_arn_output_dir).mkdir(parents=True, exist_ok=True)

            print(f"Writing out endpoint arn {endpoint_arn}")
            endpoint_arn_path = f"{endpoint_arn_output_dir}/endpoint_arn.txt"
            with open(endpoint_arn_path, "w") as f:
                f.write(endpoint_arn)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arn-path", type=str, help="Path to the Arn on S3")
    args = parser.parse_args()
    print(args)

    deploy(args)
