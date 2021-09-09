#!/usr/bin/env python
import os
import sys
import json
import argparse
from urllib.parse import urlparse

import boto3

comprehend = boto3.client("comprehend", region_name=os.environ["AWS_REGION"])
s3 = boto3.client("s3")


def lambda_handler(event, context):
    
    endpoint_arn_path = event['endpoint_arn_path']
    text = event['text']
    
    endpoint_arn = s3.get_object(
        Bucket=urlparse(endpoint_arn_path).netloc,
        Key=urlparse(f"{endpoint_arn_path}/endpoint_arn.txt").path[1:]
    )["Body"].read().decode().strip()

    endpoint_response = comprehend.classify_document(
        Text = text,
        EndpointArn = endpoint_arn
    )
    
    return {
        "statusCode": 200,
        "body": json.dumps(endpoint_response)
    }