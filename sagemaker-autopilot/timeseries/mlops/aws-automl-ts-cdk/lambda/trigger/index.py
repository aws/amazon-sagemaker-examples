import os
import boto3
from json import loads, dumps
from datetime import datetime, timedelta

sfn = boto3.client('stepfunctions')
s3 = boto3. client('s3')

def handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']

    ts = datetime.now() # note TZ is UTC
    ts = ts.strftime("%Y%m%dT%H%M%S")

    return dumps(
        sfn.start_execution(
            stateMachineArn=os.environ['STEP_FUNCTIONS_ARN'],
            name=ts,
            input=dumps(
                {
                    'uid': ts
                }
            )
        ),
        default=str
    )
