import json
import logging
import os
import time

import boto3
from botocore.config import Config
from locust.contrib.fasthttp import FastHttpUser

from locust import task, events

region = os.environ["REGION"]
content_type = os.environ["CONTENT_TYPE"]
# payload = os.environ["PAYLOAD"]

with open("payload.csv", 'r') as f:
    payload = f.read()


class BotoClient:
    def __init__(self, host):
        # Consider removing retry logic to get accurate picture of failure in locust
        config = Config(
            region_name=region, retries={"max_attempts": 0, "mode": "standard"}
        )

        self.sagemaker_client = boto3.client("sagemaker-runtime", config=config)
        self.endpoint_name = host.split("/")[-1]
        self.content_type = content_type
        self.payload = payload

    def send(self):

        request_meta = {
            "request_type": "InvokeEndpoint",
            "name": "SageMaker",
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()

        try:
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=str(self.payload),
                ContentType=self.content_type,
            )
            logging.info(response["Body"].read())
        except Exception as e:
            request_meta["exception"] = e

        request_meta["response_time"] = (
            time.perf_counter() - start_perf_counter
        ) * 1000

        events.request.fire(**request_meta)


class BotoUser(FastHttpUser):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = BotoClient(self.host)


class MyUser(BotoUser):
    @task
    def send_request(self):
        self.client.send()