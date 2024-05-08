from importlib import import_module
import time
import boto3
import os
import json
import sys

from botocore.config import Config
from urllib.parse import urlparse
from locust.contrib.fasthttp import FastHttpUser
from locust import task, events
from generate import generate



class EndpointClient:
    def __init__(self, host):
        
        parse_output = urlparse(host)
        self.endpoint_name = parse_output.path.split('/')[2]
        self.content_type =os.getenv("CONTENT_TYPE", "application/json")
        aws_region = parse_output.netloc.split(".")[2]
        config = Config(region_name=aws_region, retries={"max_attempts": 0, "mode": "standard"})
        self.smr_client = boto3.client("sagemaker-runtime", config=config)

        prompt_module_dir = os.getenv("PROMPT_MODULE_DIR", "")
        sys.path.append(prompt_module_dir)
        
        prompt_module_name = os.getenv("PROMPT_MODULE_NAME", None)
        prompt_module=import_module(prompt_module_name)
        
        prompt_generator_name = os.getenv('PROMPT_GENERATOR_NAME', None)
        prompt_generator_class = getattr(prompt_module, prompt_generator_name)
        
        self.prompt_generator = prompt_generator_class()()

        self.params = json.loads(os.getenv("MODEL_PARAMS", "{}"))
        self.streaming_enabled = os.getenv("STREAMING_ENABLED", "false").lower() in [ "true"]
        
       

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
            prompt = next(self.prompt_generator)
            text, ttft = generate(self.smr_client, self.endpoint_name, 
                                  prompt=next(self.prompt_generator), 
                                  params=self.params, 
                                  stream=self.streaming_enabled)
            if ttft is not None:
                request_meta['response'] = {"prompt": prompt, "text": text, "ttft": ttft}
            else:
                request_meta['response'] = {"prompt": prompt, "text": text}
        except Exception as e:
            request_meta["exception"] = e

        request_meta["response_time"] = (
            time.perf_counter() - start_perf_counter
        ) * 1000

        events.request.fire(**request_meta)


class EndpointUser(FastHttpUser):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.client = EndpointClient(self.host)


class SageMakerEndpointUser(EndpointUser):
    @task
    def send_request(self):
        self.client.send()