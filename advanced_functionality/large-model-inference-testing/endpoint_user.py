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
from inference import inference
from utils import fill_template, init_prompt_generator

class EndpointClient:
    def __init__(self, host):
        parse_output = urlparse(host)
        self.endpoint_name = parse_output.path.split('/')[2]
        self.content_type =os.getenv("CONTENT_TYPE", "application/json")
        aws_region = parse_output.netloc.split(".")[2]
        config = Config(region_name=aws_region, retries={"max_attempts": 0, "mode": "standard"})
        self.smr_client = boto3.client("sagemaker-runtime", config=config)

        self.streaming_enabled = os.getenv("STREAMING_ENABLED", "false").lower() in [ "true"]
        self.task_name = os.getenv("TASK_NAME", "text-generation")
        self.prompt_generator, self.prompt_template, self.prompt_template_keys = init_prompt_generator()
    
    def __text_generation_request(self, request_meta:dict):
        prompt = next(self.prompt_generator)
        data = fill_template(template=self.prompt_template, template_keys=self.prompt_template_keys, inputs=prompt)
        text, ttft = generate(self.smr_client, self.endpoint_name, data=data,stream=self.streaming_enabled)
        prompt = prompt[0] if len(prompt) == 1 else prompt
        index = text.find(prompt) if isinstance(prompt, str) else -1
        if index != -1:
            text = text[len(prompt):]
        if ttft is not None:
            request_meta['response'] = {"prompt": prompt, "text": text, "ttft": ttft}
        else:
            request_meta['response'] = {"prompt": prompt, "text": text}
      
    def __inference_request(self, request_meta:dict):
        prompt = next(self.prompt_generator)
        data = fill_template(template=self.prompt_template, template_keys=self.prompt_template_keys, inputs=prompt)
        result = inference(self.smr_client, self.endpoint_name, data=data, task=self.task_name)
        request_meta['response'] = {"prompt": data, "output": result}

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
            if self.task_name == "text-generation":
                self.__text_generation_request(request_meta)
            else:
                self.__inference_request(request_meta)
            
        except StopIteration as se:
            self.__init_prompt_generator()
            request_meta["exception"] = se
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