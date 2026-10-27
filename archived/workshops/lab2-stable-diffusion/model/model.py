#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import logging
import os
import json
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf/cache'
os.environ["HF_HOME"] = '/tmp/hf'
PAGINATION_ENABLED = os.environ.get("PAGINATION", "false") == "true"


import torch
from diffusers import DDIMScheduler

if PAGINATION_ENABLED:
    from pipeline_stable_diffusion_pagination_ait import StableDiffusionPaginationAITPipeline as SDPipeline
else:
    from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline as SDPipeline
    
import safetensors as st


# import deepspeed
from djl_python.inputs import Input
from djl_python.outputs import Output
from typing import Optional
from io import BytesIO
import base64



def get_torch_dtype_from_str(dtype: str):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype is None:
        return None
    raise ValueError(f"Invalid data type: {dtype}")

def encode_images(images):
    encoded_images = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue())
        encoded_images.append(img_str.decode("utf8"))
    
    return encoded_images


class StableDiffusionService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.ds_config = None
        self.logger = logging.getLogger()
        self.model_id_or_path = None
        self.data_type = None
        self.device = None
        self.world_size = None
        self.max_tokens = None
        self.tensor_parallel_degree = None
        self.save_image_dir = None

    def initialize(self, properties: dict):
        # option.s3url is used to copy the AIT optimized weights from S3
        # option.pretrained_model_name is used to download the base model from HF Hub

        self.model_id_or_path = properties.get("model_id")
        self.base_model = properties.get("pretrained_model_name")
        
        os.environ["AITEMPLATE_WORK_DIR"] = self.model_id_or_path
        
        
        self.data_type = get_torch_dtype_from_str(properties.get("dtype"))
        self.max_tokens = int(properties.get("max_tokens", "1024"))
        self.device = int(os.getenv("LOCAL_RANK", "0"))

        kwargs = {}
        if self.data_type == torch.float16:
            kwargs["torch_dtype"] = torch.float16
            kwargs["revision"] = "fp16"

        pipeline = SDPipeline.from_pretrained(self.base_model, **kwargs)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        pipeline.to(f"cuda:{self.device}")


        self.pipeline = pipeline
        self.initialized = True


    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            if content_type == "application/json":
                request = inputs.get_as_json()
                prompt = request.pop("prompt")
                params = request.pop("parameters")
                
                if "latents" in params:
                    params["latents"] = st.torch.load(base64.b64decode(params["latents"].encode("utf8")))["latents"].to(f"cuda:{self.device}")
                
                if PAGINATION_ENABLED:
                    images, latents, step = self.pipeline(prompt, **params)
                else:
                    images = self.pipeline(prompt, **params).images
                
            else:
                raise Exception("unsupported content type. Use application/json")
                # prompt = inputs.get_as_string()
                # result = self.pipeline(prompt)
            
            
            encoded_images = encode_images(images)
            
            if PAGINATION_ENABLED:
                latents_st = st.torch.save({"latents":latents})
                b64_latents = base64.b64encode(latents_st).decode("utf8")
                response = dict(images=encoded_images, latents=b64_latents, step=step)
            else:
                response = dict(images=encoded_images)
            
            json_resp = json.dumps(response)
            
            
            outputs = Output().add(json_resp).add_property(
                "content-type", "application/json")

        except Exception as e:
            logging.exception("DeepSpeed inference failed")
            outputs = Output().error(str(e))
        return outputs


_service = StableDiffusionService()


def handle(inputs: Input) -> Optional[Output]:
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return _service.inference(inputs)