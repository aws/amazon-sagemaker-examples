import json

import logging

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'


def model_fn(model_dir, context):
    logger.debug("model_fn: Creating model")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(pretrained_model_name_or_path=model_dir,
                                                                  torch_dtype=torch.float16)
    # distribute instantiated models among different gpus
    gpu_id = str(context.system_properties.get("gpu_id"))
    logger.debug("gpu_id:" + gpu_id)
    pipe.to("cuda:" + gpu_id)
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    logger.debug("model_fn: Model created and served via GPU: " + gpu_id)
    return pipe


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)


def predict_fn(input_data, model):
    prompt = input_data['inputs']['prompt']
    base64_image_string = input_data['inputs']['image']
    f = BytesIO(base64.b64decode(base64_image_string))
    img = Image.open(f)
    logger.debug('predict_fn: Got input prompt: {}'.format(prompt))
    logger.debug('predict_fn: Got input base64 image string (partial): {}...'.format(base64_image_string[:32]))
    logger.debug('predict_fn: Got input image: {}'.format(img))
    results = model(prompt=prompt, image=img, num_inference_steps=10, image_guidance_scale=1).images
    return results[0]


def output_fn(prediction_output, accept):
    logger.debug('output_fn: Got output image: {}'.format(prediction_output))
    logger.debug('output_fn: Accept: {}'.format(accept))

    buffered = BytesIO()
    prediction_output.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    base64_string = img_str.decode('latin1')

    logger.debug('output_fn: Response base64 string (partial): {}...'.format(base64_string[:32]))
    return base64_string
