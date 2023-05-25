import base64
import cv2
import json
import numpy as np
import os
import torch

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from io import BytesIO

from abc import ABC
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

class DiffuserHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.initialized = False

    def initialize(self, ctx: Context):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)

        #self.pipeline.to(self.device)
        self.pipeline.to('cuda')
        self.pipeline.enable_xformers_memory_efficient_attention()

        self.initialized = True

    def preprocess(self, data):

        requests = []
        for row in data:
            request = row.get("data") or row.get("body")

            if isinstance(request, (bytearray, bytes)):
                request = json.loads(request.decode('utf-8'))

            if isinstance(request, dict) and \
                    "image" in request and \
                    "prompt" in request and \
                    "negative_prompt" in request and \
                    "mask_image" in request and \
                    "gen_args" in request:
                img = request["image"]
                if isinstance(img, str):
                    img = base64.b64decode(img)

                with Image.open(BytesIO(img)) as f:
                    img_rgb = f.convert("RGB")
                    img_np_array = np.array(img_rgb)
                    request["image"] = img_np_array

                mask_img = request["mask_image"]
                if isinstance(mask_img, str):
                    mask_img = base64.b64decode(mask_img)

                with Image.open(BytesIO(mask_img)) as f:
                    mask_img_rgb = f.convert("L")
                    mask_img_np_array = np.array(mask_img_rgb)
                    request["mask_image"] = mask_img_np_array

                prompt_text = request["prompt"]
                if isinstance(prompt_text, (bytes, bytearray)):
                    prompt_text = prompt_text.decode("utf-8")
                    request["prompt"] = prompt_text

                negative_prompt = request["negative_prompt"]
                if isinstance(negative_prompt, (bytes, bytearray)):
                    negative_prompt = negative_prompt.decode("utf-8")
                    request["negative_prompt"] = negative_prompt

                requests.append(request)
            else:
                raise RuntimeError("Dict request must include image, prompt, negative_prompt, gen_args and mask_image")

        return requests

    def inference(self, data):

        responses = []
        for request in data:
            gen_args = request["gen_args"]
            gen_args_decoded = json.loads(gen_args)
            generator = [torch.Generator(device="cuda").manual_seed(gen_args_decoded['seed'])]
            #generator = [torch.Generator(device=self.device).manual_seed(gen_args_decoded['seed'])]

            img_crop, mask_crop = crop_for_filling_pre(request["image"], request["mask_image"])
            with torch.no_grad():
                img_crop_filled = self.pipeline(
                    prompt=request["prompt"],
                    negative_prompt=request["negative_prompt"],
                    image=Image.fromarray(img_crop),
                    mask_image=Image.fromarray(mask_crop),
                    num_inference_steps=gen_args_decoded['num_inference_steps'],
                    guidance_scale=gen_args_decoded['guidance_scale'],
                    generator=generator,
                ).images[0]

            image_array = crop_for_filling_post(request["image"], request["mask_image"],
                                                np.array(img_crop_filled))

            generated_image = Image.fromarray(np.squeeze(image_array))
            output_img_bytes = self.encode_image(generated_image)

            responses.append({"generated_image": output_img_bytes})

        return responses

    def handle(self, data, context):
        requests = self.preprocess(data)
        responses = self.inference(requests)

        return responses

    def encode_image(self, img):
        # Convert the image to bytes
        with BytesIO() as output:
            img.save(output, format="JPEG")
            img_bytes = output.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")


def crop_for_filling_pre(image: np.array, mask: np.array, crop_size: int = 512):
    # Calculate the aspect ratio of the image
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    # If the shorter side is less than 512, resize the image proportionally
    if min(height, width) < crop_size:
        if height < width:
            new_height = crop_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = crop_size
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))

    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Update the height and width of the resized image
    height, width = image.shape[:2]

    # # If the 512x512 square cannot cover the entire mask, resize the image accordingly
    if w > crop_size or h > crop_size:
        # padding to square at first
        if height < width:
            padding = width - height
            image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
        else:
            padding = height - width
            image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')

        resize_factor = crop_size / max(w, h)
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
        x, y, w, h = cv2.boundingRect(mask)

    # Calculate the crop coordinates
    crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
    crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)

    # Crop the image
    cropped_image = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    cropped_mask = mask[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return cropped_image, cropped_mask
    
    
def crop_for_filling_post(
        image: np.array,
        mask: np.array,
        filled_image: np.array, 
        crop_size: int = 512,
        ):
    image_copy = image.copy()
    mask_copy = mask.copy()
    # Calculate the aspect ratio of the image
    height, width = image.shape[:2]
    height_ori, width_ori = height, width
    aspect_ratio = float(width) / float(height)

    # If the shorter side is less than 512, resize the image proportionally
    if min(height, width) < crop_size:
        if height < width:
            new_height = crop_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = crop_size
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))

    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Update the height and width of the resized image
    height, width = image.shape[:2]

    # # If the 512x512 square cannot cover the entire mask, resize the image accordingly
    if w > crop_size or h > crop_size:
        flag_padding = True
        # padding to square at first
        if height < width:
            padding = width - height
            image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
            padding_side = 'h'
        else:
            padding = height - width
            image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')
            padding_side = 'w'

        resize_factor = crop_size / max(w, h)
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
        x, y, w, h = cv2.boundingRect(mask)
    else:
        flag_padding = False

    # Calculate the crop coordinates
    crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
    crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)

    # Fill the image
    image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size] = filled_image
    if flag_padding:
        image = cv2.resize(image, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
        if padding_side == 'h':
            image = image[padding // 2:padding // 2 + height_ori, :]
        else:
            image = image[:, padding // 2:padding // 2 + width_ori]

    image = cv2.resize(image, (width_ori, height_ori))

    image_copy[mask_copy==255] = image[mask_copy==255]
    return image_copy
