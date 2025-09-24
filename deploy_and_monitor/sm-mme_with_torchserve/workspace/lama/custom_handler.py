import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import json
import base64

from abc import ABC
from io import BytesIO
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

class LamaHandler(BaseHandler, ABC):

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx: Context):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        predict_config = OmegaConf.load(f'{model_dir}/configs/prediction/default.yaml')
        predict_config.model.path = f'{model_dir}/big-lama'
        with open(f'{predict_config.model.path}/config.yaml', 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        self.model = load_checkpoint(
            train_config,
            checkpoint_path,
            strict=False,
            map_location='cpu')
        self.model.freeze()
        self.model.to(self.device)

        self.initialized = True

    def preprocess(self, data):

        requests = []
        for row in data:
            request = row.get("data") or row.get("body")

            if isinstance(request, (bytearray, bytes)):
                request = json.loads(request.decode('utf-8'))            

            if isinstance(request, dict) and \
                    "image" in request and \
                    "mask_image" in request:
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

                requests.append(request)
            else:
                raise RuntimeError("Dict request must include image and mask_image")

        return requests

    def inference(self, data):

        responses = []
        for request in data:
            mod = 8
            img = torch.from_numpy(request["image"]).float().div(255.)
            mask = torch.from_numpy(request["mask_image"]).float()

            batch = {}
            batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
            batch['mask'] = mask[None, None]
            unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
            batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
            batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
            #batch = move_to_device(batch, 'cuda')
            batch = move_to_device(batch, self.device)
            batch['mask'] = (batch['mask'] > 0) * 1

            batch = self.model(batch)
            cur_res = batch['inpainted'][0].permute(1, 2, 0)
            cur_res = cur_res.detach().cpu().numpy()

            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

            output_img = Image.fromarray(cur_res)
            output_img_bytes = self.encode_image(output_img)

            print(f'output_img_bytes:{output_img_bytes}')
            responses.append({"generated_image": output_img_bytes})

        return responses

    def handle(self, data, context):
        requests = self.preprocess(data)
        responses = self.inference(requests)

        return responses

    def dilate_mask(self, mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def encode_image(self, img):
        # Convert the image to bytes
        with BytesIO() as output:
            img.save(output, format="JPEG")
            img_bytes = output.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")
