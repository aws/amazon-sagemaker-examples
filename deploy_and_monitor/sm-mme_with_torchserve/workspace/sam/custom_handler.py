import base64
import cv2
import json
import numpy as np
import os
import torch

from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from six import BytesIO
from ts.torch_handler.base_handler import BaseHandler

class SAMHandler(BaseHandler):
    def __init__(self):
        # call superclass initializer
        super().__init__()

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
                "cuda:" + str(properties.get("gpu_id"))
                if torch.cuda.is_available() and properties.get("gpu_id") is not None
                else "cpu"
        )
        sam = sam_model_registry["vit_h"](checkpoint=model_pt_path).to(self.device)
        self.predictor = SamPredictor(sam)

        self.initialized = True

    def preprocess(self, data):

        requests = []
        for row in data:
            request = row.get("data") or row.get("body")

            if isinstance(request, (bytearray, bytes)):
                request = json.loads(request.decode('utf-8'))

            if isinstance(request, dict) and \
                    "image" in request and \
                    "gen_args" in request:
                img = request["image"]
                if isinstance(img, str):
                    img = base64.b64decode(img)

                with Image.open(BytesIO(img)) as f:
                    img_rgb = f.convert("RGB")
                    img_np_array = np.array(img_rgb)

                    request["image"] = img_np_array
                    requests.append(request)
            else:
                raise RuntimeError(f'Dict request must include image and gen_args, type={type(request)}')

        return requests 

    def inference(self, data):

        responses = []
        for request in data:
            self.predictor.set_image(request["image"])

            gen_args = request["gen_args"]
            gen_args_decoded = json.loads(gen_args)
            point_coords = np.array([gen_args_decoded["point_coords"]])
            point_labels = np.array([gen_args_decoded["point_labels"]])

            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            masks = masks.astype(np.uint8) * 255
            masks = [self.dilate_mask(mask, gen_args_decoded["dilate_kernel_size"]) for mask in masks]
            mask_img = Image.fromarray(masks[1].astype(np.uint8))
            output_img_bytes = self.encode_image(mask_img)

            responses.append({"generated_image": output_img_bytes})

        return responses

    def handle(self, data, context):
        with torch.no_grad():
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
