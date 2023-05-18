import base64
import json
from json import JSONEncoder
import io
import numpy as np
from PIL import Image
import httpx
from io import BytesIO

def encode_image(img):
    
    # Convert the image to bytes
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        img_bytes = output.getvalue()
    
    return base64.b64encode(img_bytes).decode('utf8')

img_file = 'sample1.png'
img = Image.open(img_file)
img_bytes = encode_image(img)

mask_file = 'sample1_mask.jpg'
mask = Image.open(mask_file)
mask_bytes = encode_image(mask)

payload = {
        "image": img_bytes, 
        "mask_image": mask_bytes,
        }

url="http://127.0.0.1:8080/predictions/lama"
response = httpx.post(url, json=payload, timeout=None)
encoded_masks_string = response.json()['generated_image']
base64_bytes_masks = base64.b64decode(encoded_masks_string)
print(base64_bytes_masks)
#generated_image_rgb=Image.open(io.BytesIO(base64_bytes_masks)).convert("RGB")
#generated_image_rgb.show()
