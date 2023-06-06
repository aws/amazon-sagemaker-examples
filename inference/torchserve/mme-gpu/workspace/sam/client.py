import base64
import json
from json import JSONEncoder
import io
import numpy as np
from PIL import Image
import httpx

def encode_image(img):
    
    # Convert the image to bytes
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        img_bytes = output.getvalue()
    
    return base64.b64encode(img_bytes).decode('utf8')

img_file = 'sample1.png'
img_bytes = None
with Image.open(img_file) as f:
    img_bytes = encode_image(f)

gen_args = json.dumps(dict(point_coords=[750, 500], point_labels=1, dilate_kernel_size=15))

payload = {
        "image": img_bytes, 
        "gen_args": gen_args
        }

url="http://127.0.0.1:8080/predictions/sam"
response = httpx.post(url, json=payload, timeout=None)
encoded_masks_string = response.json()['generated_image']
base64_bytes_masks = base64.b64decode(encoded_masks_string)
with Image.open(io.BytesIO(base64_bytes_masks)) as f:
    generated_image_rgb=f.convert("RGB")
    generated_image_rgb.show()
