import base64
import json
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
img_bytes = None
with Image.open(img_file) as f:
    img_bytes = encode_image(f)

mask_file = 'sample1_mask.jpg'
mask = Image.open(mask_file)
mask_bytes = encode_image(mask)

prompt = "a teddy bear on a bench"
nprompt = "ugly"
gen_args = json.dumps(dict(num_inference_steps=50, guidance_scale=10, seed=1))

payload = {
        "image": img_bytes, 
        "mask_image": mask_bytes,
        "prompt": prompt,
        "negative_prompt": nprompt,
        "gen_args": gen_args
        }

url="http://127.0.0.1:8080/predictions/sd"
response = httpx.post(url, json=payload, timeout=None)
encoded_masks_string = response.json()['generated_image']
base64_bytes_masks = base64.b64decode(encoded_masks_string)
print(base64_bytes_masks)
with Image.open(io.BytesIO(base64_bytes_masks)) as f:
    generated_image_rgb=f.convert("RGB")
    generated_image_rgb.show()
