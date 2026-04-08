"""
Trigger SageMaker async endpoint with text and optional images.
Supports OpenAI-compatible vision API format for Qwen3-VL.
"""

import time
import json
import uuid
import base64
from io import BytesIO

import boto3
from PIL import Image

# ==========================================
# Configuration - ⚠️ MODIFY FOR YOUR ENVIRONMENT
# ==========================================
ENDPOINT_NAME = "your-model-endpoint"
INPUT_BUCKET = "sagemaker-us-east-2-YOUR_ACCOUNT_ID"
REGION = "us-east-2"

# Initialize clients
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def load_image(image_path: str) -> Image.Image:
    """Load image from local path or S3."""
    if image_path.startswith("s3://"):
        # Parse S3 path
        parts = image_path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(response['Body'].read()))
    else:
        return Image.open(image_path)


def invoke_sagemaker_async(
    prompt: str, 
    images: list[Image.Image] = None
) -> tuple[str, float]:
    """
    Call SageMaker async endpoint with text and images.
    
    Args:
        prompt: Text prompt
        images: List of PIL Images (optional)
    
    Returns:
        (result_text, elapsed_seconds)
    """
    start_time = time.time()
    
    # Build content for OpenAI-compatible vision API (image first, then text)
    content = []
    
    # Add images first (if any)
    if images:
        for img in images:
            base64_img = image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
    
    # Add text after images
    content.append({
        "type": "text",
        "text": prompt
    })
    
    # Prepare payload (OpenAI-compatible format for vLLM)
    data = {
        "model": "/opt/ml/model",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False
    }
    
    # Upload input to S3
    input_key = f'async_inference_input/{uuid.uuid4()}.json'
    s3_client.put_object(
        Bucket=INPUT_BUCKET,
        Key=input_key,
        Body=json.dumps(data),
        ContentType='application/json'
    )
    input_location = f's3://{INPUT_BUCKET}/{input_key}'
    print(f"Input uploaded to: {input_location}")
    
    # Invoke async endpoint
    response = sagemaker_runtime.invoke_endpoint_async(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        InputLocation=input_location
    )
    
    output_location = response['OutputLocation']
    print(f"Async inference started. Output: {output_location}")
    
    # Parse S3 path
    bucket = output_location.split('/')[2]
    key = '/'.join(output_location.split('/')[3:])
    failure_key = key + ".failure"
    
    # Poll for result with exponential backoff
    max_wait = 300  # 5 minutes timeout
    waited = 0
    poll_interval = 2
    
    print("Waiting for result", end="", flush=True)
    
    while waited < max_wait:
        # Check for failure file first
        try:
            s3_client.head_object(Bucket=bucket, Key=failure_key)
            failure_result = s3_client.get_object(Bucket=bucket, Key=failure_key)
            failure_msg = failure_result['Body'].read().decode('utf-8')
            raise Exception(f"Inference failed: {failure_msg}")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                raise e
        
        # Check for success output
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            result = s3_client.get_object(Bucket=bucket, Key=key)
            output = json.loads(result['Body'].read().decode('utf-8'))
            
            # vLLM returns OpenAI-compatible format
            if isinstance(output, dict) and 'choices' in output:
                elapsed = time.time() - start_time
                return output['choices'][0]['message']['content'], elapsed
            
            elapsed = time.time() - start_time
            return str(output), elapsed
            
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                time.sleep(poll_interval)
                waited += poll_interval
                poll_interval = min(poll_interval * 1.2, 5)
                print(".", end="", flush=True)
            else:
                raise e
    
    raise TimeoutError(f"Timeout after {max_wait}s waiting for inference result")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Example 1: Text only
    prompt = """You are a professional product tagging expert. Please analyze the following product and generate tags:
    
Product name: [FuturBeauty] Time Capsule Essence Mask (Hydrating)
Product features: Dual-layer time capsule, double repair power! Contains Centella Asiatica extract, Provitamin B5

Please output the tag categories."""

    print("=" * 60)
    print("Text-only inference")
    print("=" * 60)
    result, elapsed = invoke_sagemaker_async('/no_think ' + prompt)
    print(f"\n\n=== Result ({elapsed:.2f}s) ===")
    print(result)
    
    # Example 2: With image (uncomment to use)
    # print("\n" + "=" * 60)
    # print("Image + Text inference")
    # print("=" * 60)
    # 
    # # Load image from local path or S3
    # image = load_image("path/to/product_image.jpg")
    # # Or from S3:
    # # image = load_image("s3://your-bucket/images/product.jpg")
    # 
    # prompt_with_image = "Describe the content of this product image and generate relevant tags."
    # result, elapsed = invoke_sagemaker_async(prompt_with_image, images=[image])
    # print(f"\n\n=== Result ({elapsed:.2f}s) ===")
    # print(result)
