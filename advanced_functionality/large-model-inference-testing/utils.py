import re
import os
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from transformers import AutoTokenizer

def s3_bucket_keys(s3_client, bucket_name:str, bucket_prefix:str):
    """Generator for listing S3 bucket keys matching prefix"""
    kwargs = {'Bucket': bucket_name, 'Prefix': bucket_prefix}
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            yield obj['Key']
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break
                
def s3_download_model(s3_client, bucket: str, prefix: str, local_dir: str, includes:list=None):
    """Download model from S3"""
    for path in s3_bucket_keys(s3_client, bucket_name=bucket, bucket_prefix=prefix):
        if includes and os.path.splitext(path)[1] not in includes:
            continue
        print(f"Downloading: {path}")
        local_path = os.path.join(local_dir, os.path.basename(path))
        s3_client.download_file(bucket, path, local_path)
                    
def get_tokenizer(s3_client, model_id: str):
    tokenizer = None
    if re.match(r"^s3://([^/]+)/?(.*)?", model_id):
        s3_uri_parse = urlparse(model_id)
        model_bucket = s3_uri_parse.netloc
        model_prefix = s3_uri_parse.path[1:]
        with TemporaryDirectory(suffix="snapshot", prefix="model", dir=".") as local_dir:
            s3_download_model(s3_client, bucket=model_bucket, 
                           prefix=model_prefix, local_dir=local_dir, 
                           includes=[".json", ".model"])
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    return tokenizer