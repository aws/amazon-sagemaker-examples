import re
import os, stat
import subprocess
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
                    
def get_tokenizer(s3_client, model_id: str, hf_token:str=None):
    tokenizer = None
    if re.match(r"^s3://([^/]+)/?(.*)?", model_id):
        s3_uri_parse = urlparse(model_id)
        model_bucket = s3_uri_parse.netloc
        model_prefix = s3_uri_parse.path[1:]
        with TemporaryDirectory(suffix="snapshot", prefix="model", dir=".") as local_dir:
            s3_download_model(s3_client, bucket=model_bucket, 
                           prefix=model_prefix, local_dir=local_dir, 
                           includes=[".json", ".model", ".py"])
            tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
        
    return tokenizer

def install_pip_requirements(requirements_path:str):
    requirements_path_abs = os.path.abspath(requirements_path)
    command = ['pip', 'install', '-r', f'{requirements_path_abs}']
    return subprocess.check_output(command, shell=False, stderr=subprocess.STDOUT)

def install_pip_package(package_name:str):
    command = ['pip', 'install', f'{package_name}']
    return subprocess.check_output(command, shell=False, stderr=subprocess.STDOUT)

def snapshot_hf_model_to_s3(s3_client, s3_prefix:str, s3_bucket:str, hf_spec:dict, hf_token:str=None):
    if hf_spec: 
        hf_model = hf_spec.get('model', None)
        assert hf_model, "'huggingface.model' is required'"

        revision = hf_spec.get('revision', None)
        assert revision, "'huggingface.revision' is required'"

        tensors = hf_spec.get('tensors', None)
        if not tensors:
            print(f"Downloading {hf_model}:{revision} from HuggingFace without tensors")
        else:
            print(f"Downloading {hf_model}:{revision} from HuggingFace Hub with tensors")
    
        s3_model_prefix = f"{s3_prefix}/huggingface/models/{hf_model}/{revision}"  # folder where model checkpoint will go
        print(f"s3_model_prefix: {s3_model_prefix}")

        filtered_hf_files = []
        install_pip_package(package_name="huggingface-hub")
        from huggingface_hub import snapshot_download, list_repo_files
       
        hf_files = list_repo_files(repo_id=hf_model, revision=revision, token=hf_token)
        if not hf_files:
            raise Exception(f"Unable to find any files for model: {hf_model}, revision: {revision}")
        
        ignore_patterns = ["*.msgpack", "*.h5"] if tensors else [ "*.msgpack", "*.h5", "*.bin", "*.safetensors"]

        # Filter out ignored patterns
        for file in hf_files:
            should_ignore = False
            for pattern in ignore_patterns:
                if pattern.replace("*", "") in file:
                    should_ignore = True
                    break
            if not should_ignore:
                filtered_hf_files.append(file)

        missing_files = False
        
        for file in filtered_hf_files:
            try:
                s3_client.head_object(Bucket=s3_bucket, Key=f"{s3_model_prefix}/{file}")
            except:
                missing_files = True
                break

        if missing_files:
            from tempfile import TemporaryDirectory
            from pathlib import Path

            print(f"Downloading HuggingFace model snapshot: {hf_model}, revision: {revision}")
            with TemporaryDirectory(suffix="model", prefix="hf", dir=".") as cache_dir:
                snapshot_download(repo_id=hf_model, 
                    revision=revision, 
                    cache_dir=cache_dir,
                    ignore_patterns=ignore_patterns,
                    token=hf_token)

                local_model_path = Path(cache_dir)
                model_snapshot_path = str(list(local_model_path.glob(f"**/snapshots/{revision}"))[0])
                print(f"model_snapshot_path: {model_snapshot_path}")

                for root, dirs, files in os.walk(model_snapshot_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        with open(full_path, 'rb') as data:
                            key = f"{s3_model_prefix}/{full_path[len(model_snapshot_path)+1:]}"
                            s3_client.upload_fileobj(data, s3_bucket, key)

                print(f"Uploaded HuggingFace model snapshot to S3: {hf_model}, revision: {revision}")
    

def push_ecr_container(container_path: str, aws_region: str, aws_account_id: str) -> str:

    container_path = os.path.abspath(container_path)
    with open(os.path.join(container_path, "build.log"), "w") as logfile:
        print(f"Building and pushing {container_path} to ECR; see log file: {container_path}/build.log")
        container_build_script = os.path.join(container_path, "build_tools", "build_and_push.sh")

        st = os.stat(container_build_script)
        os.chmod(container_build_script, st.st_mode | stat.S_IXUSR)
        subprocess.check_call([container_build_script, aws_region], stdout=logfile, stderr=subprocess.STDOUT)

        image_tag = None
        image_name = None
        with open(os.path.join(container_path, "build_tools", "set_env.sh")) as f:
            for line in f:
                m = re.match(r".*IMAGE_TAG=(.*)", line)
                if m:
                    image_tag = m.group(1)
                else:
                    m = re.match(r".*IMAGE_NAME=(.*)", line)
                    if m:
                        image_name = m.group(1)

        assert image_tag, "IMAGE_TAG is not set"
        assert image_name, "IMAGE_NAME is not set"

        ecr_image_uri=f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{image_name}:{image_tag}"
    
    return ecr_image_uri

def fsx_file_systems(fsx_client):
    """Generator for listing Fsx file systems"""

    next_token = None
    while True:
        if next_token:
            resp = fsx_client.describe_file_systems(NextToken=next_token)
        else:
            resp = fsx_client.describe_file_systems()
            
        file_systems = resp['FileSystems']
        for fs in file_systems:
            yield fs

        try:
            next_token = resp['NextToken']
        except KeyError:
            break