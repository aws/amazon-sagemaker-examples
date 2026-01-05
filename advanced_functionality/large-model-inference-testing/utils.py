import glob
from importlib import import_module
import json
import re
import os, stat
import shutil
import subprocess
import sys
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
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

def fill_template(template: dict, template_keys:list, inputs:list) -> dict:
        
    assert len(template_keys) <= len(inputs), f"template_keys: {template_keys}, prompts: {inputs}"
    for i, template_key in enumerate(template_keys):
        _template = template
        keys = template_key.split(".")
        for key in keys[:-1]:
            m = re.match(r'\[(\d+)\]', key)
            if m:
                key = int(m.group(1))
            _template = _template[key]

        _template[keys[-1]] = inputs[i]
    
    return template

def init_prompt_generator():
    prompt_module_dir = os.getenv("PROMPT_MODULE_DIR", None)
    assert prompt_module_dir, "PROMPT_MODULE_DIR environment variable is not set"
    
    sys.path.append(prompt_module_dir)

    prompt_module_name = os.getenv("PROMPT_MODULE_NAME", None)
    assert prompt_module_name, "PROMPT_MODULE_NAME environment variable is not set"
    prompt_module=import_module(prompt_module_name)
    
    prompt_generator_name = os.getenv('PROMPT_GENERATOR_NAME', None)
    assert prompt_generator_name, "PROMPT_GENERATOR_NAME environment variable is not set"
    prompt_generator_class = getattr(prompt_module, prompt_generator_name)
    
    prompt_generator = prompt_generator_class()()

    prompt_template = json.loads(os.getenv("TEMPLATE", "{}"))
    assert prompt_template, "TEMPLATE environment variable is not set"
    prompt_template_keys = json.loads(os.getenv("TEMPLATE_KEYS", "[]"))
    assert prompt_template_keys, "TEMPLATE_KEYS environment variable is not set"

    return prompt_generator, prompt_template, prompt_template_keys

def set_prompt_env(test_spec: dict):
    os.environ["PROMPT_MODULE_DIR"] = test_spec.get('module_dir', None)
    assert os.environ["PROMPT_MODULE_DIR"], "PROMPT_MODULE_DIR is not set"
    os.environ["PROMPT_MODULE_NAME"] = test_spec.get('module_name', None)
    assert os.environ["PROMPT_MODULE_NAME"], "PROMPT_MODULE_NAME is not set"
    os.environ["PROMPT_GENERATOR_NAME"] = test_spec.get('prompt_generator', None)
    assert os.environ["PROMPT_GENERATOR_NAME"], "PROMPT_GENERATOR_NAME is not set"
    os.environ["TEMPLATE"] = json.dumps(test_spec.get('template', {}))
    assert os.environ["TEMPLATE"], "TEMPLATE is not set"
    os.environ["TEMPLATE_KEYS"] = json.dumps(test_spec.get('template_keys', []))
    assert os.environ["TEMPLATE_KEYS"], "TEMPLATE_KEYS is not set"

def snapshot_hf_model_to_s3(s3_client, s3_prefix:str, s3_bucket:str, model_config:dict, hf_spec:dict, hf_token:str=None):
    hf_model = hf_spec.get('model', None)
    download = hf_spec.get('download', None)
    if download: 
        revision = hf_spec.get('revision', None)
        assert revision, "'huggingface.revision' is required if 'download' is 'true'"
        
        s3_model_prefix = f"{s3_prefix}/huggingface/models/{hf_model}/{revision}"  # folder where model checkpoint will go
        print(f"s3_model_prefix: {s3_model_prefix}")

        filtered_hf_files = []
        install_pip_package(package_name="huggingface-hub")
        from huggingface_hub import snapshot_download, list_repo_files
       
        hf_files = list_repo_files(repo_id=hf_model, revision=revision, token=hf_token)
        if not hf_files:
            raise Exception(f"Unable to find any files for model: {hf_model}, revision: {revision}")
        
        ignore_patterns = ["*.msgpack", "*.h5"]

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

        model_s3_url = f"s3://{s3_bucket}/{s3_model_prefix}/"
        model_config['djl']['option.model_id'] = model_s3_url
    else:
        model_config['djl']['option.model_id'] = hf_model

def djl_model_package_to_s3(s3_client, 
                            djl_spec: dict,
                            sm_model_name: str, 
                            config_path: str, 
                            s3_prefix:str, 
                            s3_bucket:str ):
    
    model_pkg_key = None
    if djl_spec is not None:
        with TemporaryDirectory(suffix="pkg", prefix="model", dir=".") as pkg_dir:

            with open(os.path.join(pkg_dir, "serving.properties"), "w") as props_file:
                for key, value in djl_spec.items():
                    props_file.write(f"{key}={value}\n")

            code_dir = os.path.join(os.path.dirname(config_path), "code")
            if os.path.isdir(code_dir):
                files = glob.glob(f"{code_dir}/*")

                for file in files:
                    if os.path.isdir(file):
                        shutil.copytree(file, os.path.join(pkg_dir, os.path.basename(file)))
                    else:
                        shutil.copy2(file, pkg_dir)

            with NamedTemporaryFile(prefix="model", suffix=".gz") as gz_file:
                with tarfile.open(gz_file.name, "w:gz") as tar:
                    tar.add(pkg_dir, arcname="")

                gz_file.seek(0)
                model_pkg_key = f"{s3_prefix}/sagemaker/code/{sm_model_name}/model.tar.gz"
                print(f"Upload model package to s3://{s3_bucket}/{model_pkg_key}")
                s3_client.upload_fileobj(gz_file, s3_bucket, model_pkg_key)

    return model_pkg_key


def generic_model_package_to_s3(s3_client, 
                            sm_model_name: str, 
                            config_path: str, 
                            s3_prefix:str, 
                            s3_bucket:str ):
    
    model_pkg_key = None
    with TemporaryDirectory(suffix="pkg", prefix="model", dir=".") as pkg_dir:

        code_dir = os.path.join(os.path.dirname(config_path), "code")
        if os.path.isdir(code_dir):
            files = glob.glob(f"{code_dir}/*")

            for file in files:
                if os.path.basename(file).startswith('.'):
                    continue
              
                if os.path.isdir(file):
                    shutil.copytree(file, os.path.join(pkg_dir, os.path.basename(file)), ignore=shutil.ignore_patterns('.*'))
                else:
                    shutil.copy2(file, pkg_dir)

        with NamedTemporaryFile(prefix="model", suffix=".gz") as gz_file:
            with tarfile.open(gz_file.name, "w:gz") as tar:
                tar.add(pkg_dir, arcname="")

            gz_file.seek(0)
            model_pkg_key = f"{s3_prefix}/sagemaker/code/{sm_model_name}/model.tar.gz"
            print(f"Upload model package to s3://{s3_bucket}/{model_pkg_key}")
            s3_client.upload_fileobj(gz_file, s3_bucket, model_pkg_key)

    return model_pkg_key

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