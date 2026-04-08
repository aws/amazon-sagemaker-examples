"""
GRPO Training script using EasyR1/veRL framework for SageMaker.
Based on: https://github.com/hiyouga/EasyR1

Config is in train_config.yaml (copied from config.yml)
Only S3 data paths are passed as hyperparameters.
"""

import os
import sys

# ============================================================
# CRITICAL FIX: Patch Triton's driver.py file BEFORE any imports
# The UnicodeDecodeError occurs in triton/backends/nvidia/driver.py
# when it calls ldconfig -p and decodes with strict UTF-8
# ============================================================
def _patch_triton_source():
    """Patch the Triton source file to use errors='replace' for decode."""
    try:
        import site
        import glob
        
        # Find triton installation
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            if site_dir is None:
                continue
            pattern = os.path.join(site_dir, 'triton', 'backends', 'nvidia', 'driver.py')
            matches = glob.glob(pattern)
            for driver_path in matches:
                if os.path.exists(driver_path):
                    print(f"Found Triton driver at: {driver_path}")
                    with open(driver_path, 'r') as f:
                        content = f.read()
                    
                    # Check if already patched
                    if 'errors="replace"' in content or "errors='replace'" in content:
                        print("Triton driver already patched")
                        return True
                    
                    # Patch the decode calls
                    patched = content.replace('.decode("utf-8")', '.decode("utf-8", errors="replace")')
                    patched = patched.replace(".decode('utf-8')", ".decode('utf-8', errors='replace')")
                    
                    if patched != content:
                        with open(driver_path, 'w') as f:
                            f.write(patched)
                        print(f"Successfully patched Triton driver at {driver_path}")
                        return True
        
        print("Warning: Could not find Triton driver.py to patch")
        return False
    except Exception as e:
        print(f"Warning: Could not patch Triton source: {e}")
        return False

# Apply the source patch FIRST
_patch_triton_source()

# ============================================================
# Set environment variables to bypass ldconfig entirely
# TRITON_LIBCUDA_PATH tells Triton where to find libcuda.so
# ============================================================
os.environ['TRITON_LIBCUDA_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu'
os.environ['LANG'] = 'C.UTF-8'
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ============================================================
# CRITICAL: NCCL settings for GPUs without NVLink (g6e L40S, g5 A10G)
# Disable both P2P and SHM - force socket transport
# ============================================================
os.environ['NCCL_P2P_DISABLE'] = '1'       # Disable P2P
os.environ['NCCL_SHM_DISABLE'] = '1'       # Disable SHM (also uses P2P internally)
os.environ['NCCL_NET_GDR_LEVEL'] = '0'     # Disable GPUDirect RDMA
os.environ['NCCL_NVLS_ENABLE'] = '0'       # Disable NVLink features
os.environ['NCCL_IB_DISABLE'] = '1'        # Disable InfiniBand
os.environ['NCCL_DEBUG'] = 'INFO'

# ============================================================
# Load .env.local for WANDB_API_KEY and other settings
# ============================================================
def _load_env_file(env_path):
    """Load environment variables from .env file."""
    if os.path.exists(env_path):
        print(f"Loading environment from {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if value and value.lower() != 'false':
                        os.environ[key] = value
                        if 'KEY' in key or 'TOKEN' in key:
                            print(f"  Set {key}=***")
                        else:
                            print(f"  Set {key}={value}")

_load_env_file('/opt/ml/code/.env.local')

import json
import argparse
import subprocess
import shutil
from pathlib import Path

import boto3
import yaml


def download_s3_file(s3_uri: str, local_path: str):
    """Download file from S3."""
    s3 = boto3.client('s3')
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    print(f"Downloading s3://{bucket}/{key} to {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)
    return local_path


def download_s3_prefix(s3_uri: str, local_dir: str):
    """Download all files from S3 prefix."""
    s3 = boto3.client('s3')
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    os.makedirs(local_dir, exist_ok=True)
    
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel_path = key[len(prefix):].lstrip('/')
            if rel_path:
                local_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"Downloading s3://{bucket}/{key} to {local_path}")
                s3.download_file(bucket, key, local_path)
    
    return local_dir


def convert_json_to_parquet(json_path: str, parquet_path: str):
    """Convert JSON/JSONL to parquet format for EasyR1."""
    import pandas as pd
    
    if json_path.endswith('.jsonl'):
        data = []
        with open(json_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
    
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path, index=False)
    print(f"Converted {len(df)} samples to {parquet_path}")


def prepare_dataset(training_data_s3: str, data_dir: str):
    """Download and prepare dataset for EasyR1 format."""
    os.makedirs(data_dir, exist_ok=True)
    
    if training_data_s3.endswith('.parquet'):
        local_train = os.path.join(data_dir, "train.parquet")
        download_s3_file(training_data_s3, local_train)
    elif training_data_s3.endswith('.json') or training_data_s3.endswith('.jsonl'):
        local_json = os.path.join(data_dir, "data.json")
        download_s3_file(training_data_s3, local_json)
        convert_json_to_parquet(local_json, os.path.join(data_dir, "train.parquet"))
    else:
        download_s3_prefix(training_data_s3, data_dir)
    
    print(f"Dataset prepared at {data_dir}")
    print(f"Files: {os.listdir(data_dir)}")
    return data_dir


def update_data_paths(config_path: str, data_dir: str, output_dir: str, model_path: str = None):
    """Update only data paths in config, keep everything else from yaml."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data paths only
    config['data']['train_files'] = os.path.join(data_dir, 'train.parquet')
    if os.path.exists(os.path.join(data_dir, 'eval.parquet')):
        config['data']['val_files'] = os.path.join(data_dir, 'eval.parquet')
    else:
        config['data']['val_files'] = os.path.join(data_dir, 'train.parquet')
    
    # Update model path if provided (S3 model downloaded locally)
    if model_path:
        config['worker']['actor']['model']['model_path'] = model_path
    
    # Update output path
    config['trainer']['save_checkpoint_path'] = output_dir
    
    # Save updated config
    updated_config_path = '/tmp/train_config.yaml'
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config saved to: {updated_config_path}")
    return updated_config_path


def parse_args():
    """Parse SageMaker hyperparameters - only S3 paths needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_s3', type=str, required=True)
    parser.add_argument('--eval_data_s3', type=str, default=None)
    parser.add_argument('--model_s3', type=str, default=None, help='S3 path to model (optional, if not using HuggingFace)')
    args, _ = parser.parse_known_args()
    return args


def main():
    print("=" * 60)
    print("EasyR1 GRPO Training for Qwen3-VL")
    print("=" * 60)
    
    args = parse_args()
    
    print("\nS3 Data Paths:")
    print(f"  training_data_s3: {args.training_data_s3}")
    print(f"  eval_data_s3: {args.eval_data_s3}")
    print(f"  model_s3: {args.model_s3}")
    print("=" * 60)
    
    # Prepare directories
    data_dir = "/workspace/data_parquet"
    model_dir = "/workspace/model"
    output_dir = "/opt/ml/model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model from S3 if provided
    model_path = None
    if args.model_s3:
        print(f"\nDownloading model from S3: {args.model_s3}")
        download_s3_prefix(args.model_s3, model_dir)
        model_path = model_dir
        print(f"Model downloaded to: {model_path}")
    
    # Prepare dataset
    prepare_dataset(args.training_data_s3, data_dir)
    
    # Download eval data if provided
    if args.eval_data_s3:
        eval_path = os.path.join(data_dir, "eval.parquet")
        if args.eval_data_s3.endswith('.parquet'):
            download_s3_file(args.eval_data_s3, eval_path)
        else:
            local_json = os.path.join(data_dir, "eval.json")
            download_s3_file(args.eval_data_s3, local_json)
            convert_json_to_parquet(local_json, eval_path)
    
    # Base config path (not modified)
    code_dir = "/opt/ml/code"
    config_path = os.path.join(code_dir, "train_config.yaml")
    
    # Print base config
    print("\n" + "=" * 60)
    print("Base Training Config:")
    print("=" * 60)
    with open(config_path, 'r') as f:
        print(f.read())
    print("=" * 60)
    
    # Run EasyR1/veRL training
    # EasyR1 passes config overrides as CLI arguments (key=value format)
    # See: https://github.com/hiyouga/EasyR1
    print("\n" + "=" * 60)
    print("Starting EasyR1 GRPO training...")
    print("=" * 60)
    
    # Check environment
    print("\nChecking available modules...")
    subprocess.run([sys.executable, "-c", "import verl; print('verl found:', verl.__file__)"], check=False)
    subprocess.run([sys.executable, "-c", "import vllm; print('vllm version:', vllm.__version__)"], check=False)
    subprocess.run([sys.executable, "-c", "import torch; print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"], check=False)
    
    print("\nGPU info:")
    subprocess.run(["nvidia-smi"], check=False)
    
    # Build command with CLI overrides (EasyR1 style)
    train_file = os.path.join(data_dir, 'train.parquet')
    eval_file = os.path.join(data_dir, 'eval.parquet')
    if not os.path.exists(eval_file):
        eval_file = train_file
    
    cmd = [
        sys.executable, "-m", "verl.trainer.main",
        f"config={config_path}",
        f"data.train_files={train_file}",
        f"data.val_files={eval_file}",
        f"trainer.save_checkpoint_path={output_dir}",
    ]
    
    # Add model path override if using S3 model
    if model_path:
        cmd.append(f"worker.actor.model.model_path={model_path}")
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("=" * 60)
    sys.stdout.flush()
    
    # Build environment with NCCL settings
    nccl_env = {
        'NCCL_P2P_DISABLE': '1',
        'NCCL_SHM_DISABLE': '1',       # Disable shared memory (also uses P2P)
        'NCCL_NET_GDR_LEVEL': '0',
        'NCCL_NVLS_ENABLE': '0',
        'NCCL_IB_DISABLE': '1',
        'NCCL_DEBUG': 'INFO',
        'TRITON_LIBCUDA_PATH': '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu',
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'PYTHONIOENCODING': 'utf-8',
    }
    
    env = os.environ.copy()
    env.update(nccl_env)
    
    # Set RAY_RUNTIME_ENV to propagate settings to Ray workers
    import json as json_module
    env['RAY_RUNTIME_ENV'] = json_module.dumps({"env_vars": nccl_env})
    
    # Also write a usercustomize.py to patch Triton in all Python processes
    # This is more reliable than sitecustomize.py
    usercustomize_content = '''
import os
import sys

# Set environment variables for Triton
os.environ.setdefault('TRITON_LIBCUDA_PATH', '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu')
os.environ.setdefault('LANG', 'C.UTF-8')
os.environ.setdefault('LC_ALL', 'C.UTF-8')

# Patch Triton source file if not already patched
def _patch_triton():
    try:
        import site
        import glob
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            if site_dir is None:
                continue
            pattern = os.path.join(site_dir, 'triton', 'backends', 'nvidia', 'driver.py')
            for driver_path in glob.glob(pattern):
                if os.path.exists(driver_path):
                    with open(driver_path, 'r') as f:
                        content = f.read()
                    if 'errors="replace"' not in content and "errors=\\'replace\\'" not in content:
                        patched = content.replace('.decode("utf-8")', '.decode("utf-8", errors="replace")')
                        patched = patched.replace(".decode(\\'utf-8\\')", ".decode(\\'utf-8\\', errors=\\'replace\\')")
                        if patched != content:
                            with open(driver_path, 'w') as f:
                                f.write(patched)
    except:
        pass

_patch_triton()
'''
    
    # Write usercustomize.py to site-packages
    import site
    for site_dir in site.getsitepackages():
        if site_dir and os.path.exists(site_dir):
            usercustomize_path = os.path.join(site_dir, 'usercustomize.py')
            try:
                with open(usercustomize_path, 'w') as f:
                    f.write(usercustomize_content)
                print(f"Wrote usercustomize.py to {usercustomize_path}")
                # Also set PYTHONPATH to ensure it's loaded
                env['ENABLE_USER_SITE'] = 'True'
                break
            except Exception as e:
                print(f"Could not write usercustomize.py to {site_dir}: {e}")
    
    print(f"NCCL environment: {nccl_env}")
    print(f"RAY_RUNTIME_ENV: {env['RAY_RUNTIME_ENV']}")
    
    # No capture_output to allow streaming logs to CloudWatch
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"\n{'=' * 60}")
        print(f"Training failed with exit code {result.returncode}")
        print(f"{'=' * 60}")
        raise RuntimeError(f"verl.trainer.main failed with exit code {result.returncode}")
    
    print("\nTraining complete!")
    print(f"\nOutput files: {os.listdir(output_dir)}")
    
    # ============================================================
    # Merge FSDP shards to HuggingFace format for deployment
    # ============================================================
    print("\n" + "=" * 60)
    print("Merging FSDP shards to HuggingFace format...")
    print("=" * 60)
    
    # Find the latest checkpoint directory
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('global_step_')]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split('_')[-1]))
        latest_checkpoint = checkpoint_dirs[-1]
        checkpoint_dir = os.path.join(output_dir, latest_checkpoint, "actor")
        
        if os.path.exists(checkpoint_dir):
            print(f"Found checkpoint at: {checkpoint_dir}")
            print(f"Contents: {os.listdir(checkpoint_dir)}")
            
            # Use the model_merger.py script copied to /opt/ml/code during Docker build
            merger_script = "/opt/ml/code/model_merger.py"
            
            if os.path.exists(merger_script):
                print(f"Using merger script at: {merger_script}")
                merge_cmd = [
                    sys.executable,
                    merger_script,
                    "--local_dir", checkpoint_dir
                ]
                
                print(f"Running: {' '.join(merge_cmd)}")
                try:
                    merge_result = subprocess.run(merge_cmd, env=env, timeout=1800, capture_output=True, text=True)
                    print(f"Merge stdout: {merge_result.stdout}")
                    print(f"Merge stderr: {merge_result.stderr}")
                    print(f"Merge return code: {merge_result.returncode}")
                    
                    if merge_result.returncode == 0:
                        print("Merge completed successfully!")
                    else:
                        print(f"Warning: Merge failed with return code {merge_result.returncode}")
                        
                except subprocess.TimeoutExpired:
                    print("Warning: Merge timed out after 1800 seconds")
                except Exception as e:
                    print(f"Warning: Merge failed with exception: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Merger script not found at {merger_script}")
            
            # Check if huggingface folder now has model weights
            hf_model_path = os.path.join(checkpoint_dir, "huggingface")
            if os.path.exists(hf_model_path):
                hf_files = os.listdir(hf_model_path)
                print(f"\nHuggingFace folder contents: {hf_files}")
                safetensor_files = [f for f in hf_files if f.endswith('.safetensors')]
                if safetensor_files:
                    print(f"Found model weights: {safetensor_files}")
                else:
                    print("Warning: No .safetensors files found in huggingface folder")
        else:
            print(f"Warning: Checkpoint directory not found at {checkpoint_dir}")
    else:
        print("Warning: No checkpoint directories found")
    
    print("\n" + "=" * 60)
    print("Final output files:")
    print("=" * 60)
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            print(f"  {item}/ ({len(os.listdir(item_path))} files)")
        else:
            size = os.path.getsize(item_path) / (1024*1024)
            print(f"  {item} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
