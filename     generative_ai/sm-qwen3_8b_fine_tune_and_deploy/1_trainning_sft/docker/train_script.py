"""
SFT Training script using LlamaFactory for Qwen3-VL-8B on SageMaker.
"""

import os
import sys
import json
import argparse
import subprocess
import shutil

import boto3


def download_s3_file(s3_uri: str, local_path: str):
    """Download file from S3."""
    s3 = boto3.client('s3')
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    print(f"Downloading s3://{bucket}/{key} to {local_path}")
    s3.download_file(bucket, key, local_path)
    return local_path


def prepare_dataset(training_data_s3: str, dataset_dir: str):
    """Download and prepare dataset for LlamaFactory format."""
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download training data
    local_data = os.path.join(dataset_dir, "sft_data.json")
    download_s3_file(training_data_s3, local_data)
    
    # Flatten nested array if needed: [[{...}]] -> [{...}]
    with open(local_data, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        print(f"Flattening nested array: {len(data[0])} items")
        data = data[0]
        with open(local_data, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
    
    # Create dataset_info.json for LlamaFactory
    dataset_info = {
        "sft_data": {
            "file_name": "sft_data.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        }
    }
    
    with open(os.path.join(dataset_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset prepared at {dataset_dir}")
    print(f"Files: {os.listdir(dataset_dir)}")


def update_config(config_path: str, hyperparameters: dict):
    """Update LlamaFactory config with hyperparameters."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Map hyperparameters to config
    param_map = {
        'model_name_or_path': 'model_name_or_path',
        'lora_r': 'lora_rank',
        'lora_alpha': 'lora_alpha',
        'lora_dropout': 'lora_dropout',
        'per_device_train_batch_size': 'per_device_train_batch_size',
        'per_device_eval_batch_size': 'per_device_eval_batch_size',
        'gradient_accumulation_steps': 'gradient_accumulation_steps',
        'learning_rate': 'learning_rate',
        'num_train_epochs': 'num_train_epochs',
        'cutoff_len': 'cutoff_len',
        'val_size': 'val_size',
    }
    
    for hp_key, config_key in param_map.items():
        if hp_key in hyperparameters:
            value = hyperparameters[hp_key]
            # Convert types
            if config_key in ['lora_rank', 'lora_alpha', 'per_device_train_batch_size', 
                             'per_device_eval_batch_size', 'gradient_accumulation_steps',
                             'num_train_epochs', 'cutoff_len']:
                value = int(value)
            elif config_key in ['lora_dropout', 'learning_rate', 'val_size']:
                value = float(value)
            config[config_key] = value
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config: {config_path}")


def parse_args():
    """Parse SageMaker hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--training_data_s3', type=str, required=True)
    parser.add_argument('--lora_r', type=str, default='8')
    parser.add_argument('--lora_alpha', type=str, default='16')
    parser.add_argument('--lora_dropout', type=str, default='0.1')
    parser.add_argument('--per_device_train_batch_size', type=str, default='1')
    parser.add_argument('--per_device_eval_batch_size', type=str, default='2')
    parser.add_argument('--gradient_accumulation_steps', type=str, default='4')
    parser.add_argument('--learning_rate', type=str, default='5e-5')
    parser.add_argument('--num_train_epochs', type=str, default='2')
    parser.add_argument('--max_samples', type=str, default='50000')
    parser.add_argument('--val_size', type=str, default='0.05')
    parser.add_argument('--cutoff_len', type=str, default='2048')
    parser.add_argument('--gradient_checkpointing', type=str, default='1')
    parser.add_argument('--merge_weights', type=str, default='1')
    args, _ = parser.parse_known_args()
    return args


def main():
    print("=" * 60)
    print("Qwen3-VL-8B SFT Training with LlamaFactory")
    print("=" * 60)
    
    args = parse_args()
    
    print("\nConfiguration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    # Prepare dataset
    dataset_dir = "/tmp/data"
    prepare_dataset(args.training_data_s3, dataset_dir)
    
    # Copy and update config (baked into image at /opt/ml/code/)
    code_dir = "/opt/ml/code"
    config_src = os.path.join(code_dir, "train_config.yaml")
    config_dst = "/tmp/train_config.yaml"
    shutil.copy(config_src, config_dst)
    
    # Update config with hyperparameters
    update_config(config_dst, vars(args))
    
    # Run LlamaFactory training
    print("\n" + "=" * 60)
    print("Starting LlamaFactory SFT training...")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "llamafactory.cli", "train",
        config_dst
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    print("\nTraining complete!")
    
    # Debug: print merge_weights value
    print(f"\nDEBUG: merge_weights = '{args.merge_weights}' (type: {type(args.merge_weights)})")
    
    # Merge LoRA adapter with base model if requested
    if args.merge_weights == '1':
        print("\n" + "=" * 60)
        print("Merging LoRA adapter with base model...")
        print("=" * 60)
        
        merge_and_export(
            model_name_or_path=args.model_name_or_path,
            adapter_path="/opt/ml/model",
            output_path="/opt/ml/model/merged"
        )
        
        # Move merged model to output root and clean up adapter files
        merged_dir = "/opt/ml/model/merged"
        output_dir = "/opt/ml/model"
        
        # Remove adapter files from output
        adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
        for f in adapter_files:
            fpath = os.path.join(output_dir, f)
            if os.path.exists(fpath):
                os.remove(fpath)
        
        # Move merged files to output root
        for item in os.listdir(merged_dir):
            src = os.path.join(merged_dir, item)
            dst = os.path.join(output_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        
        shutil.rmtree(merged_dir)
        print("Merged model moved to /opt/ml/model")
    
    print(f"\nOutput files: {os.listdir('/opt/ml/model')}")


def merge_and_export(model_name_or_path: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter with base model using PEFT."""
    import torch
    from transformers import AutoTokenizer, AutoProcessor, AutoConfig
    from peft import PeftModel
    
    print(f"Loading base model: {model_name_or_path}")
    
    # Detect model type from config
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = config.model_type
    print(f"Detected model type: {model_type}")
    
    # Use appropriate model class based on type
    if "vl" in model_type.lower() or "vision" in model_type.lower():
        # Vision-Language model
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Text-only model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Save tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        processor.save_pretrained(output_path)
    except Exception as e:
        print(f"Note: Could not save processor: {e}")
    
    print("Merge complete!")


if __name__ == "__main__":
    main()
