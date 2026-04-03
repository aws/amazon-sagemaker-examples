import os
import argparse
import json
import re
import time
import shutil
import tempfile
import gc
from dataclasses import dataclass, fields, MISSING
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import evaluate
from vllm import LLM, SamplingParams


@dataclass
class Config:
    # Checkpoint and Model
    model_path: str = None
    base_model: str = "Qwen/Qwen3-8B"
    checkpoints_dir: str = None
    
    # Training mode
    full_ft: bool = False
    
    # LoRA settings (only used when full_ft=False)
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    
    # Data
    test_path: str = None
    max_samples: int = 1024
    batch_size: int = 128
    
    # Generation settings
    temperature: float = 0.1
    top_k: int = -1  # vLLM uses -1 for disabled
    top_p: float = 0.95
    max_tokens: int = 512  # max_new_tokens
    
    # vLLM settings
    tensor_parallel_size: int = 8
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    dtype: str = "bfloat16"
    
    @property
    def output_path(self) -> str:
        return str(Path(self.checkpoint_path) / "predictions.jsonl")
    
    @property
    def checkpoint_path(self) -> str:
        """Find the latest checkpoint in the checkpoints directory."""
        ckpt_dir_path = Path(self.checkpoints_dir)
        # Look for checkpoint-* directories (exclude converted ones)
        ckpt_dirs = []
        for d in ckpt_dir_path.glob("checkpoint-*"):
            if d.is_dir() and not re.search(r'\.(hf_model|hf_peft|merged)', d.name):
                ckpt_dirs.append(d)
        ckpt_dirs = sorted(ckpt_dirs, key=lambda p: int(p.name.split('-')[1]))
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoint directories found in {ckpt_dir_path}")
        return str(ckpt_dirs[-1])
    

    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() if k in config_fields}
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.model_path is None:
            self.model_path = self.base_model

        if self.checkpoints_dir is None:
            self.checkpoints_dir = str(Path.home() / f"results/{self.base_model}")
        
        if self.test_path is None:
            datasets_path = Path.home() / "datasets"
            test_files = list(datasets_path.rglob("test.jsonl"))
            if test_files:
                self.test_path = str(test_files[0])
                print(f"Found test file: {self.test_path}")
            else:
                raise ValueError("No test.jsonl file found under datasets folder")

def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for f in fields(dataclass_type):
        field_type = f.type
        default_value = f.default if f.default is not MISSING else None
        parser.add_argument(
            f'--{f.name}',
            type=field_type if field_type in [int, float, str] else str,
            default=default_value
        )
    return parser

def load_checkpoint_in_memory(config: Config):
    """Load FSDP checkpoint and optionally merge LoRA adapters in memory."""
    
    print("=" * 80)
    print("LOADING CHECKPOINT INTO MEMORY")
    print("=" * 80)
    print(f"Source: {config.checkpoint_path}")
    print(f"Mode: {'Full Fine-Tuning' if config.full_ft else 'LoRA (will merge adapters)'}")
    print("=" * 80)
    
    # Load base model
    print("\n[1/3] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load FSDP checkpoint
    print("\n[2/3] Loading FSDP checkpoint weights...")
    fsdp_checkpoint_dir = Path(config.checkpoint_path) / "pytorch_model_fsdp_0"
    
    if not fsdp_checkpoint_dir.exists():
        raise FileNotFoundError(f"FSDP checkpoint not found: {fsdp_checkpoint_dir}")
    
    state_dict = {}
    dcp.load(state_dict, checkpoint_id=str(fsdp_checkpoint_dir))
    
    if config.full_ft:
        # For full fine-tuning, load weights directly into base model
        print("Loading full fine-tuning weights...")
        base_model.load_state_dict(state_dict, strict=False)
        final_model = base_model
    else:
        # For LoRA, need to apply LoRA config, load weights, then merge
        print("Applying LoRA configuration...")
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[m.strip() for m in config.lora_target_modules.split(',')],
            bias="none",
        )
        model_with_adapters = get_peft_model(base_model, peft_config)
        
        print("Loading LoRA weights...")
        model_with_adapters.load_state_dict(state_dict, strict=False)
        
        print("\n[3/3] Merging LoRA adapters...")
        final_model = model_with_adapters.merge_and_unload()
    
    print("\n✓ Checkpoint loaded and ready for inference!")
    print("=" * 80)
    
    return final_model

def load_model_with_vllm(config: Config):
    """Load model using vLLM for efficient inference with temporary model storage."""
    
    # Load checkpoint and merge in memory
    model = load_checkpoint_in_memory(config)
    
    # Create temporary directory for the model
    temp_dir = tempfile.mkdtemp(prefix="vllm_model_")
    
    try:
        print("\n" + "=" * 80)
        print("INITIALIZING vLLM")
        print("=" * 80)
        print(f"Base model: {config.model_path}")
        print(f"Tensor parallel size: {config.tensor_parallel_size}")
        print(f"GPU memory utilization: {config.gpu_memory_utilization}")
        print(f"Max model length: {config.max_model_len}")
        print("=" * 80)
        
        # Save model and tokenizer to temp directory
        print(f"\nSaving model to temporary directory: {temp_dir}")
        model.save_pretrained(temp_dir)
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        tokenizer.save_pretrained(temp_dir)
        
        # Initialize vLLM from temp directory
        print("Loading model into vLLM...")
        llm = LLM(
            model=temp_dir,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
            disable_log_stats=True,
        )
        
        print("\n✓ vLLM initialized successfully!")
        print("=" * 80 + "\n")
        
        # Store temp_dir path so we can clean it up later
        llm._temp_dir = temp_dir
        
        return llm
        
    except Exception as e:
        # Clean up temp directory if initialization fails
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def generate_and_save_predictions(llm: LLM, config: Config):
    """Generate predictions for test samples using vLLM batched inference."""
    
    sampling_params = SamplingParams(
        temperature=config.temperature if config.temperature > 0 else 0.0,
        top_k=config.top_k if config.top_k > 0 else -1,
        top_p=config.top_p if config.temperature > 0 else 1.0,
        max_tokens=config.max_tokens,
    )
    
    print(f"\nGenerating predictions (batch_size={config.batch_size})...")
    start_time = time.time()
    
    total_samples = 0
    total_tokens_generated = 0
    
    with open(config.output_path, 'w') as f_out:
        chunk_inputs = []
        chunk_labels = []
        
        with open(config.test_path, 'r') as f_in:
            for idx, line in enumerate(f_in):
                if idx >= config.max_samples:
                    break
                
                sample = json.loads(line)
                chunk_inputs.append(sample['input'])
                chunk_labels.append(sample['output'])
                
                if len(chunk_inputs) >= config.batch_size:
                    outputs = llm.generate(chunk_inputs, sampling_params)
                    for input_text, label, output in zip(chunk_inputs, chunk_labels, outputs):
                        f_out.write(json.dumps({
                            'input': input_text,
                            'label': label,
                            'prediction': output.outputs[0].text
                        }) + '\n')
                        total_tokens_generated += len(output.outputs[0].token_ids)
                    
                    total_samples += len(chunk_inputs)
                    print(f"  Processed {total_samples} samples...")
                    chunk_inputs = []
                    chunk_labels = []
        
        # Process remaining samples
        if chunk_inputs:
            outputs = llm.generate(chunk_inputs, sampling_params)
            for input_text, label, output in zip(chunk_inputs, chunk_labels, outputs):
                f_out.write(json.dumps({
                    'input': input_text,
                    'label': label,
                    'prediction': output.outputs[0].text
                }) + '\n')
                total_tokens_generated += len(output.outputs[0].token_ids)
            total_samples += len(chunk_inputs)
    
    generation_time = time.time() - start_time
    tokens_per_sec = total_tokens_generated / generation_time
    
    print(f"\n✓ Predictions saved to: {config.output_path}")
    print(f"\nTiming Statistics:")
    print(f"  Total time:            {generation_time:.2f}s")
    print(f"  Samples/sec:           {total_samples/generation_time:.2f}")
    print(f"  Total tokens generated: {total_tokens_generated}")
    print(f"  Tokens/sec:            {tokens_per_sec:.0f}")
    print(f"  Avg tokens/sample:     {total_tokens_generated/total_samples:.1f}")

def evaluate_predictions(output_path):
    """Evaluate predictions using multiple metrics."""
    bertscore = evaluate.load('bertscore')
    
    predictions = []
    references = []
    
    with open(output_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            predictions.append(sample['prediction'])
            references.append(sample['label'])
    
    print("\nComputing metrics...")
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang='en')
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nBERTScore F1: {sum(bert_scores['f1'])/len(bert_scores['f1']):.4f}")
    print("="*80)
    
    return {
        'bertscore': sum(bert_scores['f1'])/len(bert_scores['f1'])
    }

def run_testing(config: Config):
    print("=" * 80)
    print("Testing Model with vLLM")
    print("=" * 80)
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Base Model: {config.model_path}")
    print(f"Mode: {'Full Fine-Tuning' if config.full_ft else 'LoRA'}")
    print("=" * 80)
    
    llm = None
    temp_dir = None
    try:
        # Load model with vLLM
        print("\n[1/2] Loading model with vLLM...")
        llm = load_model_with_vllm(config)
        temp_dir = llm._temp_dir if hasattr(llm, '_temp_dir') else None
        
        # Generate predictions
        print("\n[2/2] Generating predictions...")
        generate_and_save_predictions(llm, config)
        
        # Display sample predictions
        print("\n" + "=" * 80)
        print("SAMPLE PREDICTIONS")
        print("=" * 80)
        with open(config.output_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                pred = json.loads(line)
                print(f"\n--- Sample {i+1} ---")
                print(f"Input:\n{pred['input'][:200]}...")
                print(f"\nExpected:\n{pred['label'][:200]}...")
                print(f"\nPredicted:\n{pred['prediction'][:200]}...")
                print("-" * 80)
        
        # Unload vLLM model and clear GPU memory
        print("\n" + "=" * 80)
        print("UNLOADING VLLM MODEL")
        print("=" * 80)
        del llm
        llm = None
        gc.collect()
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
        print("=" * 80)
        
        # Evaluate
        print("\n" + "=" * 80)
        print("EVALUATING PREDICTIONS")
        print("=" * 80)
        evaluate_predictions(config.output_path)
        
        print("\n✓ Complete!")
        
    finally:
        # Clean up temporary directory
        if temp_dir is not None:
            print(f"\nCleaning up temporary model directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("✓ Cleanup complete!")

def main():
    parser = create_parser_from_dataclass(Config)
    args = parser.parse_args()
    config = Config.from_args(args)
    
    run_testing(config)

if __name__ == "__main__":
    main()
