import os
import argparse
import json
import time
import shutil
import tempfile
import gc
from dataclasses import dataclass, fields, MISSING
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
from vllm import LLM, SamplingParams
import ray

@dataclass
class Config:
    # Checkpoint and Model
    model_path: str = None
    base_model: str = "Qwen/Qwen3-8B"
    checkpoints_dir: str = None
    full_ft: bool = False
    
    # Data
    test_path: str = None
    max_samples: int = 1024
    
    # Generation settings
    temperature: float = 0.1
    top_k: int = -1
    top_p: float = 0.95
    max_tokens: int = 512
    batch_size: int = 128
    
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
        
        # Look for checkpoint_* directories, potentially nested under TorchTrainer_* dirs
        ckpt_dirs = []
        
        # First, check if there are TorchTrainer_* directories (Ray Train structure)
        trainer_dirs = list(ckpt_dir_path.glob("TorchTrainer_*"))
        if trainer_dirs:
            # Sort trainer directories by modification time and get the latest
            latest_trainer_dir = max(trainer_dirs, key=lambda p: p.stat().st_mtime)
            # Look inside the latest trainer directory for checkpoint_* directories
            ckpt_dirs = [d for d in latest_trainer_dir.glob("checkpoint_*") if d.is_dir()]
        else:
            # Fallback: look directly for checkpoint_* directories
            ckpt_dirs = [d for d in ckpt_dir_path.glob("checkpoint_*") if d.is_dir()]
        
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoint directories found in {ckpt_dir_path}")
        
        # Sort by modification time and get the latest checkpoint
        latest_ckpt = max(ckpt_dirs, key=lambda p: p.stat().st_mtime)
        
        # Return the checkpoint subdirectory inside the Ray checkpoint
        return str(latest_ckpt / "checkpoint")
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() if k in config_fields}
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.model_path is None:
            self.model_path = self.base_model

        if self.checkpoints_dir is None:
            self.checkpoints_dir = str(Path.home() / f"results/{self.base_model.replace('/', '-')}")
        
        if self.test_path is None:
            datasets_path = Path.home() / "datasets"
            test_files = list(datasets_path.rglob("test.jsonl"))
            if test_files:
                self.test_path = str(max(test_files, key=lambda p: p.stat().st_mtime))
                print(f"Found test file: {self.test_path}")
            else:
                raise ValueError("No test.jsonl file found under ~/datasets folder")
    
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
    """Load Ray Train checkpoint and merge LoRA if needed."""
    
    print("=" * 80)
    print("LOADING RAY TRAIN CHECKPOINT INTO MEMORY")
    print("=" * 80)
    print(f"Source: {config.checkpoint_path}")
    print(f"Mode: {'Full Fine-Tuning' if config.full_ft else 'LoRA (will merge adapters)'}")
    print("=" * 80)
    
    checkpoint_path = config.checkpoint_path
    model_to_load = config.model_path
    
    if config.full_ft:
        print("\nLoading fully fine-tuned model...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        print(f"\nLoading base model: {model_to_load}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("\nLoading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        print("\nMerging LoRA adapters...")
        model = model.merge_and_unload()
    
    print("\n✓ Checkpoint loaded and ready for inference!")
    print("=" * 80)
    
    return model

def load_model_with_vllm(config: Config):
    """Load Ray Train checkpoint and initialize vLLM."""
    
    model = load_checkpoint_in_memory(config)
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
        
        print(f"\nSaving model to temporary directory: {temp_dir}")
        model.save_pretrained(temp_dir)
        
        # Load tokenizer from checkpoint or base model
        checkpoint_path = config.checkpoint_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        tokenizer.save_pretrained(temp_dir)
        
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
        
        llm._temp_dir = temp_dir
        return llm
        
    except Exception as e:
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
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nBERTScore F1: {sum(bert_scores['f1'])/len(bert_scores['f1']):.4f}")
    print("="*80)
    
    return {
        'bertscore': sum(bert_scores['f1'])/len(bert_scores['f1'])
    }

def run_testing(config: Config):
    print("=" * 80)
    print("Testing Ray Train Checkpoint with vLLM")
    print("=" * 80)
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Base Model: {config.model_path}")
    print(f"Mode: {'Full Fine-Tuning' if config.full_ft else 'LoRA'}")
    print("=" * 80)
    
    llm = None
    temp_dir = None
    try:
        print("\n[1/2] Loading model with vLLM...")
        llm = load_model_with_vllm(config)
        temp_dir = llm._temp_dir if hasattr(llm, '_temp_dir') else None
        
        print("\n[2/2] Generating predictions...")
        generate_and_save_predictions(llm, config)
        
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
        
        print("\n" + "=" * 80)
        print("UNLOADING VLLM MODEL")
        print("=" * 80)
        del llm
        llm = None
        gc.collect()
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
        print("=" * 80)
        
        print("\n" + "=" * 80)
        print("EVALUATING PREDICTIONS")
        print("=" * 80)
        evaluate_predictions(config.output_path)
        
        print("\n✓ Complete!")
        
    finally:
        if temp_dir is not None:
            print(f"\nCleaning up temporary model directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("✓ Cleanup complete!")

def main():
    parser = create_parser_from_dataclass(Config)
    args = parser.parse_args()
    config = Config.from_args(args)

    # Initialize Ray with runtime_env
    if not ray.is_initialized():
        ray.init()

    run_testing(config)

if __name__ == "__main__":
    main()
