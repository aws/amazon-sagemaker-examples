import torch
import argparse
import os
from pathlib import Path
from dataclasses import dataclass, fields, MISSING
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


@dataclass
class Config:
    # Checkpoint and Model
    model_path: str = None
    base_model: str = "Qwen/Qwen3-8B"
    checkpoints_dir: str = None
    
    # LoRA settings
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
     
    # Output
    no_merge: bool = False
    
    @property
    def output_dir(self) -> str:
        """Determine the output path based on the no_merge flag."""
        suffix = ".hf_peft" if self.no_merge else ".hf_model"
        return self.checkpoint_path.replace(".ckpt", suffix)
                                
    @property
    def checkpoint_path(self) -> str:
        """Find the latest checkpoint in the checkpoints directory."""
        ckpt_dir_path = Path(self.checkpoints_dir)
        ckpt_files = list(ckpt_dir_path.glob("model-*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir_path}")
        return str(max(ckpt_files, key=lambda p: p.stat().st_mtime))
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() if k in config_fields}
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.model_path is None:
            self.model_path = self.base_model

        if self.checkpoints_dir is None:
            self.checkpoints_dir = str(Path.home() / f"results/{self.base_model}/checkpoints")
   

def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for f in fields(dataclass_type):
        field_type = f.type
        default_value = f.default if f.default is not MISSING else None
        # Handle boolean fields
        if field_type == bool:
            parser.add_argument(
                f'--{f.name}',
                action='store_true' if not default_value else 'store_false',
                default=default_value,
            )
        else:
            parser.add_argument(
                f'--{f.name}',
                type=field_type if field_type in [int, float, str] else str,
                default=default_value,
            )
    return parser

def convert_lightning_to_hf(
    base_model_id: str,
    checkpoint_path: str,
    output_dir: str,
    merge_lora: bool = True,
    config: Config = None,
):
    """
    Convert PyTorch Lightning checkpoint to Hugging Face format.
    
    Args:
        base_model_id: Base model ID from HuggingFace
        checkpoint_path: Path to PyTorch Lightning .ckpt file
        output_dir: Directory to save the converted model
        merge_lora: If True, merge LoRA weights into base model (recommended for vLLM)
                    If False, save as separate LoRA adapter
    """
    print(f"\n{'='*80}")
    print("Converting PyTorch Lightning Checkpoint to Hugging Face Format")
    print(f"{'='*80}\n")
    
    # Load checkpoint metadata
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    state_dict_keys = list(state_dict.keys())
    
    # Determine if this is a LoRA checkpoint
    is_lora = any('lora' in key for key in state_dict_keys[:100])
    hparams = checkpoint.get('hyper_parameters', {})
    
    print(f"Checkpoint type: {'LoRA' if is_lora else 'Full Fine-tuned'}")
    print(f"Output directory: {output_dir}")
    if is_lora:
        print(f"Merge LoRA weights: {merge_lora}")
    print()
    
    # Load tokenizer
    print(f"Loading tokenizer from: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # Load base model
    print(f"Loading base model: {config.model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Remove 'model.' prefix from Lightning state dict
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    if is_lora:
        # Handle LoRA checkpoint
        print("\nApplying LoRA configuration...")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[m.strip() for m in config.lora_target_modules.split(',')],
            bias="none",
        )
        
        # Apply LoRA to base model
        model = get_peft_model(base_model, peft_config)
        
        # Load the trained weights
        print("Loading LoRA weights...")
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠ Warning: {len(missing_keys)} missing keys")
        if unexpected_keys:
            print(f"⚠ Warning: {len(unexpected_keys)} unexpected keys")
        
        if merge_lora:
            # Merge LoRA weights into base model for vLLM compatibility
            print("\nMerging LoRA weights into base model...")
            model = model.merge_and_unload()
            print("✓ LoRA weights merged")
            
            # Save merged model
            print(f"\nSaving merged model to: {output_dir}")
            model.save_pretrained(
                output_dir,
                safe_serialization=True,  # Use safetensors format
            )
            tokenizer.save_pretrained(output_dir)
            print("✓ Merged model saved")
            
        else:
            # Save as LoRA adapter (smaller, but requires PEFT-compatible inference)
            print(f"\nSaving LoRA adapter to: {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Also save the base model path for reference
            with open(f"{output_dir}/base_model.txt", "w") as f:
                f.write(base_model_id)
            
            print("✓ LoRA adapter saved")
            print(f"  Note: This requires PEFT library to load")
            print(f"  Base model: {base_model_id}")
    
    else:
        # Handle full fine-tuned model
        print("\nLoading full fine-tuned weights...")
        missing_keys, unexpected_keys = base_model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠ Warning: {len(missing_keys)} missing keys")
        if unexpected_keys:
            print(f"⚠ Warning: {len(unexpected_keys)} unexpected keys")
        
        print(f"\nSaving model to: {output_dir}")
        base_model.save_pretrained(
            output_dir,
            safe_serialization=True,  # Use safetensors format
        )
        tokenizer.save_pretrained(output_dir)
        print("✓ Model saved")
    
    
    print(f"\n{'='*80}")
    print("Conversion completed successfully!")
    print(f"{'='*80}\n")
    print(f"Model saved to: {output_dir}")
    print("\nYou can now use this model with:")
    print(f"  - Transformers: AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  - vLLM: vllm.LLM(model='{output_dir}')")
    print(f"  - Any other Hugging Face compatible framework")
    print()


def main():
    parser = create_parser_from_dataclass(Config)
    args = parser.parse_args()
    config = Config.from_args(args)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Convert
    convert_lightning_to_hf(
        base_model_id=config.model_path,
        checkpoint_path=config.checkpoint_path,
        output_dir=config.output_dir,
        merge_lora=not config.no_merge,
        config=config,
    )


if __name__ == "__main__":
    main()