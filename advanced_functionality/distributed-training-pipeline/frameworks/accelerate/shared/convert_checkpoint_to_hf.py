import torch
import argparse
import os
import re
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
    
    # Training mode
    full_ft: bool = False
    
    # LoRA settings (only used when full_ft=False)
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
     
    # Output
    no_merge: bool = False
    
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
                                
    @property
    def output_dir(self) -> str:
        """Determine the output path based on the no_merge flag."""
        suffix = ".hf_peft" if self.no_merge else ".hf_model"
        return self.checkpoint_path + suffix
    
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

def convert_accelerate_to_hf(
    base_model_id: str,
    checkpoint_path: str,
    output_dir: str,
    merge_lora: bool = True,
    config: Config = None,
):
    """
    Convert Accelerate FSDP checkpoint to Hugging Face format.
    
    Args:
        base_model_id: Base model ID from HuggingFace
        checkpoint_path: Path to Accelerate checkpoint directory
        output_dir: Directory to save the converted model
        merge_lora: If True, merge LoRA weights into base model (recommended for vLLM)
                    If False, save as separate LoRA adapter (ignored if full_ft=True)
    """
    import torch.distributed.checkpoint as dcp
    
    print(f"\n{'='*80}")
    print("Converting Accelerate FSDP Checkpoint to Hugging Face Format")
    print(f"{'='*80}\n")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Full fine-tuning mode: {config.full_ft}")
    if not config.full_ft:
        print(f"Merge LoRA weights: {merge_lora}")
    print()
    
    # Load tokenizer
    print(f"Loading tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    # Load base model
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Apply LoRA config only if not full fine-tuning
    if not config.full_ft:
        print("\nApplying LoRA configuration...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[m.strip() for m in config.lora_target_modules.split(',')],
            bias="none",
        )
        model = get_peft_model(base_model, peft_config)
    else:
        model = base_model
    
    # Load FSDP checkpoint
    print("Loading FSDP checkpoint weights...")
    fsdp_checkpoint_dir = Path(checkpoint_path) / "pytorch_model_fsdp_0"
    state_dict = {}
    dcp.load(state_dict, checkpoint_id=str(fsdp_checkpoint_dir))
    model.load_state_dict(state_dict, strict=False)
    print("✓ Checkpoint loaded")
    
    if config.full_ft:
        # Full fine-tuning: save the model directly
        print(f"\nSaving full fine-tuned model to: {output_dir}")
        model.save_pretrained(
            output_dir,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_dir)
        print("✓ Model saved")
    elif merge_lora:
        # Merge LoRA weights into base model for vLLM compatibility
        print("\nMerging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("✓ LoRA weights merged")
        
        # Save merged model
        print(f"\nSaving merged model to: {output_dir}")
        model.save_pretrained(
            output_dir,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_dir)
        print("✓ Merged model saved")
    else:
        # Save as LoRA adapter
        print(f"\nSaving LoRA adapter to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        with open(f"{output_dir}/base_model.txt", "w") as f:
            f.write(base_model_id)
        
        print("✓ LoRA adapter saved")
        print(f"  Note: This requires PEFT library to load")
        print(f"  Base model: {base_model_id}")
    
    
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
    convert_accelerate_to_hf(
        base_model_id=config.model_path,
        checkpoint_path=config.checkpoint_path,
        output_dir=config.output_dir,
        merge_lora=not config.no_merge,
        config=config,
    )


if __name__ == "__main__":
    main()
