import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
from typing import List
import sys

# Add parent directory to path to import shared modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from accelerate.utils import set_seed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from dataset_module import HFDatasetConfig, CPTDataset, prepare_datasets


@dataclass
class CPTConfig:
    """Configuration for Continual Pre-Training with Accelerate."""
    
    # Model settings
    hf_model_id: str = "Qwen/Qwen3-8B"
    model_path: str = None
    trust_remote_code: bool = True
    
    # Dataset configuration
    hf_dataset_config: HFDatasetConfig = field(default_factory=lambda: HFDatasetConfig(
        dataset_name="wikimedia/wikipedia",
        dataset_config='20231101.en',
        split="train",
        train_split_ratio=0.995,
        val_test_split_ratio=0.5,
        input_template="",
        output_template="{text}",
        field_mapping=None,
        num_proc=8
    ))
    
    # Training hyperparameters (CPT-specific defaults)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5  # Lower than SFT
    weight_decay: float = 0.01
    warmup_steps: int = 1000  # More warmup for CPT
    max_grad_norm: float = 1.0
    
    # Sequence settings
    max_seq_length: int = 4096
    
    # Paths
    data_dir: str = None
    output_dir: str = None
    resume_from_checkpoint: str = None
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 2
    
    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 1000
    max_eval_samples: int = None
    use_wandb: bool = False
    
    # Other
    seed: int = 42
    num_workers: int = 4
    use_liger_kernel: bool = False
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'CPTConfig':
        """Create CPTConfig from argparse Namespace."""
        # Build HFDatasetConfig from hfdc_ prefixed args
        hf_config_kwargs = {}
        for key, value in vars(args).items():
            if key.startswith("hfdc_") and value is not None:
                field_name = key[5:]  # Remove 'hfdc_' prefix
                if field_name == "field_mapping":
                    hf_config_kwargs[field_name] = json.loads(value) if value else None
                else:
                    hf_config_kwargs[field_name] = value
        
        # Build CPTConfig kwargs
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() 
                  if k in config_fields and not k.startswith("hfdc_") and v is not None}
        
        # Create nested HFDatasetConfig if we have args for it
        if hf_config_kwargs:
            kwargs['hf_dataset_config'] = HFDatasetConfig(**hf_config_kwargs)
        
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.model_path is None:
            self.model_path = self.hf_model_id

        if self.data_dir is None:
            dataset_name = self.hf_dataset_config.dataset_name.replace('/', '_')
            dataset_config = self.hf_dataset_config.dataset_config or 'default'
            train_pct = self.hf_dataset_config.train_split_ratio * 100
            remaining_pct = 100 - train_pct
            val_pct = remaining_pct * (1 - self.hf_dataset_config.val_test_split_ratio)
            test_pct = remaining_pct - val_pct
            self.data_dir = str(Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct:.2f}%-val={val_pct:.2f}%-test={test_pct:.2f}%")
        
        if self.output_dir is None:
            self.output_dir = str(Path.home() / f"results/{self.hf_model_id.replace('/', '_')}-cpt")


def train(config: CPTConfig):
    """Main training function for Continual Pre-Training."""
    
    set_seed(config.seed)
    
    if config.use_wandb:
        import wandb
        wandb.init(project="continual-pretraining", config=vars(config))
    
    print("=" * 80)
    print("Continual Pre-Training with Hugging Face Trainer")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Data directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Number of epochs: {config.num_train_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size per device: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Checkpoint every: {config.save_steps} steps")
    print(f"Resume from: {config.resume_from_checkpoint or 'None (fresh start)'}")
    print(f"Wandb logging: {config.use_wandb}")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    
    # Important for causal LM
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare datasets (only on rank 0)
    global_rank = int(os.environ.get("RANK", 0))
    marker_file = Path(config.data_dir) / ".data_ready"
    
    if global_rank == 0:
        prepare_datasets(config.hf_dataset_config, config.data_dir)
        marker_file.touch()
    else:
        while not marker_file.exists():
            print(f"Global Rank: {global_rank} waiting for data preparation on rank 0...")
            time.sleep(10)
    
    # Create datasets (CPTDataset: all tokens are training targets, no label masking)
    train_dataset = CPTDataset(
        Path(config.data_dir) / "training.jsonl",
        tokenizer,
        config.max_seq_length,
        is_test=False,
    )
    
    eval_dataset = CPTDataset(
        Path(config.data_dir) / "validation.jsonl",
        tokenizer,
        config.max_seq_length,
        is_test=True,
    )
    
    # Optionally limit eval samples for faster evaluation
    if config.max_eval_samples is not None:
        eval_dataset = torch.utils.data.Subset(
            eval_dataset, 
            range(min(config.max_eval_samples, len(eval_dataset)))
        )
    
    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(eval_dataset):,}")
    
    # Load model
    print(f"\nLoading model: {config.model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=True,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=["wandb", "tensorboard"] if config.use_wandb else ["tensorboard"],
        logging_dir=f"{config.output_dir}/logs",
        seed=config.seed,
        dataloader_drop_last=False,
        ddp_find_unused_parameters=False,
        use_liger_kernel=config.use_liger_kernel,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,  # Optimization for tensor cores
        label_pad_token_id=-100
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting Continual Pre-Training...")
    print("=" * 80 + "\n")
    
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    # Save final model
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"Final model saved to: {config.output_dir}/final")
    print("=" * 80 + "\n")
    
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Generate argparse parser from dataclass fields."""
    parser = argparse.ArgumentParser(
        description="Continual Pre-Training with Hugging Face Accelerate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  accelerate launch --config_file fsdp_config.yaml cpt_accelerate.py \\
    --hf_model_id Qwen/Qwen2.5-32B \\
    --data_dir datasets/medical_cpt/default/train=99%-val=1%-test=0% \\
    --output_dir results/qwen32b-medical-cpt \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate 2e-5 \\
    --max_seq_length 4096
        """
    )
    
    for f in fields(dataclass_type):
        if f.name == 'hf_dataset_config':
            # Add nested HFDatasetConfig fields with prefix
            for hf_field in fields(HFDatasetConfig):
                if hf_field.name in ['custom_converter', 'load_kwargs']:
                    continue  # Skip non-CLI fields
                arg_name = f'hfdc_{hf_field.name}'
                field_type = hf_field.type
                default_value = hf_field.default if hf_field.default is not MISSING else None
                
                if field_type == bool:
                    parser.add_argument(
                        f'--{arg_name}',
                        action='store_true' if not default_value else 'store_false',
                        default=None,
                        help=f'HFDatasetConfig: {hf_field.name}'
                    )
                elif field_type in [int, float, str]:
                    parser.add_argument(
                        f'--{arg_name}', 
                        type=field_type, 
                        default=None,
                        help=f'HFDatasetConfig: {hf_field.name}'
                    )
                else:
                    parser.add_argument(
                        f'--{arg_name}', 
                        type=str, 
                        default=None,
                        help=f'HFDatasetConfig: {hf_field.name}'
                    )
        else:
            field_type = f.type
            default_value = f.default if f.default is not MISSING else (
                f.default_factory() if f.default_factory is not MISSING else None
            )
            
            # Handle boolean fields
            if field_type == bool:
                parser.add_argument(
                    f'--{f.name}',
                    action='store_true' if not default_value else 'store_false',
                    default=None,
                    help=f'Default: {default_value}'
                )
            elif field_type in [int, float, str]:
                parser.add_argument(
                    f'--{f.name}',
                    type=field_type,
                    default=None,
                    help=f'Default: {default_value}'
                )
            elif hasattr(field_type, '__origin__') and (
                field_type.__origin__ is type(None) or 
                str(field_type).startswith('typing.Optional')
            ):
                # Handle Optional types
                inner_type = field_type.__args__[0] if hasattr(field_type, '__args__') else str
                parser.add_argument(
                    f'--{f.name}',
                    type=inner_type,
                    default=None,
                    help=f'Default: {default_value}'
                )
            else:
                # For complex types, use string
                parser.add_argument(
                    f'--{f.name}',
                    type=str,
                    default=None,
                    help=f'Default: {default_value}'
                )
    
    return parser


def parse_args():
    """Parse command line arguments."""
    parser = create_parser_from_dataclass(CPTConfig)
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = CPTConfig.from_args(args)
    train(config)


if __name__ == "__main__":
    main()
