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
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from dataset_module import HFDatasetConfig, SFTDataset, prepare_datasets
from shared.callbacks import SaveOnBestMetricCallback


@dataclass
class TrainingConfig:
    """Configuration for training with Accelerate."""
    
    # Model settings
    hf_model_id: str = "Qwen/Qwen3-8B"
    model_path: str = None
    trust_remote_code: bool = True
    
    # LoRA settings
    full_ft: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ])
    
    # Dataset configuration
    hf_dataset_config: HFDatasetConfig = field(default_factory=lambda: HFDatasetConfig(
        dataset_name="cognitivecomputations/dolphin",
        dataset_config='flan1m-alpaca-uncensored',
        split="train",
        train_split_ratio=0.9,
        val_test_split_ratio=0.5,
        input_template="### Instruction:\n{instruction}\n ### Input:\n{input}\n",
        output_template="### Response:\n{output}",
        field_mapping={
            "instruction": "instruction",
            "input": "input",
            "output": "output"
        },
        num_proc=8
    ))
    
    # Training hyperparameters
    max_steps: int = 10000
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Sequence settings
    max_seq_length: int = 2048
    
    # Paths
    data_dir: str = None
    output_dir: str = None
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    max_eval_samples: int = 640
    use_wandb: bool = False
    
    # Other
    seed: int = 42
    num_workers: int = 8
    use_liger_kernel: bool = False
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """Create TrainingConfig from argparse Namespace."""
        # Build HFDatasetConfig from hfdc_ prefixed args
        hf_config_kwargs = {}
        for key, value in vars(args).items():
            if key.startswith("hfdc_") and value is not None:
                field_name = key[5:]  # Remove 'hfdc_' prefix
                if field_name == "field_mapping":
                    hf_config_kwargs[field_name] = json.loads(value)
                else:
                    hf_config_kwargs[field_name] = value
        
        # Build TrainingConfig kwargs
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() 
                  if k in config_fields and not k.startswith("hfdc_") and v is not None}
        
        # Create nested HFDatasetConfig if we have args for it
        if hf_config_kwargs:
            kwargs['hf_dataset_config'] = HFDatasetConfig(**hf_config_kwargs)
        
        # Handle lora_target_modules special case
        if 'lora_target_modules' in kwargs and isinstance(kwargs['lora_target_modules'], str):
            kwargs['lora_target_modules'] = [m.strip() for m in kwargs['lora_target_modules'].split(',')]
        
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.model_path is None:
            self.model_path = self.hf_model_id

        if self.data_dir is None:
            dataset_name = self.hf_dataset_config.dataset_name.replace('/', '_')
            dataset_config = self.hf_dataset_config.dataset_config or 'default'
            train_pct = int(self.hf_dataset_config.train_split_ratio * 100)
            remaining_pct = 100 - train_pct
            val_pct = int(remaining_pct * (1 - self.hf_dataset_config.val_test_split_ratio))
            test_pct = remaining_pct - val_pct
            self.data_dir = str(Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%")
        
        if self.output_dir is None:
            self.output_dir = str(Path.home() / f"results/{self.hf_model_id}")


def train(config: TrainingConfig):
    """Main training function."""
    
    set_seed(config.seed)
    
    if config.use_wandb:
        import wandb
        wandb.init(project="deep-learning-training", config=vars(config))
    
    print("=" * 80)
    print("Training with Hugging Face Trainer")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Dataset: {config.hf_dataset_config.dataset_name}")
    print(f"LoRA enabled: {not config.full_ft}")
    print(f"Output directory: {config.output_dir}")
    print(f"Wandb logging: {config.use_wandb}")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    
    # Important for SFT to pad correctly
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
    
    # Create datasets
    train_dataset = SFTDataset(
        Path(config.data_dir) / "training.jsonl",
        tokenizer,
        config.max_seq_length,
        is_test=False,
    )
    
    eval_dataset = SFTDataset(
        Path(config.data_dir) / "validation.jsonl",
        tokenizer,
        config.max_seq_length,
        is_test=True,
    )
    
    if config.max_eval_samples is not None:
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(min(config.max_eval_samples, len(eval_dataset))))
    
    # Load model
    print(f"Loading model: {config.model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    # Apply LoRA if enabled
    if not config.full_ft:
        print("Applying LoRA configuration...")
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_strategy="no",
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=["wandb", "tensorboard"] if config.use_wandb else ["tensorboard"],
        logging_dir=f"{config.output_dir}/logs",
        seed=config.seed,
        eval_strategy="steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=config.use_liger_kernel,
    )
    
    # Data collator
    # Using Seq2Seq collator for robust padding of input_ids AND labels
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8, # Optimization for tensor cores
        label_pad_token_id=-100
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            SaveOnBestMetricCallback(),
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience, 
                                  early_stopping_threshold=config.early_stopping_threshold),
        ],
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print("=" * 80)
    print("Training completed successfully!")
    print(f"Final checkpoint saved to {config.output_dir}")
    print("=" * 80)
    
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")

def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Generate argparse parser from dataclass fields."""
    parser = argparse.ArgumentParser(description="Train with Hugging Face Accelerate")
    
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
                    )
                elif field_type in [int, float, str]:
                    parser.add_argument(f'--{arg_name}', type=field_type, default=None)
                else:
                    parser.add_argument(f'--{arg_name}', type=str, default=None)
        elif f.name == 'lora_target_modules':
            parser.add_argument(
                f'--{f.name}',
                type=str,
                default=None,
                help='Comma-separated list of target modules for LoRA (e.g., "q_proj,k_proj,v_proj")'
            )
        else:
            field_type = f.type
            default_value = f.default if f.default is not MISSING else (f.default_factory() if f.default_factory is not MISSING else None)
            
            # Handle boolean fields
            if field_type == bool:
                parser.add_argument(
                    f'--{f.name}',
                    action='store_true' if not default_value else 'store_false',
                    default=None,
                )
            elif field_type in [int, float, str]:
                parser.add_argument(
                    f'--{f.name}',
                    type=field_type,
                    default=None,
                )
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is type(None) or str(field_type).startswith('typing.Optional'):
                # Handle Optional types
                inner_type = field_type.__args__[0] if hasattr(field_type, '__args__') else str
                parser.add_argument(
                    f'--{f.name}',
                    type=inner_type,
                    default=None,
                )
            else:
                # For complex types, use string
                parser.add_argument(
                    f'--{f.name}',
                    type=str,
                    default=None,
                )
    
    return parser


def parse_args():
    """Parse command line arguments."""
    parser = create_parser_from_dataclass(TrainingConfig)
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = TrainingConfig.from_args(args)
    train(config)


if __name__ == "__main__":
    main()
