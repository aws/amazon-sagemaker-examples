import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import re
import time
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from accelerate.utils import set_seed
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from peft import PeftModel

from rm_dataset_module import RMDatasetConfig, RewardModelDataset, prepare_rm_datasets


def find_latest_converted_checkpoint(checkpoints_dir: str) -> str:
    """Find the latest converted checkpoint (.hf_model or .hf_peft)."""
    ckpt_dir_path = Path(checkpoints_dir)
    if not ckpt_dir_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Look for converted checkpoints
    converted_ckpts = []
    for d in ckpt_dir_path.glob("checkpoint-*"):
        if d.is_dir() and re.search(r'\.(hf_model|hf_peft)', d.name):
            # Extract checkpoint number
            match = re.match(r'checkpoint-(\d+)', d.name)
            if match:
                converted_ckpts.append((int(match.group(1)), str(d)))
    
    if not converted_ckpts:
        raise FileNotFoundError(
            f"No converted checkpoints found in {checkpoints_dir}. "
            f"Please run convert_checkpoint_to_hf.py first."
        )
    
    # Return the latest one
    converted_ckpts.sort(key=lambda x: x[0])
    return converted_ckpts[-1][1]


def is_peft_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is a PEFT adapter."""
    return checkpoint_path.endswith('.hf_peft') or Path(checkpoint_path).joinpath('adapter_config.json').exists()


class RewardDataCollator:
    """Collator for reward model training with chosen/rejected pairs."""
    
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        batch_chosen = {
            'input_ids': [f['input_ids_chosen'] for f in features],
            'attention_mask': [f['attention_mask_chosen'] for f in features],
        }
        batch_rejected = {
            'input_ids': [f['input_ids_rejected'] for f in features],
            'attention_mask': [f['attention_mask_rejected'] for f in features],
        }
        
        batch_chosen = self.tokenizer.pad(
            batch_chosen,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch_rejected = self.tokenizer.pad(
            batch_rejected,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        
        return {
            'input_ids_chosen': batch_chosen['input_ids'],
            'attention_mask_chosen': batch_chosen['attention_mask'],
            'input_ids_rejected': batch_rejected['input_ids'],
            'attention_mask_rejected': batch_rejected['attention_mask'],
        }


class RewardTrainer(Trainer):
    """Custom trainer for reward model with pairwise ranking loss."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards_chosen = model(
            input_ids=inputs['input_ids_chosen'],
            attention_mask=inputs['attention_mask_chosen'],
        ).logits
        rewards_rejected = model(
            input_ids=inputs['input_ids_rejected'],
            attention_mask=inputs['attention_mask_rejected'],
        ).logits
        
        # Pairwise ranking loss: reward_chosen should be > reward_rejected
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        
        if return_outputs:
            return loss, {'rewards_chosen': rewards_chosen, 'rewards_rejected': rewards_rejected}
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for evaluation."""
        with torch.no_grad():
            # Compute rewards for chosen and rejected
            rewards_chosen = model(
                input_ids=inputs['input_ids_chosen'],
                attention_mask=inputs['attention_mask_chosen'],
            ).logits
            rewards_rejected = model(
                input_ids=inputs['input_ids_rejected'],
                attention_mask=inputs['attention_mask_rejected'],
            ).logits
            
            # Compute loss
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        
        # Return in the format expected by Trainer
        # (loss, logits, labels) where logits can be None if prediction_loss_only=True
        if prediction_loss_only:
            return (loss, None, None)
        else:
            # Stack chosen and rejected rewards for logging/metrics
            logits = torch.stack([rewards_chosen, rewards_rejected], dim=1)
            return (loss, logits, None)


class SaveOnBestMetricCallback(TrainerCallback):
    """Save checkpoint only when metric improves."""
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        metric_value = metrics.get(args.metric_for_best_model)
        if metric_value is None:
            return control
        
        if state.best_metric is None:
            control.should_save = True
        else:
            if args.greater_is_better:
                if metric_value > state.best_metric:
                    control.should_save = True
            else:
                if metric_value < state.best_metric:
                    control.should_save = True
        
        return control


@dataclass
class RewardModelConfig:
    """Configuration for reward model training."""
    
    # Model settings
    hf_model_id: str = "Qwen/Qwen3-8B"
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
    rm_dataset_config: RMDatasetConfig = field(default_factory=lambda: RMDatasetConfig(
        dataset_name="Anthropic/hh-rlhf",
        split="train",
        train_split_ratio=0.9,
        val_test_split_ratio=0.5,
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
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    sft_model_path: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    max_eval_samples: int = 640
    use_wandb: bool = False
    
    # Other
    seed: int = 42
    num_workers: int = 8
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'RewardModelConfig':
        """Create RewardModelConfig from argparse Namespace."""
        rm_config_kwargs = {}
        for key, value in vars(args).items():
            if key.startswith("rmdc_") and value is not None:
                field_name = key[5:]
                rm_config_kwargs[field_name] = value
        
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() 
                  if k in config_fields and not k.startswith("rmdc_") and v is not None}
        
        if rm_config_kwargs:
            kwargs['rm_dataset_config'] = RMDatasetConfig(**rm_config_kwargs)
        
        if 'lora_target_modules' in kwargs and isinstance(kwargs['lora_target_modules'], str):
            kwargs['lora_target_modules'] = [m.strip() for m in kwargs['lora_target_modules'].split(',')]
        
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.sft_model_path is None:
            # Find latest converted checkpoint
            checkpoints_dir = str(Path.home() / f"results/{self.hf_model_id}")
            self.sft_model_path = find_latest_converted_checkpoint(checkpoints_dir)
            print(f"Using SFT checkpoint: {self.sft_model_path}")
        
        if self.data_dir is None:
            dataset_name = self.rm_dataset_config.dataset_name.replace('/', '_')
            dataset_config = self.rm_dataset_config.dataset_config or 'default'
            train_pct = int(self.rm_dataset_config.train_split_ratio * 100)
            remaining_pct = 100 - train_pct
            val_pct = int(remaining_pct * (1 - self.rm_dataset_config.val_test_split_ratio))
            test_pct = remaining_pct - val_pct
            self.data_dir = str(Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%")
        
        if self.output_dir is None:
            self.output_dir = str(Path.home() / f"results/reward_{self.hf_model_id}")


def train(config: RewardModelConfig):
    """Main training function."""
    
    set_seed(config.seed)
    
    if config.use_wandb:
        import wandb
        wandb.init(project="reward-model-training", config=vars(config))
    
    print("=" * 80)
    print("Reward Model Training with Hugging Face Trainer")
    print("=" * 80)
    print(f"Model: {config.hf_model_id}")
    print(f"SFT checkpoint: {config.sft_model_path or 'None (training from base)'}")
    print(f"Dataset: {config.rm_dataset_config.dataset_name}")
    print(f"LoRA enabled: {not config.full_ft}")
    print(f"Output directory: {config.output_dir}")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path or config.hf_model_id,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare datasets
    global_rank = int(os.environ.get("RANK", 0))
    marker_file = Path(config.data_dir) / ".data_ready"
    
    if global_rank == 0:
        prepare_rm_datasets(config.rm_dataset_config, config.data_dir)
        marker_file.touch()
    else:
        while not marker_file.exists():
            print(f"Global Rank: {global_rank} waiting for data preparation on rank 0...")
            time.sleep(10)
    
    # Create datasets
    train_dataset = RewardModelDataset(
        Path(config.data_dir) / "training.jsonl",
        tokenizer,
        config.max_seq_length,
    )
    
    eval_dataset = RewardModelDataset(
        Path(config.data_dir) / "validation.jsonl",
        tokenizer,
        config.max_seq_length,
    )
    
    if config.max_eval_samples is not None:
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(min(config.max_eval_samples, len(eval_dataset))))
    
    # Load model
    print(f"Loading model from: {config.sft_model_path}")
    
    # Check if it's a PEFT checkpoint
    if is_peft_checkpoint(config.sft_model_path):
        # Load base model first
        base_model_path = config.hf_model_id
        base_model_file = Path(config.sft_model_path) / "base_model.txt"
        if base_model_file.exists():
            base_model_path = base_model_file.read_text().strip()
        
        print(f"Loading base model: {base_model_path}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            num_labels=1,
        )
        
        print(f"Loading PEFT adapter from: {config.sft_model_path}")
        model = PeftModel.from_pretrained(model, config.sft_model_path)
        
        # Merge adapter weights into base model for reward training
        print("Merging PEFT adapter into base model...")
        model = model.merge_and_unload()
    else:
        # Load merged model directly
        model = AutoModelForSequenceClassification.from_pretrained(
            config.sft_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            num_labels=1,
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    
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
        dataloader_drop_last=False
    )
    
    # Data collator
    data_collator = RewardDataCollator(tokenizer, pad_to_multiple_of=8)
    
    # Trainer
    trainer = RewardTrainer(
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
    parser = argparse.ArgumentParser(description="Train Reward Model with Accelerate")
    
    for f in fields(dataclass_type):
        if f.name == 'rm_dataset_config':
            for rm_field in fields(RMDatasetConfig):
                if rm_field.name in ['custom_converter', 'load_kwargs']:
                    continue
                arg_name = f'rmdc_{rm_field.name}'
                field_type = rm_field.type
                default_value = rm_field.default if rm_field.default is not MISSING else None
                
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
                help='Comma-separated list of target modules for LoRA'
            )
        else:
            field_type = f.type
            default_value = f.default if f.default is not MISSING else (f.default_factory() if f.default_factory is not MISSING else None)
            
            if field_type == bool:
                parser.add_argument(
                    f'--{f.name}',
                    action='store_true' if not default_value else 'store_false',
                    default=None,
                )
            elif field_type in [int, float, str]:
                parser.add_argument(f'--{f.name}', type=field_type, default=None)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is type(None) or str(field_type).startswith('typing.Optional'):
                inner_type = field_type.__args__[0] if hasattr(field_type, '__args__') else str
                parser.add_argument(f'--{f.name}', type=inner_type, default=None)
            else:
                parser.add_argument(f'--{f.name}', type=str, default=None)
    
    return parser


def parse_args():
    """Parse command line arguments."""
    parser = create_parser_from_dataclass(RewardModelConfig)
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = RewardModelConfig.from_args(args)
    train(config)


if __name__ == "__main__":
    main()
