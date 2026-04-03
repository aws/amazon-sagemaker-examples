"""Vision-language model continual pre-training with adapter pattern.

All tokens are training targets (no label masking). Epoch-based training
with regular checkpoint saving and resume support. For text-only CPT on
a VLM backbone, use text/cpt_accelerate.py instead.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "7200"

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from accelerate.utils import set_seed
from transformers import (
    Trainer,
    TrainingArguments,
)

# Import adapter components
from adapters.registry import get_adapter_for_model, print_supported_models
from base.base_dataset import VLMCPTDataset
from dataset_module import VLMDatasetConfig, prepare_vlm_datasets, get_converter_for_dataset


@dataclass
class VLMCPTConfig:
    """Configuration for vision-language continual pre-training."""

    # Model
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_path: str = None
    trust_remote_code: bool = True

    # Vision encoder
    freeze_vision_encoder: bool = False  # Typically unfrozen for CPT

    # Training hyperparameters (CPT-specific defaults)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Must be 1 for dynamic resolution
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Sequence
    max_seq_length: int = 4096

    # Paths
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    # HuggingFace dataset
    hf_dataset_name: Optional[str] = None
    hf_dataset_config: Optional[str] = None
    hf_split: str = "train"
    hf_max_samples: Optional[int] = None
    hf_train_split_ratio: float = 0.99
    hf_val_test_split_ratio: float = 0.5
    hf_image_field: str = "image"
    hf_conversations_field: str = "conversations"
    hf_download_images: bool = True
    hf_num_proc: int = 8

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 2

    # Logging
    logging_steps: int = 10
    eval_steps: int = 1000
    max_eval_samples: Optional[int] = None
    use_wandb: bool = False

    # Other
    seed: int = 42
    num_workers: int = 4

    def __post_init__(self):
        if self.model_path is None:
            self.model_path = self.model_id

        if self.data_dir is None and self.hf_dataset_name is not None:
            dataset_name = self.hf_dataset_name.replace('/', '_')
            dataset_config = self.hf_dataset_config or 'default'
            train_pct = self.hf_train_split_ratio * 100
            remaining_pct = 100 - train_pct
            val_pct = remaining_pct * (1 - self.hf_val_test_split_ratio)
            test_pct = remaining_pct - val_pct
            self.data_dir = str(Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct:.2f}%-val={val_pct:.2f}%-test={test_pct:.2f}%")
        elif self.data_dir is None:
            self.data_dir = str(Path.home() / "datasets/visual_cpt")

        if self.output_dir is None:
            self.output_dir = str(Path.home() / f"results/{self.model_id}-cpt")


def train(config: VLMCPTConfig):
    """Main training function for VLM continual pre-training."""

    set_seed(config.seed)

    if config.use_wandb:
        import wandb
        wandb.init(project="vlm-cpt", config=vars(config))

    print("=" * 80)
    print("Vision-Language Model Continual Pre-Training")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"HF Dataset: {config.hf_dataset_name}")
    print(f"Data directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Freeze vision encoder: {config.freeze_vision_encoder}")
    print(f"Number of epochs: {config.num_train_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Checkpoint every: {config.save_steps} steps")
    print(f"Resume from: {config.resume_from_checkpoint or 'None (fresh start)'}")
    print("=" * 80)

    # Get adapter
    adapter = get_adapter_for_model(config.model_path)
    adapter_info = adapter.get_model_info()
    print(f"\nAdapter info:")
    for key, value in adapter_info.items():
        print(f"  {key}: {value}")

    # Load model and processor via adapter
    print(f"\nLoading model...")
    model = adapter.load_model(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        use_cache=False
    )

    processor = adapter.load_processor(config.model_path)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Freeze vision encoder if requested
    if config.freeze_vision_encoder:
        print("\nFreezing vision encoder...")
        adapter.freeze_vision_encoder(model)

    param_counts = adapter.count_trainable_parameters(model)
    print(f"\nTrainable parameters: {param_counts['trainable']:,} / {param_counts['total']:,} ({param_counts['percentage']:.2f}%)")

    # Prepare dataset from HuggingFace if specified
    if config.hf_dataset_name is not None:
        import time
        marker_file = Path(config.data_dir) / ".data_ready"
        global_rank = int(os.environ.get("RANK", 0))

        if global_rank == 0 and not marker_file.exists():
            print(f"\nPreparing dataset from HuggingFace: {config.hf_dataset_name}")
            print(f"⚠️  This may take hours for large datasets. Other ranks are waiting...")
            custom_converter = get_converter_for_dataset(config.hf_dataset_name)

            vlm_dataset_config = VLMDatasetConfig(
                dataset_name=config.hf_dataset_name,
                dataset_config=config.hf_dataset_config,
                split=config.hf_split,
                max_samples=config.hf_max_samples,
                train_split_ratio=config.hf_train_split_ratio,
                val_test_split_ratio=config.hf_val_test_split_ratio,
                image_field=config.hf_image_field,
                conversations_field=config.hf_conversations_field,
                download_images=config.hf_download_images,
                num_proc=config.hf_num_proc,
                custom_converter=custom_converter,
            )

            prepare_vlm_datasets(vlm_dataset_config, config.data_dir)
            print("Dataset prepared successfully!")
        else:
            # Non-rank-0 processes poll for marker file (avoids NCCL timeout)
            while not marker_file.exists():
                print(f"Rank {global_rank}: waiting for dataset preparation on rank 0...")
                time.sleep(30)

    # Load datasets (VLMCPTDataset: all tokens are training targets)
    train_dataset = VLMCPTDataset(
        data_path=Path(config.data_dir) / "training.jsonl",
        adapter=adapter,
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        is_test=False
    )

    eval_dataset = VLMCPTDataset(
        data_path=Path(config.data_dir) / "validation.jsonl",
        adapter=adapter,
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        is_test=True
    )

    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(eval_dataset):,}")

    if config.max_eval_samples is not None and len(eval_dataset) > config.max_eval_samples:
        eval_dataset.samples = eval_dataset.samples[:config.max_eval_samples]

    # Training arguments (epoch-based, regular checkpointing)
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
    )

    # Trainer (no data collator needed with batch_size=1)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting VLM Continual Pre-Training...")
    print("=" * 80 + "\n")

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Save final model
    final_dir = f"{config.output_dir}/final"
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Saving final model to: {final_dir}")
    print("=" * 80 + "\n")

    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Continual pre-training for vision-language models"
    )

    # Model
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local model path (overrides model_id for loading)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--freeze_vision_encoder", action="store_true", default=False)
    parser.add_argument("--no_freeze_vision_encoder", dest="freeze_vision_encoder",
                        action="store_false")

    # Training
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_seq_length", type=int, default=4096)

    # Paths
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # HuggingFace dataset
    parser.add_argument("--hf_dataset_name", type=str, default=None)
    parser.add_argument("--hf_dataset_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--hf_max_samples", type=int, default=None)
    parser.add_argument("--hf_train_split_ratio", type=float, default=0.99)
    parser.add_argument("--hf_val_test_split_ratio", type=float, default=0.5)
    parser.add_argument("--hf_image_field", type=str, default="image")
    parser.add_argument("--hf_conversations_field", type=str, default="conversations")
    parser.add_argument("--hf_download_images", action="store_true", default=True)
    parser.add_argument("--hf_num_proc", type=int, default=8)

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    # Utility
    parser.add_argument("--list_models", action="store_true",
                        help="List all supported models and exit")

    args = parser.parse_args()

    if args.list_models:
        print_supported_models()
        return

    config = VLMCPTConfig(
        model_id=args.model_id,
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        freeze_vision_encoder=args.freeze_vision_encoder,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        max_seq_length=args.max_seq_length,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_split=args.hf_split,
        hf_max_samples=args.hf_max_samples,
        hf_train_split_ratio=args.hf_train_split_ratio,
        hf_val_test_split_ratio=args.hf_val_test_split_ratio,
        hf_image_field=args.hf_image_field,
        hf_conversations_field=args.hf_conversations_field,
        hf_download_images=args.hf_download_images,
        hf_num_proc=args.hf_num_proc,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        max_eval_samples=args.max_eval_samples,
        use_wandb=args.use_wandb,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    train(config)


if __name__ == "__main__":
    main()
