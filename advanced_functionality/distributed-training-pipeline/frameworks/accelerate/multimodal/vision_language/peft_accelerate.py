"""Vision-language model training with adapter pattern."""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Increase timeout for dataset preparation (2 hours for large datasets)
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "7200"

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate.utils import set_seed
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.callbacks import SaveOnBestMetricCallback

# Import adapter components
from adapters.registry import get_adapter_for_model, list_supported_models, print_supported_models
from base.base_dataset import VLMDataset
from dataset_module import VLMDatasetConfig, prepare_vlm_datasets, get_converter_for_dataset


@dataclass
class VLMTrainingConfig:
    """Configuration for vision-language model training."""
    
    # Model settings
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_path: str = None
    trust_remote_code: bool = True
    
    # Vision encoder settings
    freeze_vision_encoder: bool = True
    lora_on_vision: bool = False
    
    # LoRA settings
    full_ft: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training hyperparameters
    max_steps: int = 10000
    per_device_train_batch_size: int = 1  # Must be 1 for dynamic resolution
    per_device_eval_batch_size: int = 1  # Must be 1 for dynamic resolution
    gradient_accumulation_steps: int = 16  # Increased to compensate for batch_size=1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Sequence settings
    max_seq_length: int = 8192
    
    # Paths
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    # HuggingFace dataset configuration (optional - for automatic dataset preparation)
    hf_dataset_name: Optional[str] = None  # e.g., "lmms-lab/LLaVA-NeXT-Data"
    hf_dataset_config: Optional[str] = None
    hf_split: str = "train"
    hf_max_samples: Optional[int] = None  # Limit dataset size for testing (e.g., 10000)
    hf_train_split_ratio: float = 0.9
    hf_val_test_split_ratio: float = 0.5
    hf_image_field: str = "image"
    hf_conversations_field: str = "conversations"
    hf_download_images: bool = True  # Save PIL images and download URLs
    hf_num_proc: int = 8
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    max_eval_samples: Optional[int] = 640
    use_wandb: bool = False
    
    # Other
    seed: int = 42
    num_workers: int = 4
    
    def __post_init__(self):
        """Auto-derive data_dir and output_dir if not specified."""
        if self.model_path is None:
            self.model_path = self.model_id

        # Auto-derive data_dir from HF dataset name
        if self.data_dir is None and self.hf_dataset_name is not None:
            dataset_name = self.hf_dataset_name.replace('/', '_')
            dataset_config = self.hf_dataset_config or 'default'
            train_pct = int(self.hf_train_split_ratio * 100)
            remaining_pct = 100 - train_pct
            val_pct = int(remaining_pct * (1 - self.hf_val_test_split_ratio))
            test_pct = remaining_pct - val_pct
            self.data_dir = str(Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%")
        elif self.data_dir is None:
            # Fallback if no HF dataset specified
            self.data_dir = str(Path.home() / "datasets/visual_instruct")
        
        # Auto-derive output_dir from model_id
        if self.output_dir is None:
            self.output_dir = str(Path.home() / f"results/{self.model_id}")


def train(config: VLMTrainingConfig):
    """Main training function."""
    
    set_seed(config.seed)
    
    if config.use_wandb:
        import wandb
        wandb.init(project="vlm-training", config=vars(config))
    
    print("=" * 80)
    print("Vision-Language Model Training with Adapter Pattern")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"HF Dataset: {config.hf_dataset_name}")
    print(f"Data directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Freeze vision encoder: {config.freeze_vision_encoder}")
    print(f"LoRA enabled: {not config.full_ft}")
    print(f"LoRA on vision: {config.lora_on_vision}")
    print("=" * 80)
    
    # Get appropriate adapter for model
    print(f"\nDetecting adapter for model: {config.model_path}")
    adapter = get_adapter_for_model(config.model_path)
    
    print(f"\nAdapter info:")
    adapter_info = adapter.get_model_info()
    for key, value in adapter_info.items():
        print(f"  {key}: {value}")
    
    # Load model and processor using adapter
    print(f"\nLoading model...")
    model = adapter.load_model(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        use_cache=False
    )
    
    processor = adapter.load_processor(config.model_path)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Freeze vision encoder if requested
    if config.freeze_vision_encoder:
        print("\nFreezing vision encoder...")
        adapter.freeze_vision_encoder(model)
    
    # Print parameter counts before LoRA
    param_counts = adapter.count_trainable_parameters(model)
    print(f"\nParameters before LoRA:")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Percentage: {param_counts['percentage']:.2f}%")
    
    # Apply LoRA if not full fine-tuning
    if not config.full_ft:
        print("\nApplying LoRA...")
        
        # Get LoRA target modules from adapter
        lora_target_modules = adapter.get_lora_target_modules(
            include_vision=config.lora_on_vision
        )
        
        print(f"LoRA target modules: {lora_target_modules}")
        
        # Prepare model for k-bit training (even without quantization, this helps)
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Create datasets using adapter
    print(f"\nLoading datasets from {config.data_dir}")
    
    # Prepare dataset from HuggingFace if specified
    if config.hf_dataset_name is not None:
        import time
        marker_file = Path(config.data_dir) / ".data_ready"
        global_rank = int(os.environ.get("RANK", 0))
        
        if global_rank == 0 and not marker_file.exists():
            print(f"\nPreparing dataset from HuggingFace: {config.hf_dataset_name}")
            print(f"⚠️  This is a large dataset (700K+ samples) and will take 2-3 hours to prepare.")
            print(f"⚠️  Images are being saved to disk. This only needs to be done once.")
            print(f"⚠️  Other GPU ranks are waiting. Please be patient...")
            print()
            # Get pre-built converter if available
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
            print(f"Dataset prepared successfully!")
        else:
            # Non-rank-0 processes poll for marker file (avoids NCCL timeout)
            while not marker_file.exists():
                print(f"Rank {global_rank}: waiting for dataset preparation on rank 0...")
                time.sleep(30)
    
    train_dataset = VLMDataset(
        data_path=Path(config.data_dir) / "training.jsonl",
        adapter=adapter,
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        is_test=False
    )
    
    eval_dataset = VLMDataset(
        data_path=Path(config.data_dir) / "validation.jsonl",
        adapter=adapter,
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        is_test=True
    )
    
    # Limit eval samples if specified
    if config.max_eval_samples is not None and len(eval_dataset) > config.max_eval_samples:
        eval_dataset.samples = eval_dataset.samples[:config.max_eval_samples]
        print(f"Limited eval dataset to {config.max_eval_samples} samples")
    
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
        save_strategy="no",  # SaveOnBestMetricCallback handles saving
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
    )
    
    # Trainer (no custom data collator needed with batch_size=1)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            SaveOnBestMetricCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            ),
        ],
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    
    final_dir = f"{config.output_dir}/final"
    print(f"\nSaving final model to {final_dir}")
    
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    
    print(f"\n✓ Model saved to {final_dir}")
    print(f"✓ Processor saved to {final_dir}")
    
    # Print final stats
    param_counts = adapter.count_trainable_parameters(model)
    print(f"\nFinal model statistics:")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Percentage trainable: {param_counts['percentage']:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Train vision-language models with adapter pattern"
    )
    
    # Model settings
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                       help="HuggingFace model ID")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Local model path (overrides model_id for loading)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--freeze_vision_encoder", action="store_true", default=True,
                       help="Freeze vision encoder parameters")
    parser.add_argument("--no_freeze_vision_encoder", dest="freeze_vision_encoder",
                       action="store_false")
    
    # LoRA settings
    parser.add_argument("--full_ft", action="store_true", default=False,
                       help="Full fine-tuning (disable LoRA)")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_on_vision", action="store_true", default=False,
                       help="Apply LoRA to vision encoder")
    
    # Training settings
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="Batch size per device (must be 1 for dynamic resolution)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                       help="Eval batch size per device (must be 1 for dynamic resolution)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
    
    # Paths
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing training.jsonl and validation.jsonl (auto-derived if using HF dataset)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-derived from model_id if not specified)")
    
    # HuggingFace dataset configuration
    parser.add_argument("--hf_dataset_name", type=str, default=None,
                       help="HuggingFace dataset name (e.g., 'lmms-lab/LLaVA-NeXT-Data')")
    parser.add_argument("--hf_dataset_config", type=str, default=None,
                       help="HuggingFace dataset config/subset")
    parser.add_argument("--hf_split", type=str, default="train",
                       help="Initial split to load from HF dataset")
    parser.add_argument("--hf_max_samples", type=int, default=None,
                       help="Limit dataset size for testing (e.g., 10000 for quick test)")
    parser.add_argument("--hf_train_split_ratio", type=float, default=0.9,
                       help="Ratio of data for training")
    parser.add_argument("--hf_val_test_split_ratio", type=float, default=0.5,
                       help="Ratio of remaining data for validation vs test")
    parser.add_argument("--hf_image_field", type=str, default="image",
                       help="Field name for image in HF dataset")
    parser.add_argument("--hf_conversations_field", type=str, default="conversations",
                       help="Field name for conversations in HF dataset")
    parser.add_argument("--hf_download_images", action="store_true", default=True,
                       help="Save PIL images and download images from URLs in HF dataset")
    parser.add_argument("--hf_num_proc", type=int, default=8,
                       help="Number of processes for HF dataset loading")
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_eval_samples", type=int, default=640)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Utility commands
    parser.add_argument("--list_models", action="store_true",
                       help="List all supported models and exit")
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_models:
        print_supported_models()
        return
    
    # Create config from args
    config = VLMTrainingConfig(
        model_id=args.model_id,
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        freeze_vision_encoder=args.freeze_vision_encoder,
        lora_on_vision=args.lora_on_vision,
        full_ft=args.full_ft,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        max_seq_length=args.max_seq_length,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
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
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        max_eval_samples=args.max_eval_samples,
        use_wandb=args.use_wandb,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
