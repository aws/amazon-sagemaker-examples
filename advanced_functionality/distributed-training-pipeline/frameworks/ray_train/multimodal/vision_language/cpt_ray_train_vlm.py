"""Vision-language model continual pre-training with Ray Train.

All tokens are training targets (no label masking). Epoch-based training
with regular checkpoint saving and resume support. For text-only CPT on
a VLM backbone, use text/cpt_accelerate.py in the accelerate framework instead.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "7200"

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass, fields, MISSING, asdict
from typing import Optional, Dict

import torch
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig
from ray.train.torch import TorchTrainer
from transformers import (
    Trainer,
    TrainingArguments,
)
from ray.train.huggingface.transformers import RayTrainReportCallback

from adapters.registry import get_adapter_for_model, print_supported_models
from base.base_dataset import VLMCPTDataset
from dataset_module import VLMDatasetConfig, prepare_vlm_datasets, get_converter_for_dataset


@dataclass
class VLMCPTConfig:
    """Configuration for VLM continual pre-training with Ray Train."""
    
    # Model
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_path: str = None
    trust_remote_code: bool = True
    
    # Vision encoder
    freeze_vision_encoder: bool = False  # Typically unfrozen for CPT
    
    # Training (CPT-specific: epoch-based, no LoRA)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # Sequence
    max_seq_length: int = 4096
    
    # Paths
    data_dir: Optional[str] = None
    results_dir: str = "results"
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
            self.data_dir = str(
                Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct:.2f}%-val={val_pct:.2f}%-test={test_pct:.2f}%"
            )
        elif self.data_dir is None:
            self.data_dir = str(Path.home() / "datasets/visual_cpt")
        
        self.results_dir = str(Path.home() / self.results_dir)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'VLMCPTConfig':
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'VLMCPTConfig':
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() if k in config_fields and v is not None}
        return cls(**kwargs)


def train_func(config_dict: Dict):
    """Training function executed on each Ray Train worker."""
    
    config = VLMCPTConfig.from_dict(config_dict)
    
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    local_rank = train.get_context().get_local_rank()
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"CPT configuration: {config_dict}")
        print(f"World size: {world_size}")
    
    # Get adapter
    adapter = get_adapter_for_model(config.model_path)
    
    # Load model and processor
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
    
    # Prepare dataset from HuggingFace if specified
    if config.hf_dataset_name is not None:
        marker_file = Path(config.data_dir) / ".data_ready"
        
        if rank == 0 and not marker_file.exists():
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
            while not marker_file.exists():
                if rank == 0:
                    break
                print(f"Rank {rank}: waiting for dataset preparation on rank 0...")
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
    
    if rank == 0:
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(eval_dataset):,}")
    
    if config.max_eval_samples is not None and len(eval_dataset) > config.max_eval_samples:
        eval_dataset.samples = eval_dataset.samples[:config.max_eval_samples]
    
    # Freeze vision encoder if requested
    if config.freeze_vision_encoder:
        adapter.freeze_vision_encoder(model)
    
    if rank == 0:
        param_counts = adapter.count_trainable_parameters(model)
        print(f"Trainable: {param_counts['trainable']:,} / {param_counts['total']:,} ({param_counts['percentage']:.2f}%)")

    # FSDP configuration
    fsdp_config = {
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        "fsdp_offload_params": False,
        "fsdp_backward_prefetch": "BACKWARD_PRE",
        "fsdp_forward_prefetch": False,
        "fsdp_use_orig_params": True,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_sync_module_states": True,
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_activation_checkpointing": True,
        "fsdp_activation_checkpointing_reentrant": False,
    }
    
    report_to = ["tensorboard", "wandb"] if config.use_wandb else ["tensorboard"]
    
    training_args = TrainingArguments(
        output_dir=config.results_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=True,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        logging_dir=f"{config.results_dir}/logs",
        seed=config.seed,
        dataloader_drop_last=False,
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
        ddp_timeout=7200,
        save_on_each_node=False,
        fsdp=["full_shard", "auto_wrap"],
        fsdp_config=fsdp_config,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[RayTrainReportCallback()],
    )
    
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    
    if rank == 0:
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    
    train.report(metrics)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VLM Continual Pre-Training with Ray Train"
    )
    
    # Model
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_path", type=str, default=None)
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
    parser.add_argument("--results_dir", type=str, default="results")
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.list_models:
        print_supported_models()
        return
    
    config = VLMCPTConfig.from_args(args)
    
    if not ray.is_initialized():
        ray.init()
    
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Ray Train VLM Continual Pre-Training")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"HF Dataset: {config.hf_dataset_name}")
    print(f"Data directory: {config.data_dir}")
    print(f"Freeze vision encoder: {config.freeze_vision_encoder}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Resume from: {config.resume_from_checkpoint or 'None (fresh start)'}")
    print("=" * 80)
    
    available_gpus = int(ray.available_resources().get("GPU", 0))
    print(f"Available GPUs: {available_gpus}")
    
    scaling_config = ScalingConfig(
        num_workers=available_gpus,
        use_gpu=True,
        resources_per_worker={"CPU": 4, "GPU": 1},
        placement_strategy="SPREAD",
    )
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config.to_dict(),
        scaling_config=scaling_config,
        torch_config=train.torch.TorchConfig(backend="nccl", timeout_s=7200),
        run_config=RunConfig(
            name=f"{config.model_id.replace('/', '-')}-vlm-cpt",
            storage_path=str(Path(config.results_dir).absolute()),
            checkpoint_config=CheckpointConfig(num_to_keep=2),
            failure_config=FailureConfig(max_failures=2),
        ),
    )
    
    result = trainer.fit()
    
    print("\n" + "=" * 80)
    print("CPT completed!")
    print(f"Results: {result}")
    if result.metrics:
        print(f"Final metrics: {result.metrics}")
    print("=" * 80)


if __name__ == "__main__":
    main()
