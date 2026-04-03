"""Vision-language model SFT with Ray Train + adapter pattern.

Wraps VLM fine-tuning in Ray Train's TorchTrainer for distributed training
with automatic fault tolerance and checkpoint management.
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
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from ray.train.huggingface.transformers import RayTrainReportCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from adapters.registry import get_adapter_for_model, print_supported_models
from base.base_dataset import VLMDataset
from dataset_module import VLMDatasetConfig, prepare_vlm_datasets, get_converter_for_dataset


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
class VLMConfig:
    """Configuration for VLM SFT with Ray Train."""
    
    # Model
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_path: str = None
    trust_remote_code: bool = True
    
    # Vision encoder
    freeze_vision_encoder: bool = True
    lora_on_vision: bool = False
    
    # LoRA
    full_ft: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training
    max_steps: int = 10000
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Sequence
    max_seq_length: int = 8192
    
    # Paths
    data_dir: Optional[str] = None
    results_dir: str = "results"
    
    # HuggingFace dataset
    hf_dataset_name: Optional[str] = None
    hf_dataset_config: Optional[str] = None
    hf_split: str = "train"
    hf_max_samples: Optional[int] = None
    hf_train_split_ratio: float = 0.9
    hf_val_test_split_ratio: float = 0.5
    hf_image_field: str = "image"
    hf_conversations_field: str = "conversations"
    hf_download_images: bool = True
    hf_num_proc: int = 8
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    max_eval_samples: Optional[int] = 640
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
            train_pct = int(self.hf_train_split_ratio * 100)
            remaining_pct = 100 - train_pct
            val_pct = int(remaining_pct * (1 - self.hf_val_test_split_ratio))
            test_pct = remaining_pct - val_pct
            self.data_dir = str(
                Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%"
            )
        elif self.data_dir is None:
            self.data_dir = str(Path.home() / "datasets/visual_instruct")
        
        self.results_dir = str(Path.home() / self.results_dir)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Ray Train."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'VLMConfig':
        """Reconstruct from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'VLMConfig':
        """Create from argparse Namespace."""
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() if k in config_fields and v is not None}
        return cls(**kwargs)


def train_func(config_dict: Dict):
    """Training function executed on each Ray Train worker."""
    
    config = VLMConfig.from_dict(config_dict)
    
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    local_rank = train.get_context().get_local_rank()
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Training configuration: {config_dict}")
        print(f"World size: {world_size}")
    
    # Get adapter for model
    adapter = get_adapter_for_model(config.model_path)
    
    # Load model and processor via adapter
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
            # File-based polling to avoid NCCL timeout
            while not marker_file.exists():
                if rank == 0:
                    break
                print(f"Rank {rank}: waiting for dataset preparation on rank 0...")
                time.sleep(30)
    
    # Load datasets
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
    
    if rank == 0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(eval_dataset)}")
    
    if config.max_eval_samples is not None and len(eval_dataset) > config.max_eval_samples:
        eval_dataset.samples = eval_dataset.samples[:config.max_eval_samples]
    
    # Freeze vision encoder if requested
    if config.freeze_vision_encoder:
        adapter.freeze_vision_encoder(model)
    
    # Apply LoRA if not full fine-tuning
    if not config.full_ft:
        model = prepare_model_for_kbit_training(model)
        lora_target_modules = adapter.get_lora_target_modules(
            include_vision=config.lora_on_vision
        )
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        if rank == 0:
            model.print_trainable_parameters()
    
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
    }
    
    report_to = ["tensorboard", "wandb"] if config.use_wandb else ["tensorboard"]
    
    training_args = TrainingArguments(
        output_dir=config.results_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        bf16=True,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        logging_dir=f"{config.results_dir}/logs",
        save_strategy="no",
        eval_strategy="steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=False,
        seed=config.seed,
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
        ddp_timeout=7200,
        save_on_each_node=False,
        fsdp=["full_shard", "auto_wrap"],
        fsdp_config=fsdp_config,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[
            RayTrainReportCallback(),
            SaveOnBestMetricCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            ),
        ],
    )
    
    train_result = trainer.train()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    metrics = train_result.metrics
    
    if rank == 0:
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    
    train.report(metrics)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VLM SFT with Ray Train + adapter pattern"
    )
    
    # Model
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--freeze_vision_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_vision_encoder", dest="freeze_vision_encoder",
                        action="store_false")
    
    # LoRA
    parser.add_argument("--full_ft", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_on_vision", action="store_true", default=False)
    
    # Training
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
    
    # Paths
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    
    # HuggingFace dataset
    parser.add_argument("--hf_dataset_name", type=str, default=None)
    parser.add_argument("--hf_dataset_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--hf_max_samples", type=int, default=None)
    parser.add_argument("--hf_train_split_ratio", type=float, default=0.9)
    parser.add_argument("--hf_val_test_split_ratio", type=float, default=0.5)
    parser.add_argument("--hf_image_field", type=str, default="image")
    parser.add_argument("--hf_conversations_field", type=str, default="conversations")
    parser.add_argument("--hf_download_images", action="store_true", default=True)
    parser.add_argument("--hf_num_proc", type=int, default=8)
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_eval_samples", type=int, default=640)
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
    
    config = VLMConfig.from_args(args)
    
    if not ray.is_initialized():
        ray.init()
    
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Ray Train VLM Fine-tuning with FSDP + LoRA")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"HF Dataset: {config.hf_dataset_name}")
    print(f"Data directory: {config.data_dir}")
    print(f"Freeze vision encoder: {config.freeze_vision_encoder}")
    print(f"LoRA enabled: {not config.full_ft}")
    print(f"LoRA on vision: {config.lora_on_vision}")
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
            name=f"{config.model_id.replace('/', '-')}-vlm-sft",
            storage_path=str(Path(config.results_dir).absolute()),
            checkpoint_config=CheckpointConfig(num_to_keep=2),
            failure_config=FailureConfig(max_failures=2),
        ),
    )
    
    result = trainer.fit()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Results: {result}")
    if result.metrics:
        print(f"Final metrics: {result.metrics}")
    print("=" * 80)


if __name__ == "__main__":
    main()
