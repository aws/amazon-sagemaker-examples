"""Text continual pre-training with Ray Train.

All tokens are training targets (no label masking). Epoch-based training
with regular checkpoint saving and resume support. No LoRA — full model
parameters are trained.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import json
import time
from dataclasses import dataclass, field, asdict, fields, MISSING
from typing import Dict, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig
from ray.train.torch import TorchTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from ray.train.huggingface.transformers import RayTrainReportCallback

from dataset_module import HFDatasetConfig, CPTDataset, prepare_datasets


@dataclass
class CPTConfig:
    """Configuration for Continual Pre-Training with Ray Train."""

    # Model
    model_path: str = None
    hf_model_id: str = "Qwen/Qwen3-8B"
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

    # Training hyperparameters (CPT-specific: epoch-based, no LoRA)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # Sequence
    max_seq_length: int = 4096

    # Paths
    data_dir: str = None
    results_dir: str = "results"
    resume_from_checkpoint: Optional[str] = None

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 2

    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 1000
    max_eval_samples: Optional[int] = None
    use_wandb: bool = False

    # Other
    seed: int = 42
    num_workers: int = 4
    use_liger_kernel: bool = False

    def to_dict(self):
        config_dict = asdict(self)
        config_dict['hf_dataset_config'] = asdict(self.hf_dataset_config)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'CPTConfig':
        hf_config_dict = config_dict.pop('hf_dataset_config', {})
        hf_config = HFDatasetConfig(**hf_config_dict)
        return cls(hf_dataset_config=hf_config, **config_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'CPTConfig':
        hf_config_kwargs = {}
        for key, value in vars(args).items():
            if key.startswith("hfdc_") and value is not None:
                field_name = key[5:]
                if field_name == "field_mapping":
                    hf_config_kwargs[field_name] = json.loads(value) if value else None
                else:
                    hf_config_kwargs[field_name] = value

        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items()
                  if k in config_fields and not k.startswith("hfdc_") and v is not None}

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
            self.data_dir = str(
                Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct:.2f}%-val={val_pct:.2f}%-test={test_pct:.2f}%"
            )

        self.results_dir = str(Path.home() / self.results_dir)


def train_func(config_dict: Dict):
    """Training function executed on each Ray Train worker."""

    config = CPTConfig.from_dict(config_dict)

    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    local_rank = train.get_context().get_local_rank()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"CPT configuration: {config_dict}")
        print(f"World size: {world_size}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )

    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    marker_file = Path(config.data_dir) / ".data_ready"

    if rank == 0:
        if not marker_file.exists():
            prepare_datasets(config.hf_dataset_config, config.data_dir)
    else:
        while not marker_file.exists():
            print(f"Rank {rank}: waiting for dataset preparation on rank 0...")
            time.sleep(10)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

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

    if rank == 0:
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(eval_dataset):,}")

    if config.max_eval_samples is not None:
        eval_dataset = torch.utils.data.Subset(
            eval_dataset,
            range(min(config.max_eval_samples, len(eval_dataset)))
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    fsdp_config = {
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
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
        use_liger_kernel=config.use_liger_kernel,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continual Pre-Training with Ray Train")

    for f in fields(dataclass_type):
        if f.name == 'hf_dataset_config':
            for hf_field in fields(HFDatasetConfig):
                if hf_field.name in ['custom_converter', 'load_kwargs']:
                    continue
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
        else:
            field_type = f.type
            default_value = f.default if f.default is not MISSING else (
                f.default_factory() if f.default_factory is not MISSING else None
            )

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
            elif hasattr(field_type, '__origin__') and (
                field_type.__origin__ is type(None) or
                str(field_type).startswith('typing.Optional')
            ):
                inner_type = field_type.__args__[0] if hasattr(field_type, '__args__') else str
                parser.add_argument(
                    f'--{f.name}',
                    type=inner_type,
                    default=None,
                )
            else:
                parser.add_argument(
                    f'--{f.name}',
                    type=str,
                    default=None,
                )

    return parser


def parse_args():
    parser = create_parser_from_dataclass(CPTConfig)
    return parser.parse_args()


def main():
    args = parse_args()
    config = CPTConfig.from_args(args)

    if not ray.is_initialized():
        ray.init()

    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Ray Train Continual Pre-Training")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Dataset: {config.hf_dataset_config.dataset_name}")
    print(f"Data directory: {config.data_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Checkpoint every: {config.save_steps} steps")
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
            name=f"{config.hf_model_id.replace('/', '-')}-cpt",
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
