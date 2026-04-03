import os
import sys
import argparse
import json
from dataclasses import dataclass, field, asdict, fields, MISSING
from typing import Dict
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
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from ray.train.huggingface.transformers import RayTrainReportCallback
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from dataset_module import HFDatasetConfig, load_and_prepare_datasets

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


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
class Config:
    # Model
    model_path: str = None
    hf_model_id: str = "Qwen/Qwen3-8B"
    
    # Paths
    data_dir: str = None
    results_dir: str = "results"
    
    # Training
    max_steps: int = 10000
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    full_ft: bool = False
    
    # Dataset
    hf_dataset_config: HFDatasetConfig = field(default_factory=lambda: HFDatasetConfig(
        dataset_name="cognitivecomputations/dolphin",
        dataset_config='flan1m-alpaca-uncensored',
        split="train",
        train_split_ratio=0.9,
        val_test_split_ratio=0.5,
        input_template="### Instruction:\\n{instruction}\\n ### Input:\\n{input}\\n",
        output_template="### Response:\\n{output}",
        field_mapping={
            "instruction": "instruction",
            "input": "input",
            "output": "output"
        },
        num_proc=8
    ))
    
    # Sequence
    max_seq_length: int = 2048
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    max_eval_samples: int = 640
    use_wandb: bool = False
    
    # Other settings
    seed: int = 16257
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    logging_dir: str = None
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    use_liger_kernel: bool = False

    def to_dict(self):
        """Convert to dictionary for Ray Train."""
        config_dict = asdict(self)
        config_dict['hf_dataset_config'] = asdict(self.hf_dataset_config)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Reconstruct Config from dictionary."""
        hf_config_dict = config_dict.pop('hf_dataset_config', {})
        hf_config = HFDatasetConfig(**hf_config_dict)
        return cls(hf_dataset_config=hf_config, **config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create Config from argparse Namespace."""
        hf_config_kwargs = {}
        for key, value in vars(args).items():
            if key.startswith("hfdc_") and value is not None:
                field_name = key[5:]
                if field_name == "field_mapping":
                    hf_config_kwargs[field_name] = json.loads(value)
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
            train_pct = int(self.hf_dataset_config.train_split_ratio * 100)
            remaining_pct = 100 - train_pct
            val_pct = int(remaining_pct * (1 - self.hf_dataset_config.val_test_split_ratio))
            test_pct = remaining_pct - val_pct
            self.data_dir = str(Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%")
        
        self.results_dir = str(Path.home() / self.results_dir)
        
        if self.logging_dir is None:
            self.logging_dir = f"{self.results_dir}/logs"


def train_func(config_dict: Dict):
    """Training function executed on each worker."""  
    
    config = Config.from_dict(config_dict)
    
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    local_rank = train.get_context().get_local_rank()
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Training configuration: {config_dict}")
        print(f"World size: {world_size}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    train_dataset, eval_dataset = load_and_prepare_datasets(
        config=config.hf_dataset_config,
        dataset_root=config.data_dir,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        rank=rank,
    )
    
    if rank == 0:
        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Validation samples: {len(eval_dataset)}")
    
    if config.max_eval_samples is not None and eval_dataset:
        eval_dataset = torch.utils.data.Subset(
            eval_dataset, 
            range(min(config.max_eval_samples, len(eval_dataset)))
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    if not config.full_ft:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[m.strip() for m in config.lora_target_modules.split(',')],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        if rank == 0:
            model.print_trainable_parameters()
    
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
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        save_total_limit=config.save_total_limit,
        bf16=True,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
        report_to=report_to,
        logging_dir=config.logging_dir,
        save_strategy="no",
        eval_strategy="steps" if eval_dataset else "no",
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
        use_liger_kernel=config.use_liger_kernel,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Generate argparse parser from dataclass fields."""
    parser = argparse.ArgumentParser(description="Train with Ray Train")
    
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
                parser.add_argument(
                    f'--{f.name}',
                    type=field_type,
                    default=None,
                )
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is type(None) or str(field_type).startswith('typing.Optional'):
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
    parser = create_parser_from_dataclass(Config)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.from_args(args)
    
    if not ray.is_initialized():
        ray.init()
    
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Ray Train Fine-tuning with FSDP + LoRA")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Dataset: {config.hf_dataset_config.dataset_name}")
    print(f"LoRA enabled: {not config.full_ft}")
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
            name=f"{config.hf_model_id.replace('/', '-')}",
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
