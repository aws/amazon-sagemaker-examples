import os
import argparse
from dataclasses import dataclass, field, fields, MISSING
from typing import List
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, BackwardPrefetch, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from transformers import AutoModelForCausalLM, get_cosine_with_min_lr_schedule_with_warmup
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from peft import get_peft_model, LoraConfig, TaskType

from dataset_module import GeneralizedHFDataModule, HFDatasetConfig
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## Begin config
@dataclass
class Config:
    # HuggingFace Model
    model_path: str = None
    hf_model_id: str = "Qwen/Qwen3-8B"
    
    # Paths
    data_dir: str = None
    results_dir: str = None
    
    # Distributed Training Configuration
    num_nodes: int = 1
    gpus_per_node: int = 8

    # Training Hyperparameters
    max_steps: int = 10000
    val_check_interval: int = 400
    log_every_n_steps: int = 10
    micro_batch_size: int = 2
    accumulate_grad_batches: int = 4
    limit_val_batches: int = 40
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Optimizer Configuration
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_learning_rate: float = 1e-05
    min_learning_rate: float = 1e-07
    
    # LoRA Configuration
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ])
    full_ft: bool = False
    
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

    # Sequence Configuration
    max_seq_length: int = 2048
    
    # FSDP Configuration
    cpu_offload: bool = False
    
    # wandb logging Configuration
    use_wandb: bool = False
    
    # Reproducibility
    seed: int = 42

    @property
    def num_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node
    
    @property
    def global_batch_size(self) -> int:
        return self.micro_batch_size * self.num_gpus * self.accumulate_grad_batches
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create Config from argparse Namespace, only using provided args"""
        config_fields = {f.name for f in fields(cls)}
        kwargs = {}
        
        # First, handle HFDatasetConfig parsing
        import json
        hf_config_kwargs = {}
        for hf_field in fields(HFDatasetConfig):
            arg_name = f'hfdc_{hf_field.name}'
            if hasattr(args, arg_name):
                val = getattr(args, arg_name)
                if val is not None:
                    if hf_field.name == 'field_mapping' and isinstance(val, str):
                        hf_config_kwargs[hf_field.name] = json.loads(val)
                    else:
                        hf_config_kwargs[hf_field.name] = val
        if hf_config_kwargs:
            kwargs['hf_dataset_config'] = HFDatasetConfig(**hf_config_kwargs)
        
        # Then handle other config fields
        for k, v in vars(args).items():
            if k in config_fields and k != 'hf_dataset_config' and v is not None:
                if k == 'lora_target_modules' and isinstance(v, str):
                    kwargs[k] = [m.strip() for m in v.split(',')]
                else:
                    kwargs[k] = v
        
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
        
        if self.results_dir is None:
            self.results_dir = str(Path.home() / f"results/{self.hf_model_id}")

config = None
# End config

class HFCausalLMModule(pl.LightningModule):
    """Pure PyTorch Lightning module with LoRA for causal LM fine-tuning."""

    def __init__(
        self,
        model_name: str,
        lora_config: dict,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_steps: int,
        weight_decay: float,
        max_steps: int,
        trust_remote_code: bool = True,
        dtype=torch.bfloat16,
        enable_lora: bool = True,
        activation_checkpointing: bool = True,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.lora_config = lora_config
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.enable_lora = enable_lora
        self.enable_activation_checkpointing = activation_checkpointing

        # Load base model with attention implementation for stability
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path if config else model_name,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            attn_implementation="flash_attention_2",  # Use flash attention for stability
            low_cpu_mem_usage=True,
            **model_kwargs
        )
        
        # Apply LoRA if enabled
        if enable_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config['rank'],
                lora_alpha=lora_config['alpha'],
                lora_dropout=lora_config['dropout'],
                target_modules=lora_config['target_modules'],
                bias="none",
                init_lora_weights=True,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model = self.model.to(dtype=torch.bfloat16)
            
            print("\n=== LoRA Configuration ===")
            self.model.print_trainable_parameters()
            print("=========================\n")
    
    def configure_model(self):
        """Called after FSDP wrapping to apply activation checkpointing."""
        if not self.enable_activation_checkpointing:
            return
        
        print("Applying activation checkpointing to decoder layers...")
        
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        
        def check_fn(submodule):
            return isinstance(submodule, Qwen3DecoderLayer)
        
        apply_activation_checkpointing(
            self.model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )
        print("✓ Applied FSDP activation checkpointing to decoder layers")

    def forward(self, input_ids, labels, attention_mask=None):
        """Forward pass through the model."""
        return self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=False,
        )
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        if batch_idx == 0 and self.trainer.is_global_zero:
            print(f"GPU {torch.cuda.current_device()} memory before training_step forward:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        outputs = self(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            attention_mask=batch.get('attention_mask'),
        )
        loss = outputs.loss

        if batch_idx == 0 and self.trainer.is_global_zero:
            print(f"GPU {torch.cuda.current_device()} memory after training_step forward:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if batch_idx == 0 and self.trainer.is_global_zero:
            print(f"GPU {torch.cuda.current_device()} memory before validation_step forward:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        outputs = self(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            attention_mask=batch.get('attention_mask'),
        )
        loss = outputs.loss

        # Add this check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf detected in val batch {batch_idx}")
            print(f"Input shape: {batch['input_ids'].shape}")
            print(f"Labels unique values: {torch.unique(batch['labels'])}")
            print(f"Number of non-masked labels: {(batch['labels'] != -100).sum()}")
            return None  # Skip this batch
        
        if batch_idx == 0 and self.trainer.is_global_zero:
            print(f"GPU {torch.cuda.current_device()} memory after validation_step forward:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """Manually clip gradients before optimizer step."""
        # Manual gradient clipping for FSDP + PEFT compatibility
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(
                params, 
                max_norm=1.0,
                error_if_nonfinite=True 
            )
                
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to bias and layer norm parameters
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.weight_decay,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.max_learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True,  # Use fused optimizer for better performance
        )
        
        # Cosine annealing scheduler with warmup
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
            min_lr=self.min_learning_rate,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

def configure_callbacks(config: Config):
    """Configure training callbacks."""
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=config.early_stopping_threshold,
        patience=config.early_stopping_patience,
        verbose=True,
        mode='min',
        strict=True,
        check_finite=True,
    )
    
    chkpt_filename = "model-ft-{epoch:02d}-{step}" if config.full_ft else "model-peft-lora-{epoch:02d}-{step}" 
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_last="link",
        save_top_k=1,
        dirpath=os.path.join(config.results_dir, "checkpoints"),
        filename=chkpt_filename,
        save_weights_only=False,
    )

    callbacks = [early_stopping_callback, checkpoint_callback]

    return callbacks

def configure_loggers(config: Config):
    """Configure loggers for training."""
    loggers = []
    
    # TensorBoard logger - always enabled
    tb_logger = TensorBoardLogger(
        save_dir=f"{config.results_dir}/tb_logs",
        name="peft_hf",
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger - optional
    if config.use_wandb:
        try:
            wandb_logger = WandbLogger(
                project="peft_hf",
                save_dir=f"{config.results_dir}/wandb_logs",
                log_model=False,
            )
            loggers.append(wandb_logger)
            print(f"✓ Weights & Biases logging enabled)")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize Weights & Biases logger: {e}")
            print("  Continuing with TensorBoard only...")
    
    return loggers


def configure_strategy(config: Config):
    """Configure FSDP strategy."""
    # Define auto wrap policy for FSDP
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3DecoderLayer},
    )
    
    # Mixed precision configuration
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
        cast_root_forward_inputs=True,
    )
    
    # CPU offload configuration (optional)
    cpu_offload_config = CPUOffload(offload_params=True) if config.cpu_offload else None
    
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=mixed_precision_policy,
        cpu_offload=cpu_offload_config,
        sync_module_states=True,
        forward_prefetch=False,
        limit_all_gathers=True,
        use_orig_params=True,
        activation_checkpointing_policy=None,  # We handle this in configure_model
    )
    
    return strategy


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Generate argparse parser from dataclass fields"""
    parser = argparse.ArgumentParser()
    
    for f in fields(dataclass_type):
        if f.name == 'hf_dataset_config':
            # Add nested HFDatasetConfig fields with prefix
            for hf_field in fields(HFDatasetConfig):
                if hf_field.name in ['custom_converter', 'load_kwargs']:
                    continue  # Skip non-CLI fields
                arg_name = f'hfdc_{hf_field.name}'
                field_type = hf_field.type
                if field_type in [int, float, str]:
                    parser.add_argument(f'--{arg_name}', type=field_type)
                elif field_type == bool:
                    parser.add_argument(f'--{arg_name}', type=lambda x: x.lower() == 'true')
                else:
                    parser.add_argument(f'--{arg_name}', type=str)
        elif f.name == 'lora_target_modules':
            parser.add_argument(
                f'--{f.name}',
                type=str,
                help='Comma-separated list of target modules for LoRA (e.g., "q_proj,k_proj,v_proj")'
            )
        else:
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


def main():
    global config
    parser = create_parser_from_dataclass(Config)
    args = parser.parse_args()
    config = Config.from_args(args)
    
    # Create directories
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Starting HuggingFace Model Fine-tuning with PyTorch Lightning + FSDP + LoRA")
    print("=" * 80)
    print(f"Model: {config.hf_model_id}")
    print(f"Dataset: {config.hf_dataset_config.dataset_name}")
    print(f"LoRA enabled: {not config.full_ft}")
    print(f"Global batch size: {config.global_batch_size}")
    print(f"Max steps: {config.max_steps}")
    print("=" * 80)
    
    # Initialize data module
    data_module = GeneralizedHFDataModule(
        config=config.hf_dataset_config,
        dataset_root=config.data_dir,
        tokenizer_name=config.hf_model_id,
        max_seq_length=config.max_seq_length,
        micro_batch_size=config.micro_batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
    )
    
    # Initialize model
    model = HFCausalLMModule(
        model_name=config.model_path,
        lora_config={
            'rank': config.lora_rank,
            'alpha': config.lora_alpha,
            'dropout': config.lora_dropout,
            'target_modules': config.lora_target_modules,
        },
        max_learning_rate=config.max_learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_steps=config.max_steps,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        activation_checkpointing=True,
        enable_lora=not config.full_ft,
    )
    
    # Initialize logger
    loggers = configure_loggers(config)
    
    # Set seed for reproducibility
    pl.seed_everything(config.seed, workers=True)
    
    # Initialize trainer
    trainer = pl.Trainer(
        devices=config.gpus_per_node,
        num_nodes=config.num_nodes,
        accelerator="gpu",
        strategy=configure_strategy(config),
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        limit_val_batches=config.limit_val_batches,
        gradient_clip_val=None,
        gradient_clip_algorithm=None, 
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=configure_callbacks(config),
        logger=loggers,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=False,  # Disable for performance, enable for debugging
        deterministic=False,
    )
    
    # Start training
    try:
        print("\nStarting training...")
        trainer.fit(model, datamodule=data_module)
        print("\n" + "=" * 80)
        print("Fine-tuning completed successfully!")
        print(f"Checkpoint saved to: {config.results_dir}/checkpoints")
        print(f"TensorBoard log saved to: {config.results_dir}/tb_logs")
        if config.use_wandb:
            print(f"Weights & Biases log saved to: {config.results_dir}/wandb_logs")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()