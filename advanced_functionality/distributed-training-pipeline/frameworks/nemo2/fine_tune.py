import os
import argparse
from importlib import import_module
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
import nemo_run as run
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from nemo.lightning.pytorch.callbacks import (
    PytorchProfilerCallback, MegatronProgressBar, NsysCallback, ModelCheckpoint,
    RuntimeEstimator, MemoryMonitor, SpeedMonitor, MemoryProfileCallback
)
from dataset_module import GeneralizedHFDataModule, HFDatasetConfig
from nemo.collections.llm import import_ckpt
from nemo_run.run.experiment import Experiment
import nemo.lightning as nl

os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"

## Begin config
@dataclass
class Config:
    # HuggingFace Model
    model_path: str = None
    hf_model_id: str = "Qwen/Qwen3-8B"
    
    # Nemo 2.0 recipse
    recipe_cls_name: str = "qwen3_8b"

    # Paths
    data_dir: str = None
    output_dir: str = None
    nemo_ckpt_dir: str = None
    
    # Distributed Training Configuration
    num_nodes: int = 1
    gpus_per_node: int = 8
    node_rank: int = 0
    tensor_parallel_size: int = 8
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1

    # Training Hyperparameters
    max_steps: int = 10000
    val_check_interval: int = 800
    log_every_n_steps: int = 10
    micro_batch_size: int = 8
    accumulate_grad_batches: int = 8
    limit_val_batches: int = 80
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # LoRA Configuration
    peft_scheme: str = "lora"
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

    # monitoring and profiling
    enable_megatron_progress_bar: bool = False
    enable_nsys_callback: bool = False
    enable_pytorch_profiler: bool = False
    enable_speed_monitor: bool = False
    enable_memory_monitor: bool = False
    enable_runtime_estimator: bool = False
    enable_memory_profile: bool = False
    log_level = "INFO"
    use_wandb: bool = False
    
    @property
    def global_batch_size(self) -> int:
        dp_size = (self.num_nodes * self.gpus_per_node) // self.tensor_parallel_size // self.pipeline_parallel_size
        return self.micro_batch_size * self.accumulate_grad_batches * dp_size
    
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
            self.data_dir = str(Path.home() /  f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%" )
        
        if self.output_dir is None:
            self.output_dir = str(Path.home() /  f"outputs/{self.hf_model_id}" )

        if self.nemo_ckpt_dir is None:
            self.nemo_ckpt_dir = str(Path(self.output_dir) / "imported_hf_ckpt")

config = None
# End config

def import_nemo_recipe(recipe_name):
    module_path = f"nemo.collections.llm.recipes.{recipe_name}"
    try:
        recipe_module = import_module(module_path)
        return recipe_module
    except ImportError as e:
        print(f"Failed to import {module_path}: {e}")
        return None

def import_hf_ckpt(model_config: run.Config, node_rank:int):
    context_dir = os.path.join(config.nemo_ckpt_dir, 'context')
    weights_dir = os.path.join(config.nemo_ckpt_dir, 'weights')
    
    if node_rank == 0:
        if not os.path.isdir(context_dir) or not os.path.isdir(weights_dir):
            hf_source = f"hf://{config.model_path}"
            import_ckpt_partial = run.Partial(
                import_ckpt,
                model=model_config,
                source=hf_source,
                output_path=config.nemo_ckpt_dir,
                overwrite=False
            )

            run.run(import_ckpt_partial, 
                    executor=run.LocalExecutor(), 
                    name=f"{config.recipe_cls_name}_importer")
    else:
        import time
        while not os.path.isdir(context_dir) or not os.path.isdir(weights_dir):
            print(f"Waiting for {context_dir} and {weights_dir} to be imported from rank 0...")
            time.sleep(10)

def configure_data():

    return run.Config(
        GeneralizedHFDataModule,
        config=config.hf_dataset_config,
        hf_model_id=config.model_path,
        dataset_root=config.data_dir,
        seq_length=config.max_seq_length,
        micro_batch_size=config.micro_batch_size,
        global_batch_size=config.global_batch_size
    )

def configure_logger():

     # TensorBoard logger - always enabled
    tb_logger = run.Config(
        TensorBoardLogger,
        save_dir=f"tb_logs",
        name="peft_megatron",
    )
    
    # Weights & Biases logger - optional
    if config.use_wandb:
        wandb_logger = run.Config(
            WandbLogger,
            project="peft_megatron",
            save_dir=f"wandb_logs",
            log_model=False,
        )
    else:
        wandb_logger = None

    checkpoint_callback = run.Config(
        ModelCheckpoint,
        monitor="val_loss",
        mode="min",
        save_last="link",
        save_top_k=1,
        save_weights_only=True,
    )


    return run.Config(
        nl.NeMoLogger,
        name="nemo_logs",
        tensorboard=tb_logger,
        wandb=wandb_logger,
        log_dir=config.output_dir,
        ckpt=checkpoint_callback
    )

def configure_callbacks():
   
    early_stopping_callback = run.Config(
        EarlyStopping,
        monitor='val_loss',           
        min_delta=config.early_stopping_threshold,              
        patience=config.early_stopping_patience,                   
        verbose=True,                 
        mode='min',                   
        strict=True,                 
        check_finite=True,            
        stopping_threshold=None,      
        divergence_threshold=None,    
        check_on_train_epoch_end=False, 
    )
 
    callbacks = [early_stopping_callback]
    
    assert not (config.enable_pytorch_profiler and config.enable_nsys_callback), \
        "Cannot enable both PyTorch and Nsys profiling"
    
    if config.enable_megatron_progress_bar:
            megatron_progress_bar_callback = run.Config(
                MegatronProgressBar,
                refresh_rate=config.log_every_n_steps,
            )
            callbacks.append(megatron_progress_bar_callback)

    if config.enable_memory_monitor:
        memory_monitor_callback = run.Config(
            MemoryMonitor,
        )
        callbacks.append(memory_monitor_callback)
    
    if config.enable_speed_monitor:
        speed_monitor_callback = run.Config(
            SpeedMonitor,
            window_size=100,
        )
        callbacks.append(speed_monitor_callback)

    if config.enable_memory_profile:
        mem_profile_path = Path(config.output_dir) / "mem_profile"
        mem_profile_path.mkdir(parents=True, exist_ok=True)
        memory_profile_callback = run.Config(
            MemoryProfileCallback,
            dir=str(mem_profile_path), ranks=[0]
        )
        callbacks.append(memory_profile_callback)

    if config.enable_runtime_estimator:
        runtime_estimator_callback = run.Config(
            RuntimeEstimator,
        )
        callbacks.append(runtime_estimator_callback)

    if config.enable_pytorch_profiler:
        trace_path = Path(config.output_dir) / "trace"
        trace_path.mkdir(parents=True, exist_ok=True)
        pytorch_profiler_callback = run.Config(
            PytorchProfilerCallback,
            start_step=0, end_step=1, trace_dir=str(trace_path),
        )
        callbacks.append(pytorch_profiler_callback)
    
    if config.enable_nsys_callback:
        nsys_callback = run.Config(
            NsysCallback,
            start_step=0, end_step=1, ranks=[0], gen_shape=True, nvtx_ranges=False,
        )
        callbacks.append(nsys_callback)

    return callbacks

def configure_executor():
    return run.LocalExecutor(
        ntasks_per_node=config.gpus_per_node,
        nodes=config.num_nodes,
        launcher="torchrun",
        env_vars={
        }
    )

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
        else:
            field_type = f.type
            default_value = f.default if f.default is not MISSING else None
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

    nemo_recipe = import_nemo_recipe(recipe_name=config.recipe_cls_name)
    import_hf_ckpt(model_config=nemo_recipe.model(), node_rank=config.node_rank)

    nemo_recipe = nemo_recipe.finetune_recipe(
        dir=config.output_dir,
        name=config.recipe_cls_name,
        num_nodes=config.num_nodes,
        num_gpus_per_node=config.gpus_per_node,
        peft_scheme=None if config.full_ft else config.peft_scheme,
        packed_sequence=False,
    )
    nemo_recipe.data = configure_data()
    nemo_recipe.log = configure_logger()
    nemo_recipe.resume.restore_config.path = config.nemo_ckpt_dir
    nemo_recipe.trainer.max_steps = config.max_steps
    nemo_recipe.trainer.num_sanity_val_steps = 1
    nemo_recipe.trainer.val_check_interval=config.val_check_interval
    nemo_recipe.trainer.limit_val_batches=config.limit_val_batches
    nemo_recipe.trainer.accumulate_grad_batches = config.accumulate_grad_batches
    nemo_recipe.trainer.strategy.tensor_model_parallel_size=config.tensor_parallel_size
    nemo_recipe.trainer.strategy.pipeline_model_parallel_size=config.pipeline_parallel_size
    nemo_recipe.trainer.strategy.context_parallel_size=config.context_parallel_size
    nemo_recipe.trainer.callbacks.extend(configure_callbacks())
    nemo_recipe.trainer.strategy.ckpt_load_strictness = False
    nemo_recipe.tokenizer="data"

    try:
        print(f"Starting Nemo recipe {config.recipe_cls_name} fine-tuning...")
        executor = configure_executor()
        exp_title = "full_ft" if config.full_ft else f"peft_{config.peft_scheme}"
        with Experiment(title=exp_title, executor=executor, 
                        log_level=config.log_level, base_dir=config.output_dir) as exp:
            exp.add(nemo_recipe, tail_logs=True, name=config.recipe_cls_name)
            exp.run(detach=False)
        print("Fine-tuning completed successfully!")
        print(f"Outputs saved to: {config.output_dir}")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()