import argparse
from dataclasses import dataclass, fields, MISSING
import torch
import json
from typing import  Optional
from pathlib import Path
import re
import evaluate
import nemo.lightning as nl
import lightning.pytorch as pl
from nemo.lightning import io
from megatron.core import parallel_state
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext, ContextOverflowError
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from nemo.collections.llm.inference import MCoreTokenizerWrappper
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir, ADAPTER_META_FILENAME
from nemo.collections.llm.modelopt import set_modelopt_spec_if_exists_in_ckpt
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.lightning.pytorch.callbacks import PEFT
from lightning.pytorch.trainer.states import TrainerFn
from megatron.core.transformer.module import MegatronModule
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from typing import List
import time
from tqdm import tqdm

@dataclass
class Config:
    # Checkpoint
    nemo_logs_dir: str = None

    # Data
    test_path: str = None
    max_samples: int = 1024
    
    # Parallelism
    gpus_per_node: int = 8
    num_nodes: int = 1
    tensor_parallel_size: int = 8
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    
    # Dynamic Inference settings
    temperature: float = 0.1
    top_k: int = 0
    top_p: float = 0.95
    num_tokens_to_generate: int = 512
    random_seed: int = 42
    
    # Dynamic batching settings
    max_batch_size: int = 16  # Maximum concurrent requests
    block_size_tokens: int = 256  # KV cache block size
    buffer_size_gb: float = 20.0  # KV cache buffer size (increased for larger batches)
    max_tokens: int = 65536  # Max tokens per batch (batch_size * seq_len) - allows ~16 full sequences
    inference_max_seq_length: int = 8192  # Maximum sequence length (prompt + generation)
    
    @property
    def output_path(self) -> str:
        return self.checkpoint_path + ".jsonl"
    
    @property
    def checkpoint_path(self) -> str:
        """Find the latest checkpoint in the checkpoints directory."""
        output_path = Path(self.nemo_logs_dir)
        
        # Find all timestamp directories (format: YYYY-MM-DD_HH-MM-SS)
        timestamp_dirs = [d for d in output_path.iterdir() if d.is_dir() and 
                        len(d.name.split('_')) == 2 and 
                        len(d.name.split('_')[0].split('-')) == 3]
        
        if not timestamp_dirs:
            raise FileNotFoundError(f"No timestamp directories found in {output_path}")
        
        # Get the latest timestamp directory by modification time
        latest_timestamp_dir = max(timestamp_dirs, key=lambda p: p.stat().st_mtime)
        
        # Look for checkpoint directories under checkpoints/
        checkpoints_dir = latest_timestamp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory found in {latest_timestamp_dir}")
        
        # Find all checkpoint directories (starting with "nemo_logs--")
        ckpt_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and 
                    d.name.startswith("nemo_logs--") and 
                    not re.search(r'\.(hf_model|hf_peft|merged)$', d.name)]
        
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")
        
        # Return the latest checkpoint directory by modification time
        latest_ckpt = max(ckpt_dirs, key=lambda p: p.stat().st_mtime)
        return str(latest_ckpt)
    
    def __post_init__(self):
        if self.nemo_logs_dir is None:
            outputs_path =Path.home() / "outputs"
            nemo_logs_dirs = list(outputs_path.rglob("nemo_logs"))
            if nemo_logs_dirs:
                self.nemo_logs_dir = str(max(nemo_logs_dirs, key=lambda p: p.stat().st_mtime))
                print(f"Found nemo_logs directory: {self.nemo_logs_dir}")
            else:
                raise ValueError("No nemo_logs folder found under outputs folder")
            
        if self.test_path is None:
            datasets_path = Path.home() / "datasets"
            test_files = list(datasets_path.rglob("test.jsonl"))
            if test_files:
                self.test_path = str(max(test_files, key=lambda p: p.stat().st_mtime))
                print(f"Found test file: {self.test_path}")
            else:
                raise ValueError("No test.jsonl file found under datasets folder")
            
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create Config from argparse Namespace, only using provided args"""
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() if k in config_fields}
        return cls(**kwargs)

config = None

def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Generate argparse parser from dataclass fields"""
    parser = argparse.ArgumentParser()
    
    for f in fields(dataclass_type):
        field_type = f.type
        default_value = f.default if f.default is not MISSING else None
        
        parser.add_argument(
            f'--{f.name}',
            type=field_type if field_type in [int, float, str, bool] else str,
            default=default_value
        )
    
    return parser

def evaluate_predictions(output_path):
    """Evaluate predictions using multiple metrics."""
    bertscore = evaluate.load('bertscore')
    
    predictions = []
    references = []
    
    with open(output_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            predictions.append(sample['prediction'])
            references.append(sample['label'])
    
    print("\nComputing metrics...")
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang='en')
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nBERTScore F1: {sum(bert_scores['f1'])/len(bert_scores['f1']):.4f}")
    print("="*80)
    
    return {
        'bertscore': sum(bert_scores['f1'])/len(bert_scores['f1'])
    }

def setup_trainer_to_restore_model(path: Path, trainer: nl.Trainer, model: pl.LightningModule):
    """Setup trainer to restore model from checkpoint."""
    print("Setting up trainer to restore model...")

    set_modelopt_spec_if_exists_in_ckpt(model, path)

    if (adapter_meta_path := ckpt_to_weights_subdir(path, is_saving=False) / ADAPTER_META_FILENAME).exists():
        with open(adapter_meta_path, "r") as f:
            metadata = json.load(f)
        restore_config = RestoreConfig(
            path=metadata["model_ckpt_path"],
            load_model_state=True,
            load_optim_state=False,
        )
    else:
        restore_config = RestoreConfig(
            path=path,
            load_model_state=True,
            load_optim_state=False,
        )

    trainer.strategy.restore_config = restore_config
    trainer.strategy._setup_optimizers = False
    trainer.ckpt_path = None
    trainer.strategy.connect(model)
    model.trainer = trainer

    trainer.strategy.setup_environment()

    if not model.state_dict():
        model.configure_model()

    trainer.state.fn = TrainerFn.TESTING
    trainer.strategy.setup_megatron_parallel(trainer=trainer)
    trainer.strategy.trainer = trainer
    trainer.strategy.selective_restore()

    peft: Optional[PEFT] = model.model_transform
    if isinstance(peft, PEFT):
        model = peft(model)
        sharded_sd_metadata = trainer.strategy.unwrapped_checkpoint_io.load_content_metadata(path)
        sharded_state_dict = MegatronModule.sharded_state_dict(model, metadata=sharded_sd_metadata)
        adapter_sharded_state_dict = {k: v for k, v in sharded_state_dict.items() if ".adapter." in k}
        adapter_state = trainer.strategy.checkpoint_io.load_checkpoint(
            ckpt_to_weights_subdir(path, is_saving=False), sharded_state_dict=adapter_sharded_state_dict
        )
        trainer.strategy.load_model_state_dict(adapter_state, strict=False)

def setup_model_and_dynamic_inference(
    path: str,
    trainer: nl.Trainer,
    params_dtype: torch.dtype,
):
    """Setup model and create dynamic inference components."""
        
    # Initialize the trainer to set up parallel state FIRST
    # This is required before loading checkpoints with tensor parallelism
    print("Setting up distributed environment...")
    trainer.strategy.setup_environment()
    
    print("Loading model from checkpoint...")
    
    model = io.load_context(ckpt_to_context_subdir(path), subpath="model")
    setup_trainer_to_restore_model(path=path, trainer=trainer, model=model)
    
    # Get tokenizer
    tokenizer =  MCoreTokenizerWrappper(model.tokenizer, getattr(model.config, "vocab_size", None))
    
    print(f"Model type: {type(model)}")
    
    # For NeMo GPTModel, the MCore model is stored in model.module
    # If it doesn't exist yet, configure_model() will create it
    if not hasattr(model, 'module'):
        print("Configuring model to create module...")
        model.configure_model()
    
    # Now get the MCore model from model.module
    mcore_model = model.module
    print(f"MCore model type: {type(mcore_model)}")
    
    # The module might be wrapped in Float16Module or other wrappers
    # Unwrap until we get to MCoreGPTModel
    while mcore_model and type(mcore_model) is not MCoreGPTModel:
        if hasattr(mcore_model, 'module'):
            print(f"Unwrapping {type(mcore_model).__name__}...")
            mcore_model = mcore_model.module
        else:
            break
    
    if type(mcore_model) is not MCoreGPTModel:
        raise ValueError(
            f"Could not find MCoreGPTModel. Final type is {type(mcore_model)}. "
            f"Available attributes: {[attr for attr in dir(mcore_model) if not attr.startswith('_')][:20]}"
        )
    print(f"✓ Found MCore model: {type(mcore_model)}")
    mcore_model.eval()
    
    # Create dynamic inference context with correct parameters
    print("Creating dynamic inference context...")
    inference_context = DynamicInferenceContext(
        params_dtype=params_dtype,
        num_layers=mcore_model.config.num_layers,
        kv_channels=mcore_model.config.kv_channels,
        num_attention_heads=mcore_model.config.num_query_groups if hasattr(mcore_model.config, 'num_query_groups') else mcore_model.config.num_attention_heads,
        max_sequence_length=config.inference_max_seq_length,
        buffer_size_gb=config.buffer_size_gb,
        buffer_guaranteed_fraction=0.5,  # Reserve 50% for guaranteed requests
        chunk_size_tokens=config.block_size_tokens,
        max_tokens_override=config.max_tokens,
        tensor_model_parallel_size=config.tensor_parallel_size,
        materialize_only_last_token_logits=True,
    )
    
    # Create inference wrapper config
    vocab_size = tokenizer.vocab_size
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=mcore_model.config.hidden_size,
        params_dtype=params_dtype,
        inference_batch_times_seqlen_threshold=config.inference_max_seq_length*config.max_batch_size,
        padded_vocab_size=vocab_size,
        fp32_residual_connection=False,
    )
    
    # Wrap model for inference
    print("Creating inference wrapper...")
    inference_wrapped_model = GPTInferenceWrapper(mcore_model, inference_wrapper_config, inference_context)
    inference_wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
    
    # Create text generation controller
    print("Creating text generation controller...")
    controller = TextGenerationController(inference_wrapped_model, tokenizer)
    
    # Create dynamic inference engine
    print("Creating dynamic inference engine...")
    engine = DynamicInferenceEngine(
        controller=controller,
        context=inference_context,
        termination_id=tokenizer.eod,  
        enable_cuda_graph=False,
        random_seed=config.random_seed,
    )
    
    return engine, tokenizer


def generate_with_dynamic_inference(
    engine: DynamicInferenceEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> List[str]:
    """Generate outputs using dynamic inference engine."""

    # Process requests
    print("Generating outputs...")
    results = {}
    num_requests_total = len(prompts)
    tbar = tqdm(total=num_requests_total)
    request_id = 0
    num_requests_finished = 0
    while True:
        while request_id < num_requests_total:
            try:
                engine.add_request(
                    request_id, prompts[request_id], sampling_params.num_tokens_to_generate
                )
                request_id += 1
            except ContextOverflowError:
                break
        _, finished_requests, _ = engine.step(sampling_params, verbose=False)
        
        if len(finished_requests) > 0:
            for finished_request in finished_requests:
                results[finished_request.request_id] = finished_request.generated_text
                num_requests_finished += 1
                tbar.update(1)
        
        if not (engine.has_unfinished_requests() or request_id < num_requests_total):
            break

    tbar.close()
    assert len(results) == num_requests_total, f"Expected {num_requests_total} results, got {len(results)}"
    
    # Return results in original order
    return [results[i] for i in range(len(prompts))]


def run_testing():
    print("=" * 80)
    print("Testing PEFT Model with Dynamic Inference")
    print("=" * 80)
    print(f"Checkpoint: {config.checkpoint_path}")
    print("=" * 80)
    
    # ========================================================================
    # 1. Setup Trainer
    # ========================================================================
    print("\n[1/3] Setting up trainer...")
    
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=config.tensor_parallel_size,
        pipeline_model_parallel_size=config.pipeline_parallel_size,
        context_parallel_size=config.context_parallel_size,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=config.gpus_per_node,
        num_nodes=config.num_nodes,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )
    print("✓ Trainer configured")
    
    # ========================================================================
    # 2. Load test dataset
    # ========================================================================
    print("\n[2/3] Loading test dataset...")
    
    # Determine how many samples to load
    if config.max_samples is not None:
        # Round up to nearest multiple of max_batch_size
        samples_to_load = ((config.max_samples + config.max_batch_size - 1) // config.max_batch_size) * config.max_batch_size
        print(f"✓ Loading {samples_to_load} samples (rounded up from {config.max_samples} to nearest multiple of {config.max_batch_size})")
    else:
        samples_to_load = None
        print(f"✓ Loading all samples from dataset")
    
    # Load dataset directly from original file
    dataset = []
    with open(config.test_path, 'r') as f:
        for i, line in enumerate(f):
            if samples_to_load is not None and i >= samples_to_load:
                break
            dataset.append(json.loads(line))
    
    prompts = [sample['input'] for sample in dataset]
    labels = [sample.get('output', None) for sample in dataset]
    
    print(f"✓ Loaded {len(prompts)} samples from {config.test_path}")
    
    # Calculate max prompt length (rough estimate)
    max_prompt_length = config.inference_max_seq_length - config.num_tokens_to_generate
    print(f"✓ Estimated max prompt length: {max_prompt_length} tokens")
    
    # ========================================================================
    # 3. Setup sampling params and model
    # ========================================================================
    print("\n[3/3] Setting up dynamic inference...")
    
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        num_tokens_to_generate=config.num_tokens_to_generate,
        return_log_probs=False,
    )
    
    engine, tokenizer = setup_model_and_dynamic_inference(
        path=config.checkpoint_path,
        trainer=trainer,
        params_dtype=torch.bfloat16,

    )
    
    print("✓ Dynamic inference setup complete")
    
    # ========================================================================
    # 4. Generate predictions
    # ========================================================================
    print("\n[4/4] Generating predictions...")
    
    start_time = time.time()
    predictions = generate_with_dynamic_inference(
        engine=engine,
        prompts=prompts,
        sampling_params=sampling_params,
    )
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = len(prompts) / total_time
    
    print(f"✓ Generated {len(predictions)} predictions in {total_time:.2f}s")
    print(f"✓ Throughput: {throughput:.2f} samples/s")
    
    # ========================================================================
    # 5. Save results
    # ========================================================================
    print("\n[5/5] Saving results...")
    
    if trainer.global_rank == 0:
        with open(config.output_path, 'w') as f:
            for sample, prediction in zip(dataset, predictions):
                sample['label'] = sample.pop('output', None)
                sample['prediction'] = prediction
                f.write(json.dumps(sample) + '\n')
        
        print(f"✓ Results saved to {config.output_path}")
    
    # ========================================================================
    # Display sample results
    # ========================================================================
    if trainer.global_rank == 0:
        print("\n" + "=" * 80)
        print("SAMPLE PREDICTIONS")
        print("=" * 80)
        
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n--- Sample {i+1} ---")
            print(f"Input:\n{sample['input'][:200]}...")
            print(f"\nExpected:\n{sample.get('label', 'N/A')[:200]}...")
            print(f"\nPredicted:\n{predictions[i][:200]}...")
            print("-" * 80)
        
        # Evaluate
        print("\n" + "=" * 80)
        print("EVALUATING PREDICTIONS")
        print("=" * 80)
        evaluate_predictions(config.output_path)
    
    print("\n✓ Complete!")


def main():
    global config
    parser = create_parser_from_dataclass(Config)
    args = parser.parse_args()
    config = Config.from_args(args)
    
    run_testing()


if __name__ == "__main__":
    main()