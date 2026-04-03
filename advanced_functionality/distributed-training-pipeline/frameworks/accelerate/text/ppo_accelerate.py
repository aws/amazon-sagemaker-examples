import os
# Help minimize fragmentation for the dual-engine setup
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_TIMEOUT"] = "3600"  # 1 hour timeout
os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Better error reporting

import argparse
import json
import re
import time
import tempfile
import gc
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from vllm import LLM, SamplingParams

# Ray for vLLM isolation
import ray
from dataset_module import HFDatasetConfig, prepare_datasets


def find_latest_converted_checkpoint(checkpoints_dir: str) -> str:
    """Find the latest converted checkpoint (.hf_model or .hf_peft)."""
    ckpt_dir_path = Path(checkpoints_dir)
    if not ckpt_dir_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    converted_ckpts = []
    for d in ckpt_dir_path.glob("checkpoint-*"):
        if d.is_dir() and re.search(r'\.(hf_model|hf_peft)', d.name):
            match = re.match(r'checkpoint-(\d+)', d.name)
            if match:
                converted_ckpts.append((int(match.group(1)), str(d)))
    
    if not converted_ckpts:
        raise FileNotFoundError(
            f"No converted checkpoints found in {checkpoints_dir}. "
            f"Please run convert_checkpoint_to_hf.py first."
        )
    
    converted_ckpts.sort(key=lambda x: x[0])
    return converted_ckpts[-1][1]


def is_peft_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is a PEFT adapter."""
    return checkpoint_path.endswith('.hf_peft') or Path(checkpoint_path).joinpath('adapter_config.json').exists()


# --- Ray-based vLLM Engine Actor with Tensor Parallelism ---
@ray.remote(num_gpus=8)  # Request all 8 GPUs for tensor parallelism
class VLLMEngineActor:
    """
    Ray actor that wraps a vLLM engine with tensor parallelism.
    Uses all GPUs together for distributed inference.
    Multiple training ranks can call this single actor concurrently.
    """
    
    def __init__(self, model_path: str, base_model_id: str, tensor_parallel_size: int = 8, gpu_memory_utilization: float = 0.3):
        """Initialize vLLM engine with tensor parallelism."""
        self.temp_dir = None
        
        # CRITICAL: Clear ALL distributed training environment variables
        distributed_vars = [
            'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'GROUP_RANK',
            'MASTER_ADDR', 'MASTER_PORT',
            'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME',
            'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_RESTART_COUNT',
            'TORCHELASTIC_MAX_RESTARTS', 'TORCHELASTIC_USE_AGENT_STORE',
            'LOCAL_WORLD_SIZE', 'NODE_RANK',
        ]
        
        print(f"[vLLM Actor] Cleaning environment variables...")
        for var in distributed_vars:
            if var in os.environ:
                print(f"[vLLM Actor] Removing {var}={os.environ[var]}")
                del os.environ[var]
        
        # Destroy any existing torch distributed state
        if torch.distributed.is_initialized():
            print(f"[vLLM Actor] Destroying existing torch.distributed state...")
            torch.distributed.destroy_process_group()
        
        # Clear torch distributed internal state
        import torch.distributed as dist
        if hasattr(dist, '_default_pg'):
            dist._default_pg = None
        if hasattr(dist, '_world'):
            dist._world = None
        
        # Check if PEFT checkpoint
        is_peft = model_path.endswith('.hf_peft') or Path(model_path).joinpath('adapter_config.json').exists()
        
        if is_peft:
            self.temp_dir = tempfile.mkdtemp(prefix=f"vllm_merged_model_")
            print(f"[vLLM Actor] Merging PEFT adapter for vLLM into {self.temp_dir}...")
            
            base_model_path = base_model_id
            base_model_file = Path(model_path) / "base_model.txt"
            if base_model_file.exists():
                base_model_path = base_model_file.read_text().strip()
                
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="cpu"
            )
            
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            
            model.save_pretrained(self.temp_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer.save_pretrained(self.temp_dir)
            
            load_path = self.temp_dir
            
            del model
            del base_model
            gc.collect()
        else:
            print(f"[vLLM Actor] Loading vLLM directly from {model_path}...")
            load_path = model_path
        
        print(f"[vLLM Actor] Initializing vLLM Engine with TP={tensor_parallel_size}, GPU Util={gpu_memory_utilization}")
        print(f"[vLLM Actor] Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        
        # Initialize vLLM with tensor parallelism across all 8 GPUs
        self.llm = LLM(
            model=load_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            disable_log_stats=True,
            seed=42,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            max_model_len=4096,
        )
        
        print(f"[vLLM Actor] vLLM Engine initialized successfully with TP={tensor_parallel_size}!")
    
    def generate(self, prompts: List[str], sampling_params_dict: dict):
        """Generate completions for prompts. Thread-safe for concurrent calls from multiple ranks."""
        sampling_params = SamplingParams(**sampling_params_dict)
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract token IDs from outputs
        results = []
        for output in outputs:
            results.append({
                'prompt_token_ids': output.prompt_token_ids,
                'generated_token_ids': list(output.outputs[0].token_ids)
            })
        
        return results
    
    def cleanup(self):
        """Clean up temporary directories."""
        if self.temp_dir:
            print(f"[vLLM Actor] Cleaning up temp directory: {self.temp_dir}")
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        return True


class ValueHead(nn.Module):
    """
    Value head for computing state values V_ψ(s).
    Takes hidden states from the policy model and outputs a scalar value.
    """
    def __init__(self, hidden_size: int, dtype=torch.bfloat16):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1, bias=False, dtype=dtype)
        with torch.no_grad():
            self.value_head.weight.normal_(std=0.01)
    
    def forward(self, hidden_states):
        values = self.value_head(hidden_states).squeeze(-1)
        return values


class PolicyWithValue(nn.Module):
    """
    Wrapper that combines policy model with value head.
    Shares the backbone for efficiency.
    """
    def __init__(self, policy_model, hidden_size: int, dtype=torch.bfloat16):
        super().__init__()
        self.policy_model = policy_model
        self.value_head = ValueHead(hidden_size, dtype=dtype)
        
    def forward(self, input_ids, attention_mask, return_values=False):
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_values
        )
        
        values = None
        if return_values:
            hidden_states = outputs.hidden_states[-1]
            values = self.value_head(hidden_states)
        
        return outputs, values
    
    def get_base_model(self):
        return self.policy_model
    
    def save_pretrained(self, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(save_path)
        value_head_path = save_path / "value_head.pt"
        torch.save(self.value_head.state_dict(), value_head_path)
        print(f"Saved value head to {value_head_path}")


class PromptDataset(Dataset):
    """Dataset of prompts for PPO training."""
    
    def __init__(self, data_path: Path, tokenizer: AutoTokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompts = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                self.prompts.append(sample['input'])
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized.get('attention_mask', [1] * len(tokenized['input_ids'])),
            'prompt_text': prompt,
        }


def collate_fn(batch, tokenizer, pad_to_multiple_of=8):
    """Collate function for prompt batches."""
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    prompt_texts = [item['prompt_text'] for item in batch]
    
    padded = tokenizer.pad(
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors='pt',
    )
    
    return {
        'input_ids': padded['input_ids'],
        'attention_mask': padded['attention_mask'],
        'prompt_texts': prompt_texts,
    }


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # Model paths
    sft_model_path: str = None
    reward_model_path: str = None
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
    
    # PPO hyperparameters
    ppo_epochs: int = 4
    num_rollouts: int = 128
    max_steps: int = 1000
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    clip_range: float = 0.2
    vf_coef: float = 1.0
    kl_coef: float = 0.02
    gamma: float = 1.0
    lam: float = 0.95
    entropy_coef: float = 0.01
    target_kl: float = 0.015
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    
    # vLLM Settings
    vllm_tensor_parallel_size: int = 8
    vllm_gpu_memory_utilization: float = 0.10
    
    # Sequence settings
    max_seq_length: int = 2048
    
    # Paths
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    use_wandb: bool = False
    
    # Other
    seed: int = 42
    num_workers: int = 4
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'PPOConfig':
        """Create PPOConfig from argparse Namespace."""
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
        
        if 'lora_target_modules' in kwargs and isinstance(kwargs['lora_target_modules'], str):
            kwargs['lora_target_modules'] = [m.strip() for m in kwargs['lora_target_modules'].split(',')]
        
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.sft_model_path is None:
            checkpoints_dir = str(Path.home() / f"results/{self.hf_model_id}")
            self.sft_model_path = find_latest_converted_checkpoint(checkpoints_dir)
            print(f"Using SFT checkpoint: {self.sft_model_path}")
        
        if self.reward_model_path is None:
            reward_checkpoints_dir = str(Path.home() / f"results/reward_{self.hf_model_id}")
            self.reward_model_path = find_latest_converted_checkpoint(reward_checkpoints_dir)
            print(f"Using reward model checkpoint: {self.reward_model_path}")
        
        if self.data_dir is None:
            dataset_name = self.hf_dataset_config.dataset_name.replace('/', '_')
            dataset_config = self.hf_dataset_config.dataset_config or 'default'
            train_pct = int(self.hf_dataset_config.train_split_ratio * 100)
            remaining_pct = 100 - train_pct
            val_pct = int(remaining_pct * (1 - self.hf_dataset_config.val_test_split_ratio))
            test_pct = remaining_pct - val_pct
            self.data_dir = str(Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%")
        
        if self.output_dir is None:
            self.output_dir = str(Path.home() / f"results/ppo_{self.hf_model_id}")


def compute_advantages_dense(rewards, values, gamma, lam, mask):
    """Compute GAE advantages for DENSE rewards (per-token)."""
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    for i in range(batch_size):
        last_gae = 0
        seq_mask = mask[i]
        actual_len = seq_mask.sum().item()
        
        if actual_len == 0:
            continue
        
        for t in reversed(range(int(actual_len))):
            if t == actual_len - 1:
                next_value = 0
            else:
                next_value = values[i, t + 1]
            
            delta = rewards[i, t] + gamma * next_value - values[i, t]
            advantages[i, t] = last_gae = delta + gamma * lam * last_gae
    
    returns = advantages + values
    advantages = advantages * mask
    returns = returns * mask
    
    return advantages, returns


def train(config: PPOConfig):
    """Main PPO training function following the complete RLHF algorithm with DENSE rewards."""
    
    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.use_wandb else None,
    )
    
    if config.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project="ppo-training-dense", config=vars(config))
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("PPO-RLHF Training (Complete Algorithm with DENSE Rewards)")
        print("=" * 80)
        print(f"Policy model: {config.sft_model_path or config.hf_model_id}")
        print(f"Reward model: {config.reward_model_path}")
        print(f"Dataset: {config.hf_dataset_config.dataset_name}")
        print(f"Output directory: {config.output_dir}")
        print(f"Number of training GPUs: {accelerator.num_processes}")
        print(f"vLLM tensor parallel size: {config.vllm_tensor_parallel_size}")
        print("=" * 80)
    
    # Use a fixed namespace for all processes
    RAY_NAMESPACE = "ppo_training"
    
    # Initialize Ray with explicit namespace
    if accelerator.is_main_process:
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_gpus=accelerator.num_processes,
                namespace=RAY_NAMESPACE,
                _temp_dir="/tmp/ray",
                include_dashboard=False,
                _metrics_export_port=None,
            )
            print(f"Ray initialized with {accelerator.num_processes} GPUs in namespace '{RAY_NAMESPACE}'")
    
    accelerator.wait_for_everyone()
    
    if not accelerator.is_main_process:
        if not ray.is_initialized():
            ray.init(address='auto', namespace=RAY_NAMESPACE, ignore_reinit_error=True)
            print(f"[Rank {accelerator.process_index}] Connected to Ray namespace '{RAY_NAMESPACE}'")
    
    accelerator.wait_for_everyone()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path or config.hf_model_id,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare datasets
    marker_file = Path(config.data_dir) / ".data_ready"
    if accelerator.is_main_process:
        prepare_datasets(config.hf_dataset_config, config.data_dir)
        marker_file.touch()
    
    accelerator.wait_for_everyone()
    
    # Create dataset
    train_dataset = PromptDataset(
        Path(config.data_dir) / "training.jsonl",
        tokenizer,
        config.max_seq_length,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    
    # =========================================================================
    # STEP 1: INITIALIZE
    # =========================================================================
    
    # 1a. Policy π_θ from SFT model π^SFT
    if accelerator.is_main_process:
        print(f"Loading policy model from: {config.sft_model_path}")
    
    if is_peft_checkpoint(config.sft_model_path):
        base_model_path = config.hf_model_id
        base_model_file = Path(config.sft_model_path) / "base_model.txt"
        if base_model_file.exists():
            base_model_path = base_model_file.read_text().strip()
        
        policy_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        policy_model = PeftModel.from_pretrained(policy_model, config.sft_model_path)
        
        if accelerator.is_main_process:
            print("Merging SFT PEFT adapter into base model for training...")
        policy_model = policy_model.merge_and_unload()
    else:
        policy_model = AutoModelForCausalLM.from_pretrained(
            config.sft_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    
    policy_model.gradient_checkpointing_enable()
    
    hidden_size = policy_model.config.hidden_size
    
    if not config.full_ft:
        if accelerator.is_main_process:
            print("Applying new LoRA to policy model for PPO training...")
        
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, peft_config)
        policy_model = policy_model.to(torch.bfloat16)
        
        if accelerator.is_main_process:
            policy_model.print_trainable_parameters()
    
    # 1b. Wrap policy with value head V_ψ
    if accelerator.is_main_process:
        print(f"Adding value head (hidden_size={hidden_size})...")
    
    policy_with_value = PolicyWithValue(policy_model, hidden_size, dtype=torch.bfloat16)
    policy_with_value = policy_with_value.to(torch.bfloat16)
    
    # 1c. Fix reference policy π_ref ← π^SFT (ON CPU)
    if accelerator.is_main_process:
        print("Loading reference model (will be on CPU)...")
    
    if is_peft_checkpoint(config.sft_model_path):
        base_model_path = config.hf_model_id
        base_model_file = Path(config.sft_model_path) / "base_model.txt"
        if base_model_file.exists():
            base_model_path = base_model_file.read_text().strip()
        
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            use_cache=True,
            device_map="cpu",
        )
        ref_model = PeftModel.from_pretrained(ref_model, config.sft_model_path)
        ref_model = ref_model.merge_and_unload()
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.sft_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            use_cache=True,
            device_map="cpu",
        )
    
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model = ref_model.to('cpu').to(torch.float32)
    
    # 1d. Load trained reward model r_φ (ON CPU)
    if accelerator.is_main_process:
        print(f"Loading reward model from: {config.reward_model_path} (will be on CPU)...")
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=True,
        num_labels=1,
        device_map="cpu",
    )

    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = tokenizer.pad_token_id
        if accelerator.is_main_process:
            print(f"Set reward model pad_token_id to {tokenizer.pad_token_id}")
        
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_model = reward_model.to('cpu').to(torch.float32)
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("Memory Configuration:")
        print("✓ Reference model: CPU (eager attention, float32)")
        print("✓ Reward model: CPU (eager attention, float32)")
        print("✓ Policy model + Value head: GPU (DDP across all ranks)")
        print(f"✓ vLLM: Distributed across {config.vllm_tensor_parallel_size} GPUs (TP={config.vllm_tensor_parallel_size})")
        print("=" * 80)
    
    # Initialize SINGLE Ray vLLM Engine with TP=8 (Main process only)
    vllm_actor = None
    
    if accelerator.is_main_process:
        print(f"[Main Process] Initializing Ray vLLM actor with TP={config.vllm_tensor_parallel_size}...")
        
        try:
            vllm_actor = VLLMEngineActor.options(
                num_gpus=config.vllm_tensor_parallel_size,
                name="vllm_inference_actor",
                lifetime="detached",
                get_if_exists=False,
            ).remote(
                model_path=config.sft_model_path,
                base_model_id=config.hf_model_id,
                tensor_parallel_size=config.vllm_tensor_parallel_size,
                gpu_memory_utilization=config.vllm_gpu_memory_utilization
            )
            
            ray.get(vllm_actor.generate.remote(["test"], {"max_tokens": 1, "temperature": 1.0}))
            print(f"[Main Process] Ray vLLM actor initialized successfully!")
            
        except Exception as e:
            print(f"[Main Process] Critical Ray vLLM Error: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    accelerator.wait_for_everyone()
    
    if not accelerator.is_main_process:
        vllm_actor = ray.get_actor("vllm_inference_actor")
        print(f"[Rank {accelerator.process_index}] Retrieved vLLM actor")
    
    accelerator.wait_for_everyone()
    
    sampling_params_dict = {
        'temperature': config.temperature,
        'top_p': config.top_p,
        'max_tokens': config.max_new_tokens,
    }
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        policy_with_value.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    num_training_steps = config.max_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    policy_with_value, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy_with_value, optimizer, train_dataloader, lr_scheduler
    )
    
    # =========================================================================
    # STEP 2: TRAINING LOOP
    # =========================================================================
    
    global_step = 0
    policy_with_value.train()
    
    if accelerator.is_main_process:
        print("\nStarting PPO-RLHF training with DENSE rewards...")
        print("=" * 80)
    
    dataloader_iterator = iter(train_dataloader)
    
    for step in range(config.max_steps):
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_dataloader)
            batch = next(dataloader_iterator)
        
        prompt_texts = batch['prompt_texts']
        
        # =====================================================================
        # PHASE 1: EXPERIENCE COLLECTION
        # =====================================================================
        with torch.no_grad():
            generation_results = ray.get(
                vllm_actor.generate.remote(prompt_texts, sampling_params_dict)
            )
            
            batch_input_ids = []
            prompt_lengths = []
            max_len_in_batch = 0
            
            for result in generation_results:
                prompt_len = len(result['prompt_token_ids'])
                full_seq = result['prompt_token_ids'] + result['generated_token_ids']
                batch_input_ids.append(full_seq)
                prompt_lengths.append(prompt_len)
                max_len_in_batch = max(max_len_in_batch, len(full_seq))
            
            padded_batch = []
            for seq in batch_input_ids:
                pad_len = max_len_in_batch - len(seq)
                padded_seq = seq + [tokenizer.pad_token_id] * pad_len
                padded_batch.append(padded_seq)
            
            generated = torch.tensor(padded_batch, dtype=torch.long, device=accelerator.device)
            attention_mask = (generated != tokenizer.pad_token_id).long()
            
            generation_mask = torch.zeros_like(attention_mask)
            for i, prompt_len in enumerate(prompt_lengths):
                generation_mask[i, prompt_len:] = attention_mask[i, prompt_len:]
            
            accelerator.wait_for_everyone()
            
            generated_cpu = generated.cpu()
            attention_mask_cpu = attention_mask.cpu()
            
            accelerator.wait_for_everyone()
            
            reward_outputs = reward_model(
                input_ids=generated_cpu,
                attention_mask=attention_mask_cpu
            )
            reward_scores = reward_outputs.logits.squeeze(-1).to(accelerator.device)
            
            accelerator.wait_for_everyone()
            
            policy_outputs, _ = policy_with_value(
                input_ids=generated,
                attention_mask=attention_mask,
                return_values=False
            )
            logits = policy_outputs.logits[:, :-1, :]
            labels = generated[:, 1:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            old_log_probs_per_token = torch.gather(
                log_probs, 2, labels.unsqueeze(-1)
            ).squeeze(-1)
            
            old_log_probs_stored = old_log_probs_per_token.detach()
            
            accelerator.wait_for_everyone()
            
            labels_cpu = labels.cpu()
            
            ref_outputs = ref_model(
                input_ids=generated_cpu,
                attention_mask=attention_mask_cpu
            )
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs_dist = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs_per_token = torch.gather(
                ref_log_probs_dist, 2, labels_cpu.unsqueeze(-1)
            ).squeeze(-1).to(accelerator.device)
            
            accelerator.wait_for_everyone()
            
            kl_per_token = old_log_probs_per_token - ref_log_probs_per_token
            dense_rewards = -config.kl_coef * kl_per_token
            
            for i in range(generated.size(0)):
                gen_mask_i = generation_mask[i, 1:]
                if gen_mask_i.sum() > 0:
                    last_gen_idx = gen_mask_i.nonzero(as_tuple=True)[0][-1].item()
                    dense_rewards[i, last_gen_idx] += reward_scores[i]
            
            reward_mask = generation_mask[:, 1:]
            dense_rewards = dense_rewards * reward_mask
            
            accelerator.wait_for_everyone()
            
            _, values = policy_with_value(
                input_ids=generated,
                attention_mask=attention_mask,
                return_values=True
            )
            values = values[:, :-1]
            values = values.detach()
            
            accelerator.wait_for_everyone()
        
        # =====================================================================
        # PHASE 2: GENERALIZED ADVANTAGE ESTIMATION
        # =====================================================================
        with torch.no_grad():
            advantages, returns = compute_advantages_dense(
                dense_rewards,
                values,
                config.gamma,
                config.lam,
                reward_mask
            )
            
            valid_advantages = advantages[reward_mask.bool()]
            if len(valid_advantages) > 1:
                adv_mean = valid_advantages.mean()
                adv_std = valid_advantages.std()
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                advantages = advantages * reward_mask
            
            advantages = advantages.detach()
            returns = returns.detach()
            
            accelerator.wait_for_everyone()
        
        # =====================================================================
        # PHASE 3: POLICY AND VALUE UPDATES
        # =====================================================================
        
        for ppo_epoch in range(config.ppo_epochs):
            outputs, new_values = policy_with_value(
                input_ids=generated,
                attention_mask=attention_mask,
                return_values=True
            )
            new_values = new_values[:, :-1]
            
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs_per_token = torch.gather(
                log_probs, 2, labels.unsqueeze(-1)
            ).squeeze(-1)
            
            ratio = torch.exp(new_log_probs_per_token - old_log_probs_stored)
            
            clipped_ratio = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range)
            policy_loss_unclipped = ratio * advantages
            policy_loss_clipped = clipped_ratio * advantages
            
            policy_loss_per_token = -torch.min(policy_loss_unclipped, policy_loss_clipped)
            policy_loss = (policy_loss_per_token * reward_mask).sum() / (reward_mask.sum() + 1e-8)
            
            value_loss_per_token = (new_values - returns) ** 2
            value_loss = (value_loss_per_token * reward_mask).sum() / (reward_mask.sum() + 1e-8)
            
            entropy_per_token = -(log_probs.exp() * log_probs).sum(dim=-1)
            entropy = (entropy_per_token * reward_mask).sum() / (reward_mask.sum() + 1e-8)
            
            total_loss = policy_loss + config.vf_coef * value_loss - config.entropy_coef * entropy
            
            accelerator.backward(total_loss)
            
            has_gradients = any(
                p.grad is not None for p in policy_with_value.parameters() if p.requires_grad
            )
            
            if accelerator.sync_gradients:
                if has_gradients:
                    accelerator.clip_grad_norm_(policy_with_value.parameters(), config.max_grad_norm)
                else:
                    if accelerator.is_main_process:
                        print(f"Warning: No gradients on step {global_step}, epoch {ppo_epoch}")
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                approx_kl_per_token = old_log_probs_stored - new_log_probs_per_token
                approx_kl = (approx_kl_per_token * reward_mask).sum() / (reward_mask.sum() + 1e-8)
                
                if approx_kl > config.target_kl:
                    if accelerator.is_main_process and ppo_epoch < config.ppo_epochs - 1:
                        print(f"  Early stopping at PPO epoch {ppo_epoch+1}/{config.ppo_epochs}, KL={approx_kl:.4f}")
                    break
            
            global_step += 1
            
            if global_step % config.logging_steps == 0 and accelerator.is_main_process:
                avg_reward = (dense_rewards * reward_mask).sum() / (reward_mask.sum() + 1e-8)
                avg_kl = (kl_per_token * reward_mask).sum() / (reward_mask.sum() + 1e-8)
                avg_advantage = (advantages * reward_mask).sum() / (reward_mask.sum() + 1e-8)
                avg_ratio = (ratio * reward_mask).sum() / (reward_mask.sum() + 1e-8)
                
                print(f"Step {global_step}/{config.max_steps} | "
                      f"Reward Score: {reward_scores.mean().item():.4f} | "
                      f"Avg Dense Reward: {avg_reward.item():.4f} | "
                      f"Avg KL: {avg_kl.item():.4f} | "
                      f"Advantage: {avg_advantage.item():.4f} | "
                      f"Policy Loss: {policy_loss.item():.4f} | "
                      f"Value Loss: {value_loss.item():.4f} | "
                      f"Ratio: {avg_ratio.item():.4f}")
                
                if config.use_wandb:
                    wandb.log({
                        "reward_score": reward_scores.mean().item(),
                        "avg_dense_reward": avg_reward.item(),
                        "avg_kl_divergence": avg_kl.item(),
                        "avg_advantage": avg_advantage.item(),
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy": entropy.item(),
                        "avg_ratio": avg_ratio.item(),
                        "approx_kl": approx_kl.item(),
                        "global_step": global_step,
                    })
            
            if global_step % config.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_path = Path(config.output_dir) / f"checkpoint-{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(policy_with_value)
                    unwrapped_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")
                accelerator.wait_for_everyone()
    
    # =========================================================================
    # FINAL CLEANUP
    # =========================================================================
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("Training loop completed - starting cleanup")
        print("=" * 80)
    
    if accelerator.is_main_process:
        try:
            ray.get(vllm_actor.cleanup.remote())
            print("[Main Process] Ray vLLM actor cleaned up successfully")
        except Exception as e:
            print(f"[Main Process] Error cleaning up Ray actor: {e}")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("Training completed! Saving final model...")
        print("=" * 80)
        
        final_path = Path(config.output_dir) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(policy_with_value)
        unwrapped_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"Final model saved to {final_path}")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("Shutting down Ray...")
        try:
            ray.shutdown()
            print("Ray shutdown complete")
        except Exception as e:
            print(f"Error during Ray shutdown: {e}")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("All cleanup complete - exiting")
        print("=" * 80)


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Generate argparse parser from dataclass fields."""
    parser = argparse.ArgumentParser(description="Train Policy Model with PPO (Dense Rewards)")
    
    for f in fields(dataclass_type):
        if f.name == 'hf_dataset_config':
            for hf_field in fields(HFDatasetConfig):
                if hf_field.name in ['custom_converter', 'load_kwargs']:
                    continue
                arg_name = f'hfdc_{hf_field.name}'
                field_type = hf_field.type
                default_value = hf_field.default if hf_field.default is not MISSING else None
                
                if field_type == bool:
                    parser.add_argument(f'--{arg_name}', action='store_true' if not default_value else 'store_false', default=None)
                elif field_type in [int, float, str]:
                    parser.add_argument(f'--{arg_name}', type=field_type, default=None)
                else:
                    parser.add_argument(f'--{arg_name}', type=str, default=None)
        elif f.name == 'lora_target_modules':
            parser.add_argument(f'--{f.name}', type=str, default=None, help='Comma-separated list of target modules for LoRA')
        else:
            field_type = f.type
            default_value = f.default if f.default is not MISSING else (f.default_factory() if f.default_factory is not MISSING else None)
            
            if field_type == bool:
                parser.add_argument(f'--{f.name}', action='store_true' if not default_value else 'store_false', default=None)
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
    parser = create_parser_from_dataclass(PPOConfig)
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = PPOConfig.from_args(args)
    train(config)


if __name__ == "__main__":
    main()
