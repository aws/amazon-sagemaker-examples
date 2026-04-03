import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re
from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

from rm_dataset_module import RMDatasetConfig, prepare_rm_datasets


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


def find_common_prefix_length(tokens1: List[int], tokens2: List[int]) -> int:
    """Find the length of the common prefix between two token sequences.
    
    Args:
        tokens1: First token sequence
        tokens2: Second token sequence
    
    Returns:
        Length of common prefix
    """
    min_len = min(len(tokens1), len(tokens2))
    for i in range(min_len):
        if tokens1[i] != tokens2[i]:
            return i
    return min_len


class DPODataset(Dataset):
    """Dataset for DPO training with chosen/rejected pairs."""
    
    def __init__(self, data_path: Path, tokenizer: AutoTokenizer, max_seq_length: int = 2048, debug: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        self.debug = debug
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        # Debug: Print first sample structure
        if self.debug and len(self.samples) > 0:
            print("\n=== DEBUG: First sample structure ===")
            print(f"Keys: {self.samples[0].keys()}")
            for key in self.samples[0].keys():
                value = self.samples[0][key]
                if isinstance(value, str):
                    print(f"{key}: {value[:100]}..." if len(value) > 100 else f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Handle both formats: with or without separate input field
        if 'input' in sample and sample['input']:
            # Format 1: Separate prompt field
            prompt_text = sample['input']
            chosen_text = sample['input'] + sample['chosen']
            rejected_text = sample['input'] + sample['rejected']
            
            # Tokenize prompt separately to get accurate length
            prompt_tok = self.tokenizer(
                prompt_text,
                truncation=False,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )
            prompt_len = len(prompt_tok['input_ids'])
            
            if self.debug and idx == 0:
                print(f"\n=== DEBUG: Format 1 (with 'input' field) ===")
                print(f"Prompt length: {prompt_len}")
                print(f"Prompt: {prompt_text[:100]}...")
            
        else:
            # Format 2: No separate prompt - find common prefix
            chosen_text = sample['chosen']
            rejected_text = sample['rejected']
            
            # Tokenize both to find common prefix
            chosen_tok_temp = self.tokenizer(
                chosen_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )
            rejected_tok_temp = self.tokenizer(
                rejected_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )
            
            # Find common prefix length
            prompt_len = find_common_prefix_length(
                chosen_tok_temp['input_ids'],
                rejected_tok_temp['input_ids']
            )
            
            if self.debug and idx == 0:
                print(f"\n=== DEBUG: Format 2 (no 'input' field) ===")
                print(f"Chosen text: {chosen_text[:200]}...")
                print(f"Rejected text: {rejected_text[:200]}...")
                print(f"Chosen tokens (first 20): {chosen_tok_temp['input_ids'][:20]}")
                print(f"Rejected tokens (first 20): {rejected_tok_temp['input_ids'][:20]}")
                print(f"Common prefix length: {prompt_len}")
                print(f"Chosen length: {len(chosen_tok_temp['input_ids'])}")
                print(f"Rejected length: {len(rejected_tok_temp['input_ids'])}")
        
        # Tokenize full sequences
        chosen_tok = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        
        rejected_tok = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        
        # If sequences were truncated, adjust prompt_len
        # The prompt_len should not exceed the actual sequence length
        if len(chosen_tok['input_ids']) < prompt_len:
            prompt_len = len(chosen_tok['input_ids'])
        if len(rejected_tok['input_ids']) < prompt_len:
            prompt_len = min(prompt_len, len(rejected_tok['input_ids']))
        
        # Ensure prompt_len is at least 1 (we need at least some completion tokens)
        # If prompt_len equals sequence length, set it to length-1 to have at least 1 completion token
        if prompt_len >= len(chosen_tok['input_ids']):
            prompt_len = max(0, len(chosen_tok['input_ids']) - 1)
        if prompt_len >= len(rejected_tok['input_ids']):
            prompt_len = max(0, len(rejected_tok['input_ids']) - 1)
        
        if self.debug and idx == 0:
            print(f"\n=== DEBUG: Final prompt_len ===")
            print(f"Final prompt_len: {prompt_len}")
            print(f"Chosen seq length: {len(chosen_tok['input_ids'])}")
            print(f"Rejected seq length: {len(rejected_tok['input_ids'])}")
            print(f"Completion tokens in chosen: {len(chosen_tok['input_ids']) - prompt_len}")
            print(f"Completion tokens in rejected: {len(rejected_tok['input_ids']) - prompt_len}")
        
        return {
            'input_ids_chosen': chosen_tok['input_ids'],
            'attention_mask_chosen': chosen_tok.get('attention_mask', [1] * len(chosen_tok['input_ids'])),
            'input_ids_rejected': rejected_tok['input_ids'],
            'attention_mask_rejected': rejected_tok.get('attention_mask', [1] * len(rejected_tok['input_ids'])),
            'prompt_len': prompt_len,
        }


def collate_fn(batch, tokenizer, pad_to_multiple_of=8):
    """Collate function for DPO batches."""
    batch_chosen = {
        'input_ids': [item['input_ids_chosen'] for item in batch],
        'attention_mask': [item['attention_mask_chosen'] for item in batch],
    }
    batch_rejected = {
        'input_ids': [item['input_ids_rejected'] for item in batch],
        'attention_mask': [item['attention_mask_rejected'] for item in batch],
    }
    
    batch_chosen = tokenizer.pad(
        batch_chosen,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors='pt',
    )
    batch_rejected = tokenizer.pad(
        batch_rejected,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors='pt',
    )
    
    return {
        'input_ids_chosen': batch_chosen['input_ids'],
        'attention_mask_chosen': batch_chosen['attention_mask'],
        'input_ids_rejected': batch_rejected['input_ids'],
        'attention_mask_rejected': batch_rejected['attention_mask'],
        'prompt_lens': [item['prompt_len'] for item in batch],
    }


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    
    # Model paths
    sft_model_path: str = None
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
    rm_dataset_config: RMDatasetConfig = field(default_factory=lambda: RMDatasetConfig(
        dataset_name="Anthropic/hh-rlhf",
        split="train",
        train_split_ratio=0.9,
        val_test_split_ratio=0.5,
        num_proc=8
    ))
    
    # DPO hyperparameters
    beta: float = 0.1
    max_steps: int = 10000
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Sequence settings
    max_seq_length: int = 2048
    
    # Paths
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    use_wandb: bool = False
    debug: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Other
    seed: int = 42
    num_workers: int = 4
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'DPOConfig':
        """Create DPOConfig from argparse Namespace."""
        rm_config_kwargs = {}
        for key, value in vars(args).items():
            if key.startswith("rmdc_") and value is not None:
                field_name = key[5:]
                rm_config_kwargs[field_name] = value
        
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() 
                  if k in config_fields and not k.startswith("rmdc_") and v is not None}
        
        if rm_config_kwargs:
            kwargs['rm_dataset_config'] = RMDatasetConfig(**rm_config_kwargs)
        
        if 'lora_target_modules' in kwargs and isinstance(kwargs['lora_target_modules'], str):
            kwargs['lora_target_modules'] = [m.strip() for m in kwargs['lora_target_modules'].split(',')]
        
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.sft_model_path is None:
            checkpoints_dir = str(Path.home() / f"results/{self.hf_model_id}")
            self.sft_model_path = find_latest_converted_checkpoint(checkpoints_dir)
            print(f"Using SFT checkpoint: {self.sft_model_path}")
        
        if self.data_dir is None:
            dataset_name = self.rm_dataset_config.dataset_name.replace('/', '_')
            dataset_config = self.rm_dataset_config.dataset_config or 'default'
            train_pct = int(self.rm_dataset_config.train_split_ratio * 100)
            remaining_pct = 100 - train_pct
            val_pct = int(remaining_pct * (1 - self.rm_dataset_config.val_test_split_ratio))
            test_pct = remaining_pct - val_pct
            self.data_dir = str(Path.home() / f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%")
        
        if self.output_dir is None:
            self.output_dir = str(Path.home() / f"results/dpo_{self.hf_model_id}")


def compute_log_probs(model, input_ids, attention_mask, prompt_mask=None):
    """Compute per-token log probabilities for completion tokens only.
    
    Args:
        model: Language model
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        prompt_mask: [batch_size, seq_len] - 1 for completion tokens, 0 for prompt/padding
    
    Returns:
        log_probs: [batch_size] - sum of log probs for completion tokens only
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
    labels = input_ids[:, 1:]  # [batch_size, seq_len-1]
    
    # Compute per-token log probs
    log_probs_dist = F.log_softmax(logits, dim=-1)
    log_probs_per_token = torch.gather(
        log_probs_dist, 2, labels.unsqueeze(-1)
    ).squeeze(-1)  # [batch_size, seq_len-1]
    
    # Create mask for completion tokens only (excluding prompt and padding)
    if prompt_mask is not None:
        # Shift prompt_mask to align with labels (which start at position 1)
        completion_mask = prompt_mask[:, 1:].float()
    else:
        # Fallback: use attention mask (will include prompt, not ideal for DPO)
        completion_mask = attention_mask[:, 1:].float()
    
    # Only sum log probs for completion tokens
    log_probs = (log_probs_per_token * completion_mask).sum(dim=-1)  # [batch_size]
    
    return log_probs


def create_prompt_masks(input_ids, attention_mask, prompt_lens, debug=False):
    """Create masks that identify completion tokens (1) vs prompt/padding (0).
    
    Args:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        prompt_lens: list of prompt lengths for each item in batch
        debug: whether to print debug information
    
    Returns:
        prompt_mask: [batch_size, seq_len] - 1 for completion, 0 for prompt/padding
    """
    batch_size, seq_len = input_ids.shape
    prompt_mask = torch.zeros_like(input_ids, dtype=torch.float)
    
    for i, plen in enumerate(prompt_lens):
        # Mark completion tokens (from prompt_len to end of actual content)
        # attention_mask tells us where actual content ends
        seq_length = attention_mask[i].sum().item()
        if plen < seq_length:
            prompt_mask[i, plen:int(seq_length)] = 1.0
        
        if debug and i == 0:
            print(f"\n=== DEBUG: Prompt mask creation ===")
            print(f"Batch item {i}:")
            print(f"  Prompt length: {plen}")
            print(f"  Sequence length (non-padding): {seq_length}")
            print(f"  Completion tokens: {int(seq_length) - plen}")
            print(f"  Prompt mask sum: {prompt_mask[i].sum().item()}")
    
    return prompt_mask


def disable_dropout_in_model(model):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def train(config: DPOConfig):
    """Main DPO training function following Algorithm 1."""
    
    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.use_wandb else None,
    )
    
    if config.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project="dpo-training", config=vars(config))
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("Direct Preference Optimization (DPO) Training")
        print("=" * 80)
        print(f"Policy model: {config.sft_model_path or config.hf_model_id}")
        print(f"Dataset: {config.rm_dataset_config.dataset_name}")
        print(f"Beta: {config.beta}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Output directory: {config.output_dir}")
        print(f"Debug mode: {config.debug}")
        print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path or config.hf_model_id,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare datasets
    marker_file = Path(config.data_dir) / ".data_ready"
    if accelerator.is_main_process:
        prepare_rm_datasets(config.rm_dataset_config, config.data_dir)
        marker_file.touch()
    
    accelerator.wait_for_everyone()
    
    # Create datasets with debug flag
    train_dataset = DPODataset(
        Path(config.data_dir) / "training.jsonl",
        tokenizer,
        config.max_seq_length,
        debug=config.debug and accelerator.is_main_process,
    )
    
    eval_dataset = DPODataset(
        Path(config.data_dir) / "validation.jsonl",
        tokenizer,
        config.max_seq_length,
        debug=False,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    
    # Initialize policy model π_θ from π_ref
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
            print("Merging SFT PEFT adapter into base model...")
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
    
    # Apply LoRA if not full fine-tuning
    if not config.full_ft:
        if accelerator.is_main_process:
            print("Applying LoRA to policy model...")
        
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
        )
        policy_model = get_peft_model(policy_model, peft_config)
        policy_model = policy_model.to(torch.bfloat16)
        
        for name, param in policy_model.named_parameters():
            if 'lora_B' in name and param.requires_grad:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

        if accelerator.is_main_process:
            policy_model.print_trainable_parameters()

    
    # Initialize frozen reference model π_ref
    if accelerator.is_main_process:
        print("Loading reference model (frozen)...")
    
    if is_peft_checkpoint(config.sft_model_path):
        base_model_path = config.hf_model_id
        base_model_file = Path(config.sft_model_path) / "base_model.txt"
        if base_model_file.exists():
            base_model_path = base_model_file.read_text().strip()
        
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=True,
        )
        ref_model = PeftModel.from_pretrained(ref_model, config.sft_model_path)
        ref_model = ref_model.merge_and_unload()
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.sft_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=True,
        )
    
    # Disable dropout in reference model for consistency
    disable_dropout_in_model(ref_model)
    
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    
    # Prepare both models together with accelerator
    policy_model, ref_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        policy_model, ref_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Make sure reference model stays frozen even after prepare
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    if accelerator.is_main_process:
        print("Models prepared with accelerator")
        print(f"Policy model device: {next(policy_model.parameters()).device}")
        print(f"Reference model device: {next(ref_model.parameters()).device}")
    
    # Training loop
    global_step = 0
    best_eval_loss = float('inf')
    patience_counter = 0
    policy_model.train()

    if accelerator.is_main_process:
        print("\nStarting DPO training...")
        print("=" * 80)
    
    dataloader_iterator = iter(train_dataloader)
    
    while global_step < config.max_steps:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_dataloader)
            batch = next(dataloader_iterator)
        
        with accelerator.accumulate(policy_model):
            # Create prompt masks to identify completion tokens
            prompt_mask_chosen = create_prompt_masks(
                batch['input_ids_chosen'],
                batch['attention_mask_chosen'],
                batch['prompt_lens'],
                debug=config.debug  and accelerator.is_main_process
            )
            prompt_mask_rejected = create_prompt_masks(
                batch['input_ids_rejected'],
                batch['attention_mask_rejected'],
                batch['prompt_lens'],
                debug=False
            )
            
            # Debug: Print on first step
            if config.debug and accelerator.is_main_process:
                print(f"\n=== DEBUG: First batch ===")
                print(f"Prompt lens: {batch['prompt_lens']}")
                print(f"Chosen shape: {batch['input_ids_chosen'].shape}")
                print(f"Rejected shape: {batch['input_ids_rejected'].shape}")
                print(f"Prompt mask chosen sum: {prompt_mask_chosen.sum(dim=1)}")
                print(f"Prompt mask rejected sum: {prompt_mask_rejected.sum(dim=1)}")
            
            # Forward pass through policy model π_θ (only on completion tokens)
            log_pi_theta_chosen = compute_log_probs(
                policy_model,
                batch['input_ids_chosen'],
                batch['attention_mask_chosen'],
                prompt_mask_chosen
            )
            log_pi_theta_rejected = compute_log_probs(
                policy_model,
                batch['input_ids_rejected'],
                batch['attention_mask_rejected'],
                prompt_mask_rejected
            )
            
            # Forward pass through reference model π_ref (no gradients, only on completion tokens)
            with torch.no_grad():
                log_pi_ref_chosen = compute_log_probs(
                    ref_model,
                    batch['input_ids_chosen'],
                    batch['attention_mask_chosen'],
                    prompt_mask_chosen
                )
                log_pi_ref_rejected = compute_log_probs(
                    ref_model,
                    batch['input_ids_rejected'],
                    batch['attention_mask_rejected'],
                    prompt_mask_rejected
                )
            
            if config.debug and accelerator.is_main_process:
                print(f"\n=== DEBUG: Log probabilities ===")
                print(f"log_pi_theta_chosen: {log_pi_theta_chosen}")
                print(f"log_pi_theta_rejected: {log_pi_theta_rejected}")
                print(f"log_pi_ref_chosen: {log_pi_ref_chosen}")
                print(f"log_pi_ref_rejected: {log_pi_ref_rejected}")
            
            # Compute implicit rewards (log ratios)
            r_theta_chosen = config.beta * (log_pi_theta_chosen - log_pi_ref_chosen)
            r_theta_rejected = config.beta * (log_pi_theta_rejected - log_pi_ref_rejected)
            
            if config.debug and accelerator.is_main_process:
                print(f"\n=== DEBUG: Rewards ===")
                print(f"r_theta_chosen: {r_theta_chosen}")
                print(f"r_theta_rejected: {r_theta_rejected}")
                print(f"Reward margin: {(r_theta_chosen - r_theta_rejected).mean()}")
            
            # DPO loss: -log(σ(r_θ(x, y_w) - r_θ(x, y_l)))
            loss = -F.logsigmoid(r_theta_chosen - r_theta_rejected).mean()
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            
            # Logging
            if global_step % config.logging_steps == 0 and accelerator.is_main_process:
                # Compute accuracy (how often chosen is preferred)
                with torch.no_grad():
                    accuracy = ((r_theta_chosen - r_theta_rejected) > 0).float().mean()
                
                print(f"Step {global_step}/{config.max_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Accuracy: {accuracy.item():.4f} | "
                      f"Reward Margin: {(r_theta_chosen - r_theta_rejected).mean().item():.4f}")
                
                if config.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/accuracy": accuracy.item(),
                        "train/reward_margin": (r_theta_chosen - r_theta_rejected).mean().item(),
                        "train/reward_chosen": r_theta_chosen.mean().item(),
                        "train/reward_rejected": r_theta_rejected.mean().item(),
                        "global_step": global_step,
                    })
            
            # Evaluation
            if global_step % config.eval_steps == 0:
                policy_model.eval()
                eval_losses = []
                eval_accuracies = []
                
                with torch.no_grad():
                    for eval_batch in eval_dataloader:
                        # Create prompt masks for evaluation
                        eval_prompt_mask_chosen = create_prompt_masks(
                            eval_batch['input_ids_chosen'],
                            eval_batch['attention_mask_chosen'],
                            eval_batch['prompt_lens']
                        )
                        eval_prompt_mask_rejected = create_prompt_masks(
                            eval_batch['input_ids_rejected'],
                            eval_batch['attention_mask_rejected'],
                            eval_batch['prompt_lens']
                        )
                        
                        log_pi_theta_chosen = compute_log_probs(
                            policy_model,
                            eval_batch['input_ids_chosen'],
                            eval_batch['attention_mask_chosen'],
                            eval_prompt_mask_chosen
                        )
                        log_pi_theta_rejected = compute_log_probs(
                            policy_model,
                            eval_batch['input_ids_rejected'],
                            eval_batch['attention_mask_rejected'],
                            eval_prompt_mask_rejected
                        )
                        
                        log_pi_ref_chosen = compute_log_probs(
                            ref_model,
                            eval_batch['input_ids_chosen'],
                            eval_batch['attention_mask_chosen'],
                            eval_prompt_mask_chosen
                        )
                        log_pi_ref_rejected = compute_log_probs(
                            ref_model,
                            eval_batch['input_ids_rejected'],
                            eval_batch['attention_mask_rejected'],
                            eval_prompt_mask_rejected
                        )
                        
                        r_theta_chosen = config.beta * (log_pi_theta_chosen - log_pi_ref_chosen)
                        r_theta_rejected = config.beta * (log_pi_theta_rejected - log_pi_ref_rejected)
                        
                        eval_loss = -F.logsigmoid(r_theta_chosen - r_theta_rejected).mean()
                        eval_accuracy = ((r_theta_chosen - r_theta_rejected) > 0).float().mean()
                        
                        eval_losses.append(eval_loss.item())
                        eval_accuracies.append(eval_accuracy.item())
                
                avg_eval_loss = sum(eval_losses) / len(eval_losses)
                avg_eval_accuracy = sum(eval_accuracies) / len(eval_accuracies)
                
                if accelerator.is_main_process:
                    print(f"Eval at step {global_step} | "
                          f"Loss: {avg_eval_loss:.4f} | "
                          f"Accuracy: {avg_eval_accuracy:.4f}")
                    
                    if config.use_wandb:
                        wandb.log({
                            "eval/loss": avg_eval_loss,
                            "eval/accuracy": avg_eval_accuracy,
                            "global_step": global_step,
                        })
                
                # Early stopping check
                if avg_eval_loss < best_eval_loss - config.early_stopping_threshold:
                    best_eval_loss = avg_eval_loss
                    patience_counter = 0
                    
                    # Save best model
                    accelerator.wait_for_everyone()
                    save_path = Path(config.output_dir) / "best_model"
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    accelerator.save_model(policy_model, save_path)
                    
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(save_path)
                        print(f"New best model saved (loss: {best_eval_loss:.4f})")
                    
                    accelerator.wait_for_everyone()
                else:
                    patience_counter += 1
                    if accelerator.is_main_process:
                        print(f"No improvement. Patience: {patience_counter}/{config.early_stopping_patience}")
                    
                    if patience_counter >= config.early_stopping_patience:
                        if accelerator.is_main_process:
                            print(f"Early stopping triggered after {global_step} steps")
                        break
                
                policy_model.train()
            
           # Save checkpoint
            if global_step % config.save_steps == 0:
                accelerator.wait_for_everyone()
                
                save_path = Path(config.output_dir) / f"checkpoint-{global_step}"
                save_path.mkdir(parents=True, exist_ok=True)
                
                accelerator.save_model(policy_model, save_path)
                
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")
                
                accelerator.wait_for_everyone()
                
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("=" * 80)
        print("Training completed! Saving final model...")
        print("=" * 80)

    final_path = Path(config.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    accelerator.save_model(policy_model, final_path)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_path)
        print(f"Final model saved to {final_path}")

    accelerator.wait_for_everyone()


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Generate argparse parser from dataclass fields."""
    parser = argparse.ArgumentParser(description="Train Policy Model with DPO")
    
    for f in fields(dataclass_type):
        if f.name == 'rm_dataset_config':
            for rm_field in fields(RMDatasetConfig):
                if rm_field.name in ['custom_converter', 'load_kwargs']:
                    continue
                arg_name = f'rmdc_{rm_field.name}'
                field_type = rm_field.type
                default_value = rm_field.default if rm_field.default is not MISSING else None
                
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
    parser = create_parser_from_dataclass(DPOConfig)
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = DPOConfig.from_args(args)
    train(config)


if __name__ == "__main__":
    main()
