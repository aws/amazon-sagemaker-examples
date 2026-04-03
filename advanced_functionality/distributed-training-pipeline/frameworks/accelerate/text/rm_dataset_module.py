import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class RMDatasetConfig:
    """Configuration for loading and converting Reward Model datasets."""
    
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = "train"
    train_split_ratio: float = 0.9
    val_test_split_ratio: float = 0.5
    num_proc: int = 8
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    custom_converter: Optional[Callable] = None


class RewardModelDataset(Dataset):
    """Dataset for Reward Model training with chosen/rejected pairs.
    
    Supports two JSONL formats:
    1. Separate input: {"input": prompt, "chosen": chosen_response, "rejected": rejected_response}
    2. Combined format: {"chosen": full_chosen_text, "rejected": full_rejected_text}
    
    Common reward dataset conversion patterns:
    
    # Anthropic HH-RLHF (chosen/rejected fields)
    def hh_rlhf_converter(sample):
        return {"chosen": sample["chosen"], "rejected": sample["rejected"]}
    
    # With separate prompt
    def with_prompt_converter(sample):
        return {"input": sample["prompt"], "chosen": sample["chosen"], "rejected": sample["rejected"]}
    
    # OpenAssistant (messages with rank)
    def oasst_converter(sample):
        prompt = sample["messages"][0]["content"]
        chosen = sample["messages"][1]["content"] if sample["messages"][1]["rank"] == 0 else sample["messages"][2]["content"]
        rejected = sample["messages"][2]["content"] if sample["messages"][1]["rank"] == 0 else sample["messages"][1]["content"]
        return {"input": prompt, "chosen": chosen, "rejected": rejected}
    
    # Stanford SHP (score_A vs score_B)
    def shp_converter(sample):
        chosen = sample["human_ref_A"] if sample["labels"] == 1 else sample["human_ref_B"]
        rejected = sample["human_ref_B"] if sample["labels"] == 1 else sample["human_ref_A"]
        return {"input": sample["history"], "chosen": chosen, "rejected": rejected}
    """
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Handle both formats: with or without separate input field
        if 'input' in sample:
            chosen_text = sample['input'] + sample['chosen']
            rejected_text = sample['input'] + sample['rejected']
        else:
            chosen_text = sample['chosen']
            rejected_text = sample['rejected']
        
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
        
        return {
            'input_ids_chosen': chosen_tok['input_ids'],
            'attention_mask_chosen': chosen_tok.get('attention_mask', [1] * len(chosen_tok['input_ids'])),
            'input_ids_rejected': rejected_tok['input_ids'],
            'attention_mask_rejected': rejected_tok.get('attention_mask', [1] * len(rejected_tok['input_ids'])),
        }


def prepare_rm_datasets(config: RMDatasetConfig, dataset_root: str):
    """Prepare reward model datasets by converting HuggingFace dataset to JSONL format."""
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    marker_file = dataset_root / ".data_ready"
    
    if marker_file.exists():
        print("Dataset already prepared.")
        return
    
    print(f"Loading dataset '{config.dataset_name}'...")
    
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        num_proc=config.num_proc,
        **config.load_kwargs
    )
    
    if config.split not in dataset:
        raise ValueError(
            f"Split '{config.split}' not found in dataset. "
            f"Available splits: {list(dataset.keys())}"
        )
    
    initial_data = dataset[config.split]
    
    train_testval = initial_data.train_test_split(
        test_size=1.0 - config.train_split_ratio
    )
    
    test_val = train_testval['test'].train_test_split(
        test_size=config.val_test_split_ratio
    )
    
    split_dataset = DatasetDict({
        'train': train_testval['train'],
        'val': test_val['train'],
        'test': test_val['test']
    })
    
    print(f"Converting to JSONL format...")
    print(f"  Train samples: {len(split_dataset['train'])}")
    print(f"  Val samples: {len(split_dataset['val'])}")
    print(f"  Test samples: {len(split_dataset['test'])}")
    
    _convert_rm_dataset_to_jsonl(split_dataset['train'], dataset_root / "training.jsonl", config)
    _convert_rm_dataset_to_jsonl(split_dataset['val'], dataset_root / "validation.jsonl", config)
    _convert_rm_dataset_to_jsonl(split_dataset['test'], dataset_root / "test.jsonl", config)
    
    marker_file.write_text('ready')
    print("Dataset preparation complete!")


def _convert_rm_dataset_to_jsonl(dataset, path: Path, config: RMDatasetConfig):
    """Convert HuggingFace reward model dataset to JSONL format."""
    with open(path, "w", encoding='utf-8') as f:
        for sample in dataset:
            converted = _convert_rm_sample(sample, config)
            json_string = json.dumps(converted, ensure_ascii=False) + "\n"
            f.write(json_string)


def _convert_rm_sample(sample: Dict[str, Any], config: RMDatasetConfig) -> Dict[str, str]:
    """Convert a single reward model sample to chosen/rejected format."""
    if config.custom_converter is not None:
        return config.custom_converter(sample)
    
    # Default: expect chosen/rejected fields directly
    if 'chosen' in sample and 'rejected' in sample:
        result = {"chosen": sample["chosen"], "rejected": sample["rejected"]}
        if 'input' in sample or 'prompt' in sample:
            result["input"] = sample.get("input", sample.get("prompt", ""))
        return result
    
    raise ValueError(
        f"Cannot convert sample. Expected 'chosen' and 'rejected' fields. "
        f"Available fields: {list(sample.keys())}. "
        f"Use custom_converter for non-standard formats."
    )
