import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class HFDatasetConfig:
    """Configuration for loading and converting HuggingFace datasets."""
    
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = "train"
    train_split_ratio: float = 0.9
    val_test_split_ratio: float = 0.5
    input_template: str = "### Instruction:\n{instruction}\n ### Input:\n{input}\n"
    output_template: str = "### Response:\n{output}"
    field_mapping: Optional[Dict[str, str]] = None
    num_proc: int = 8
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    custom_converter: Optional[Callable] = None


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning with tokenization."""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
        is_test: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_test = is_test
        
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        full_text = sample['input'] + sample['output']
        
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None, # Returns python lists
            add_special_tokens=True,
        )
        
        input_ids = tokenized['input_ids']
        labels = list(input_ids) # Ensure list copy
        
        input_tokenized = self.tokenizer(
            sample['input'],
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        input_length = len(input_tokenized['input_ids'])
        
        # Mask the instruction/input part
        for i in range(min(input_length, len(labels))):
            labels[i] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': tokenized.get('attention_mask', [1] * len(input_ids))
        }


class CPTDataset(Dataset):
    """Dataset for Continual Pre-Training. All tokens are training targets (no label masking)."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 4096,
        is_test: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_test = is_test

        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # For CPT, concatenate input+output (input is typically empty)
        full_text = sample.get('input', '') + sample.get('output', '')

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )

        input_ids = tokenized['input_ids']
        # All tokens are training targets — no masking
        labels = list(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': tokenized.get('attention_mask', [1] * len(input_ids))
        }

def prepare_datasets(config: HFDatasetConfig, dataset_root: str):
    """Prepare datasets by converting HuggingFace dataset to JSONL format."""
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    marker_file = dataset_root / ".data_ready"
    
    if marker_file.exists():
        print("Dataset already prepared.")
        return
    
    print(f"Loading dataset '{config.dataset_name}'...")
    
    # Load dataset
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
    
    # Split into train and (val+test)
    train_testval = initial_data.train_test_split(
        test_size=1.0 - config.train_split_ratio
    )
    
    # Split (val+test) into val and test
    if config.val_test_split_ratio <= 0.0:
        # All remaining data goes to validation, none to test
        split_dataset = DatasetDict({
            'train': train_testval['train'],
            'val': train_testval['test'],
            'test': train_testval['test'].select([])
        })
    elif config.val_test_split_ratio >= 1.0:
        # All remaining data goes to test, none to validation
        split_dataset = DatasetDict({
            'train': train_testval['train'],
            'val': train_testval['test'].select([]),
            'test': train_testval['test']
        })
    else:
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
    
    # Convert to JSONL
    _convert_hf_dataset_to_jsonl(split_dataset['train'], dataset_root / "training.jsonl", config)
    _convert_hf_dataset_to_jsonl(split_dataset['val'], dataset_root / "validation.jsonl", config)
    _convert_hf_dataset_to_jsonl(split_dataset['test'], dataset_root / "test.jsonl", config)
    
    marker_file.write_text('ready')
    print("Dataset preparation complete!")


def _convert_hf_dataset_to_jsonl(dataset, path: Path, config: HFDatasetConfig):
    """Convert HuggingFace dataset to JSONL format."""
    with open(path, "w", encoding='utf-8') as f:
        for sample in dataset:
            converted = _convert_sample(sample, config)
            json_string = json.dumps(converted, ensure_ascii=False) + "\n"
            f.write(json_string)


def _convert_sample(sample: Dict[str, Any], config: HFDatasetConfig) -> Dict[str, str]:
    """Convert a single dataset sample to input/output format."""
    if config.custom_converter is not None:
        return config.custom_converter(sample)
    
    if config.field_mapping is not None:
        mapped_sample = {
            placeholder: sample.get(config.field_mapping.get(placeholder, placeholder), "")
            for placeholder in _extract_template_fields(config)
        }
    else:
        mapped_sample = sample
    
    try:
        input_text = config.input_template.format(**mapped_sample)
        output_text = config.output_template.format(**mapped_sample)
    except KeyError as e:
        raise KeyError(
            f"Missing field {e} in dataset sample. "
            f"Available fields: {list(sample.keys())}. "
            f"Consider using field_mapping to map template placeholders to dataset columns."
        )
    
    return {"input": input_text, "output": output_text}


def _extract_template_fields(config: HFDatasetConfig) -> set:
    """Extract field names from templates."""
    pattern = r'\{(\w+)\}'
    
    input_fields = set(re.findall(pattern, config.input_template))
    output_fields = set(re.findall(pattern, config.output_template))
    
    return input_fields | output_fields
