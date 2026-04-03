import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


@dataclass
class HFDatasetConfig:
    """Configuration for loading and converting HuggingFace datasets."""
    
    # Dataset loading parameters
    dataset_name: str
    """HuggingFace dataset name (e.g., 'cognitivecomputations/dolphin')"""
    
    dataset_config: Optional[str] = None
    """Dataset configuration/subset name (e.g., 'flan1m-alpaca-uncensored')"""
    
    split: str = "train"
    """Initial split to load from HuggingFace"""
    
    # Train/val/test split configuration
    train_split_ratio: float = 0.9
    """Ratio of data to use for training (remaining is split between val and test)"""
    
    val_test_split_ratio: float = 0.5
    """Ratio to split remaining data between validation and test"""
    
    # Data conversion parameters
    input_template: str = "### Instruction:\n{instruction}\n ### Input:\n{input}\n"
    """Template for formatting input. Use {field_name} for dataset field placeholders"""
    
    output_template: str = "### Response:\n{output}"
    """Template for formatting output. Use {field_name} for dataset field placeholders"""
    
    field_mapping: Optional[Dict[str, str]] = None
    """Mapping from template placeholders to actual dataset column names.
    Example: {'instruction': 'text', 'input': 'context', 'output': 'answer'}
    If None, assumes template placeholders match dataset column names exactly."""
    
    # Additional loading parameters
    num_proc: int = 8
    """Number of processes for dataset loading"""
    
    cache_dir: Optional[str] = None
    """Directory to cache the downloaded dataset"""
    
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to load_dataset"""
    
    # Custom conversion function (advanced usage)
    custom_converter: Optional[Callable] = None
    """Optional custom function to convert a dataset sample to input/output dict.
    Should have signature: func(sample: Dict) -> Dict[str, str] with keys 'input' and 'output'"""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.train_split_ratio <= 0 or self.train_split_ratio >= 1:
            raise ValueError(
                f"train_split_ratio must be between 0 and 1, got {self.train_split_ratio}"
            )
        if self.val_test_split_ratio < 0 or self.val_test_split_ratio > 1:
            raise ValueError(
                f"val_test_split_ratio must be between 0 and 1, got {self.val_test_split_ratio}"
            )


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning with tokenization."""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
        is_test: bool = False,
        min_output_tokens: int = 1,
    ):
        """
        Initialize SFT Dataset.
        
        Args:
            data_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length
            is_test: Whether this is a test dataset
            min_output_tokens: Minimum number of output tokens required (prevents all-masked samples)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_test = is_test
        self.min_output_tokens = min_output_tokens
        
        # OPTIMIZATION: Load all data at once instead of line-by-line
        print(f"Loading data from {data_path.name}...", end=" ", flush=True)
        start_time = time.time()
        
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # OPTIMIZATION: Parse JSON in parallel using list comprehension
        self.samples = []
        skipped_count = 0
        
        for line_num, line in enumerate(lines, 1):
            try:
                sample = json.loads(line)
                if self._validate_sample(sample):
                    self.samples.append(sample)
                else:
                    skipped_count += 1
            except json.JSONDecodeError:
                skipped_count += 1
            except Exception:
                skipped_count += 1
        
        load_time = time.time() - start_time
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples loaded from {data_path}")
        
        print(f"✓ Loaded {len(self.samples)} samples in {load_time:.2f}s", end="")
        if skipped_count > 0:
            print(f" (skipped {skipped_count})")
        else:
            print()
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that sample will have enough output tokens.
        Output is never truncated, so we only check the raw output length.
        
        Args:
            sample: Sample dictionary with 'input' and 'output' keys
            
        Returns:
            True if sample is valid, False otherwise
        """
        # Check that required keys exist
        if 'input' not in sample or 'output' not in sample:
            return False
        
        # Check that output is not empty
        if not sample['output'] or not sample['output'].strip():
            return False
        
        try:
            # OPTIMIZATION: Quick length check before tokenizing
            # Approximate: 1 token ≈ 4 characters (conservative)
            output_char_count = len(sample['output'])
            estimated_tokens = output_char_count / 4
            
            # If estimated tokens are way too long, skip expensive tokenization
            if estimated_tokens >= self.max_seq_length:
                return False
            
            # If estimated tokens are way too short, skip expensive tokenization
            if estimated_tokens < self.min_output_tokens * 0.5:  # Conservative estimate
                return False
            
            # Only tokenize if we're in the reasonable range
            output_tokenized = self.tokenizer(
                sample['output'],
                truncation=False,
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )
            
            output_length = len(output_tokenized['input_ids'])
            
            # Reject if output alone exceeds max sequence length
            if output_length >= self.max_seq_length:
                return False
            
            # Ensure we have at least min_output_tokens
            return output_length >= self.min_output_tokens
            
        except Exception:
            return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input and output separately
        input_tokenized = self.tokenizer(
            sample['input'],
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,  # Add BOS token
        )
        
        output_tokenized = self.tokenizer(
            sample['output'],
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        
        input_ids_list = input_tokenized['input_ids']
        output_ids_list = output_tokenized['input_ids']
        
        # Calculate lengths
        input_length = len(input_ids_list)
        output_length = len(output_ids_list)
        total_length = input_length + output_length
        
        # Truncate from the LEFT of the input if necessary to preserve output
        if total_length > self.max_seq_length:
            # Calculate how much input we can keep
            max_input_length = self.max_seq_length - output_length
            
            if max_input_length < 1:
                # Output alone is too long - truncate output from right as last resort
                output_ids_list = output_ids_list[:self.max_seq_length - 1]
                input_ids_list = input_ids_list[:1]  # Keep at least BOS token
                input_length = 1
                output_length = len(output_ids_list)
            else:
                # Truncate input from the LEFT (keep the most recent context)
                input_ids_list = input_ids_list[-max_input_length:]
                input_length = len(input_ids_list)
        
        # Combine input and output
        input_ids = input_ids_list + output_ids_list
        
        # Create labels: mask input portion, keep output portion
        labels = ([-100] * input_length) + output_ids_list
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


class GeneralizedHFDataModule(pl.LightningDataModule):
    """Pure PyTorch Lightning data module for HuggingFace datasets."""
    
    def __init__(
        self,
        config: HFDatasetConfig,
        dataset_root: str,
        tokenizer_name: str,
        max_seq_length: int = 2048,
        micro_batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        min_output_tokens: int = 1,
    ):
        """
        Initialize the data module.
        
        Args:
            config: HFDatasetConfig instance with dataset configuration
            dataset_root: Root directory to store converted datasets
            tokenizer_name: HuggingFace tokenizer name/path
            max_seq_length: Maximum sequence length
            micro_batch_size: Batch size per device
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            persistent_workers: Whether to keep workers persistent
            min_output_tokens: Minimum output tokens required per sample
        """
        super().__init__()
        self.config = config
        self.dataset_root = Path(dataset_root)
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.min_output_tokens = min_output_tokens
        
        # Create dataset directory
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer (will be set in setup)
        self.tokenizer = None
    
    def _convert_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Convert a single dataset sample to input/output format.
        
        Args:
            sample: A dictionary containing the dataset sample
            
        Returns:
            Dictionary with 'input' and 'output' keys, or None if conversion fails
        """
        # Use custom converter if provided
        if self.config.custom_converter is not None:
            try:
                result = self.config.custom_converter(sample)
                # Validate result
                if result is None or 'input' not in result or 'output' not in result:
                    return None
                return result
            except Exception as e:
                print(f"⚠ Custom converter failed: {e}")
                return None
        
        # Apply field mapping if provided
        if self.config.field_mapping is not None:
            mapped_sample = {}
            for placeholder in self._extract_template_fields():
                field_name = self.config.field_mapping.get(placeholder, placeholder)
                value = sample.get(field_name, "")
                
                # Handle None values
                if value is None:
                    value = ""
                
                mapped_sample[placeholder] = str(value)
        else:
            # Convert all values to strings and handle None
            mapped_sample = {k: str(v) if v is not None else "" for k, v in sample.items()}
        
        # Format input and output using templates
        try:
            input_text = self.config.input_template.format(**mapped_sample)
            output_text = self.config.output_template.format(**mapped_sample)
        except KeyError as e:
            return None
        except Exception as e:
            return None
        
        # Validate that output is not empty
        if not output_text.strip():
            return None
        
        return {"input": input_text, "output": output_text}
    
    def _extract_template_fields(self) -> set:
        """Extract field names from templates."""
        pattern = r'\{(\w+)\}'
        
        input_fields = set(re.findall(pattern, self.config.input_template))
        output_fields = set(re.findall(pattern, self.config.output_template))
        
        return input_fields | output_fields
    
    def _convert_hf_dataset_to_jsonl(self, dataset, path: Path):
        """
        Convert HuggingFace dataset to JSONL format, skipping invalid samples.
        
        Args:
            dataset: HuggingFace dataset or dataset split
            path: Output path for JSONL file
        """
        print(f"  Converting {len(dataset)} samples...", end=" ", flush=True)
        start_time = time.time()
        
        converted_count = 0
        skipped_count = 0
        
        # OPTIMIZATION: Batch write to file
        json_lines = []
        
        for sample in dataset:
            converted = self._convert_sample(sample)
            if converted is not None:
                try:
                    json_string = json.dumps(converted, ensure_ascii=False)
                    json_lines.append(json_string)
                    converted_count += 1
                except Exception:
                    skipped_count += 1
            else:
                skipped_count += 1
        
        # Write all at once
        with open(path, "w", encoding='utf-8') as f:
            f.write('\n'.join(json_lines) + '\n')
        
        convert_time = time.time() - start_time
        
        skip_pct = 100 * skipped_count / (converted_count + skipped_count) if (converted_count + skipped_count) > 0 else 0
        print(f"✓ {converted_count} samples in {convert_time:.2f}s (skipped {skipped_count}, {skip_pct:.1f}%)")
        
        if converted_count == 0:
            raise ValueError(f"No valid samples were converted for {path.name}")
    
    def _load_and_split_dataset(self) -> DatasetDict:
        """
        Load dataset from HuggingFace and split into train/val/test.
        
        Returns:
            DatasetDict with 'train', 'val', and 'test' splits
        """
        # Prepare load_dataset kwargs
        load_kwargs = self.config.load_kwargs.copy()
        if self.config.cache_dir:
            load_kwargs['cache_dir'] = self.config.cache_dir
        
        # Load dataset
        print(f"Loading dataset '{self.config.dataset_name}'...")
        if self.config.dataset_config:
            print(f"  Config: {self.config.dataset_config}")
        
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            num_proc=self.config.num_proc,
            **load_kwargs
        )
        
        # Get the initial split
        if self.config.split not in dataset:
            raise ValueError(
                f"Split '{self.config.split}' not found in dataset. "
                f"Available splits: {list(dataset.keys())}"
            )
        
        initial_data = dataset[self.config.split]
        print(f"  Initial split '{self.config.split}' size: {len(initial_data)}")
        
        # Split into train and (val+test)
        train_testval = initial_data.train_test_split(
            test_size=1.0 - self.config.train_split_ratio,
            seed=42,  # For reproducibility
        )
        
        # Split (val+test) into val and test
        test_val = train_testval['test'].train_test_split(
            test_size=self.config.val_test_split_ratio,
            seed=42,  # For reproducibility
        )
        
        # Create final DatasetDict
        split_dataset = DatasetDict({
            'train': train_testval['train'],
            'val': test_val['train'],
            'test': test_val['test']
        })
        
        print(f"  Split sizes - Train: {len(split_dataset['train'])}, Val: {len(split_dataset['val'])}, Test: {len(split_dataset['test'])}")
        
        return split_dataset
    
    @property
    def train_path(self) -> Path:
        """Path to training dataset file"""
        return self.dataset_root / "training.jsonl"
    
    @property
    def validation_path(self) -> Path:
        """Path to validation dataset file"""
        return self.dataset_root / "validation.jsonl"
    
    @property
    def test_path(self) -> Path:
        """Path to test dataset file"""
        return self.dataset_root / "test.jsonl"
    
    def prepare_data(self):
        """Prepare data by converting HuggingFace dataset to JSONL format if needed."""
        marker_file = os.path.join(os.path.dirname(self.train_path), ".data_ready")

        if self.trainer is None or self.trainer.is_global_zero:
            if not os.path.exists(marker_file):
                print("\n" + "=" * 80)
                print("Preparing dataset...")
                print("=" * 80)
                
                overall_start = time.time()
                
                hf_dataset = self._load_and_split_dataset()
                
                print(f"\nConverting to JSONL format...")
                print(f"Training set:")
                self._convert_hf_dataset_to_jsonl(hf_dataset['train'], self.train_path)
                
                print(f"Validation set:")
                self._convert_hf_dataset_to_jsonl(hf_dataset['val'], self.validation_path)
                
                print(f"Test set:")
                self._convert_hf_dataset_to_jsonl(hf_dataset['test'], self.test_path)
                
                overall_time = time.time() - overall_start
                
                # Create marker file
                with open(marker_file, 'w') as f:
                    f.write('ready')
                
                print("=" * 80)
                print(f"Dataset preparation complete in {overall_time:.2f}s!")
                print("=" * 80 + "\n")
            else:
                print(f"✓ Dataset already prepared at {self.dataset_root}")
        else:
            print(f"Rank {self.trainer.global_rank} waiting for data preparation...")
            # OPTIMIZATION: Check less frequently (every 5s instead of 10s)
            while not os.path.exists(marker_file):
                time.sleep(5)
            print(f"Rank {self.trainer.global_rank} - data ready!")
        
        super().prepare_data()
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        # OPTIMIZATION: Load tokenizer only once
        if self.tokenizer is None:
            print(f"Loading tokenizer: {self.tokenizer_name}...", end=" ", flush=True)
            start_time = time.time()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,
                trust_remote_code=True,
            )
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_time = time.time() - start_time
            print(f"✓ Loaded in {load_time:.2f}s")
        
        # Create datasets
        if stage == "fit" or stage is None:
            print("\nSetting up datasets...")
            setup_start = time.time()
            
            self.train_dataset = SFTDataset(
                self.train_path,
                self.tokenizer,
                self.max_seq_length,
                is_test=False,
                min_output_tokens=self.min_output_tokens,
            )
            self.val_dataset = SFTDataset(
                self.validation_path,
                self.tokenizer,
                self.max_seq_length,
                is_test=True,
                min_output_tokens=self.min_output_tokens,
            )
            
            setup_time = time.time() - setup_start
            
            # Print dataset statistics
            print("\n" + "=" * 80)
            print("Dataset Statistics")
            print("=" * 80)
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
            print(f"Max sequence length: {self.max_seq_length}")
            print(f"Micro batch size: {self.micro_batch_size}")
            print(f"Setup time: {setup_time:.2f}s")
            print("=" * 80 + "\n")
        
        if stage == "test" or stage is None:
            self.test_dataset = SFTDataset(
                self.test_path,
                self.tokenizer,
                self.max_seq_length,
                is_test=True,
                min_output_tokens=self.min_output_tokens,
            )
    
    def collate_fn(self, batch):
        """Collate function to pad batch."""
        # Extract fields
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
        
        # Pad sequences
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        return {
            'input_ids': input_ids_padded,
            'labels': labels_padded,
            'attention_mask': attention_mask_padded,
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
        )