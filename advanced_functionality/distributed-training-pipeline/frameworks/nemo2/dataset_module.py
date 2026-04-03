import os
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from datasets import load_dataset, DatasetDict
import json
import time


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
    
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to load_dataset"""
    
    # Custom conversion function (advanced usage)
    custom_converter: Optional[Callable] = None
    """Optional custom function to convert a dataset sample to input/output dict.
    Should have signature: func(sample: Dict) -> Dict[str, str] with keys 'input' and 'output'"""


class GeneralizedHFDataModule(FineTuningDataModule):
    """Generalized data module for HuggingFace datasets compatible with NeMo 2.0."""
    
    def __init__(self, config: HFDatasetConfig, hf_model_id:str= None, **kwargs):
        """
        Initialize the data module.
        
        Args:
            config: HFDatasetConfig instance with dataset configuration
            **kwargs: Additional arguments passed to FineTuningDataModule
        """
        super().__init__(**kwargs)
        self.config = config
        assert kwargs["dataset_root"] is not None, "dataset_root must be provided"
        os.makedirs(kwargs["dataset_root"], exist_ok=True)
        self.tokenizer = get_nmt_tokenizer(
            library="huggingface",
            model_name=hf_model_id,
            use_fast=True,
        )
    
    def _convert_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert a single dataset sample to input/output format.
        
        Args:
            sample: A dictionary containing the dataset sample
            
        Returns:
            Dictionary with 'input' and 'output' keys
        """
        # Use custom converter if provided
        if self.config.custom_converter is not None:
            return self.config.custom_converter(sample)
        
        # Apply field mapping if provided
        if self.config.field_mapping is not None:
            mapped_sample = {
                placeholder: sample.get(self.config.field_mapping.get(placeholder, placeholder), "")
                for placeholder in self._extract_template_fields()
            }
        else:
            mapped_sample = sample
        
        # Format input and output using templates
        try:
            input_text = self.config.input_template.format(**mapped_sample)
            output_text = self.config.output_template.format(**mapped_sample)
        except KeyError as e:
            raise KeyError(
                f"Missing field {e} in dataset sample. "
                f"Available fields: {list(sample.keys())}. "
                f"Consider using field_mapping to map template placeholders to dataset columns."
            )
        
        return {"input": input_text, "output": output_text}
    
    def _extract_template_fields(self) -> set:
        """Extract field names from templates."""
        import re
        pattern = r'\{(\w+)\}'
        
        input_fields = set(re.findall(pattern, self.config.input_template))
        output_fields = set(re.findall(pattern, self.config.output_template))
        
        return input_fields | output_fields
    
    def _convert_hf_dataset_to_jsonl(self, dataset, path: str):
        """
        Convert HuggingFace dataset to JSONL format.
        
        Args:
            dataset: HuggingFace dataset or dataset split
            path: Output path for JSONL file
        """
        with open(path, "w", encoding='utf-8') as f:
            for sample in dataset:
                converted = self._convert_sample(sample)
                json_string = json.dumps(converted, ensure_ascii=False) + "\n"
                f.write(json_string)
    
    def _load_and_split_dataset(self) -> DatasetDict:
        """
        Load dataset from HuggingFace and split into train/val/test.
        
        Returns:
            DatasetDict with 'train', 'val', and 'test' splits
        """
        # Load dataset
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            num_proc=self.config.num_proc,
            **self.config.load_kwargs
        )
        
        # Get the initial split
        if self.config.split not in dataset:
            raise ValueError(
                f"Split '{self.config.split}' not found in dataset. "
                f"Available splits: {list(dataset.keys())}"
            )
        
        initial_data = dataset[self.config.split]
        
        # Split into train and (val+test)
        train_testval = initial_data.train_test_split(
            test_size=1.0 - self.config.train_split_ratio
        )
        
        # Split (val+test) into val and test
        test_val = train_testval['test'].train_test_split(
            test_size=self.config.val_test_split_ratio
        )
        
        # Create final DatasetDict
        split_dataset = DatasetDict({
            'train': train_testval['train'],
            'val': test_val['train'],
            'test': test_val['test']
        })
        
        return split_dataset
    
    def prepare_data(self):
        """Prepare data by converting HuggingFace dataset to JSONL format if needed."""
        marker_file = os.path.join(os.path.dirname(self.train_path), ".data_ready")

        if self.trainer is None or self.trainer.is_global_zero:
            if not os.path.exists(marker_file):
                print(f"Loading dataset '{self.config.dataset_name}'...")
                hf_dataset = self._load_and_split_dataset()
                
                print(f"Converting to JSONL format...")
                print(f"  Train samples: {len(hf_dataset['train'])}")
                print(f"  Val samples: {len(hf_dataset['val'])}")
                print(f"  Test samples: {len(hf_dataset['test'])}")
                
                self._convert_hf_dataset_to_jsonl(hf_dataset['train'], self.train_path)
                self._convert_hf_dataset_to_jsonl(hf_dataset['val'], self.validation_path)
                self._convert_hf_dataset_to_jsonl(hf_dataset['test'], self.test_path)
                
                with open(marker_file, 'w') as f:
                    f.write('ready')
            print("Dataset preparation complete!")
        else:
            print(f"Global rank: {self.trainer.global_rank} waiting for data preparation to complete...")
            while not os.path.exists(marker_file):
                time.sleep(10)
        
        super().prepare_data()
