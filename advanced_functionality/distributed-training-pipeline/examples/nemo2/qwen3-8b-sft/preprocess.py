"""Preprocess training data for NeMo 2.0 SFT pipeline.

Downloads HuggingFace dataset, splits into train/val/test,
and writes JSONL files for use by fine_tune.py.
"""

import os
import sys
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from pathlib import Path
from datasets import load_dataset, DatasetDict


@dataclass
class HFDatasetConfig:
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


def _extract_template_fields(config):
    pattern = r'\{(\w+)\}'
    input_fields = set(re.findall(pattern, config.input_template))
    output_fields = set(re.findall(pattern, config.output_template))
    return input_fields | output_fields


def _convert_sample(sample, config):
    if config.field_mapping is not None:
        mapped_sample = {
            placeholder: sample.get(config.field_mapping.get(placeholder, placeholder), "")
            for placeholder in _extract_template_fields(config)
        }
    else:
        mapped_sample = sample
    input_text = config.input_template.format(**mapped_sample)
    output_text = config.output_template.format(**mapped_sample)
    return {"input": input_text, "output": output_text}


def _convert_to_jsonl(dataset, path, config):
    with open(path, "w", encoding='utf-8') as f:
        for sample in dataset:
            converted = _convert_sample(sample, config)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")


def prepare_datasets(config, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    marker_file = data_dir / ".data_ready"

    if marker_file.exists():
        print("Dataset already prepared.")
        return

    dataset = load_dataset(config.dataset_name, config.dataset_config,
                           num_proc=config.num_proc, **config.load_kwargs)
    initial_data = dataset[config.split]

    train_testval = initial_data.train_test_split(test_size=1.0 - config.train_split_ratio)
    test_val = train_testval['test'].train_test_split(test_size=config.val_test_split_ratio)
    split_dataset = DatasetDict({
        'train': train_testval['train'],
        'val': test_val['train'],
        'test': test_val['test']
    })

    print(f"Train: {len(split_dataset['train'])}, Val: {len(split_dataset['val'])}, Test: {len(split_dataset['test'])}")

    _convert_to_jsonl(split_dataset['train'], data_dir / "training.jsonl", config)
    _convert_to_jsonl(split_dataset['val'], data_dir / "validation.jsonl", config)
    _convert_to_jsonl(split_dataset['test'], data_dir / "test.jsonl", config)

    marker_file.write_text('ready')
    print("Dataset preparation complete!")


config = HFDatasetConfig(
    dataset_name="cognitivecomputations/dolphin",
    dataset_config="flan1m-alpaca-uncensored",
    split="train",
    train_split_ratio=0.9,
    val_test_split_ratio=0.5,
    input_template="### Instruction:\n{instruction}\n ### Input:\n{input}\n",
    output_template="### Response:\n{output}",
    field_mapping={
        "instruction": "instruction",
        "input": "input",
        "output": "output",
    },
    num_proc=8,
)

data_dir = os.environ.get("DATA_DIR", os.path.join(os.environ.get("HOME", "/tmp"), "data"))
prepare_datasets(config, data_dir)
