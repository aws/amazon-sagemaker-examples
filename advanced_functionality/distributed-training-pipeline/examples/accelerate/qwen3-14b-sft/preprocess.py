"""Preprocess training data for SFT pipeline.

Downloads HuggingFace dataset, splits into train/val/test,
and writes JSONL files for use by peft_accelerate.py.
"""

import os
import sys

sys.path.insert(0, ".")

from text.dataset_module import HFDatasetConfig, prepare_datasets

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
print("Preprocessing complete!")
