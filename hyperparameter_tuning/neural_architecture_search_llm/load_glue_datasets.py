# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import logging
import numpy as np

from datasets import load_dataset
from torch.utils.data import DataLoader, Subset

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

logger = logging.getLogger(__name__)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_glue_datasets(training_args, model_args, data_args):
    raw_datasets = load_dataset(
        "glue", data_args.task_name, cache_dir=model_args.cache_dir
    )

    # Load tokenizer
    model_type = model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_type,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_type.startswith("gpt2"):
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets[
        "validation_matched" if data_args.task_name == "mnli" else "validation"
    ]

    train_dataset = train_dataset.remove_columns(["idx"])
    test_dataset = test_dataset.remove_columns(["idx"])

    # Split training dataset in training / validation
    split = train_dataset.train_test_split(
        train_size=0.7, seed=training_args.seed
    )  # fix seed, all trials have the same data split
    train_dataset = split["train"]
    valid_dataset = split["test"]

    if data_args.task_name in ["sst2", "qqp", "qnli", "mnli"]:
        valid_dataset = Subset(
            valid_dataset,
            np.random.choice(len(valid_dataset), 2048).tolist(),
        )

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        valid_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, eval_dataloader, test_dataloader
