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
from torch.utils.data import Subset

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

from task_data import GLUE_TASK_INFO

logger = logging.getLogger(__name__)


def load_imdb_dataset(training_args, model_args, data_args):

    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    sentence1_key, sentence2_key = "text", None

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
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    # Split training dataset in training / validation
    split = train_dataset.train_test_split(
        train_size=0.7, seed=training_args.seed
    )  # fix seed, all trials have the same data split
    train_dataset = split["train"]
    valid_dataset = split["test"]

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    return train_dataset, valid_dataset, test_dataset, data_collator
