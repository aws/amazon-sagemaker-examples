#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

# TODO Do multi-GPU and TPU tests and make sure the dataset length works as expected
# TODO Duplicate all changes over to the CLM script

import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import transformers
from transformers import (
    CONFIG_MAPPING,
    CONFIG_NAME,
    MODEL_FOR_MASKED_LM_MAPPING,
    TF2_WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForMaskedLM,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.utils.versions import require_version

os.environ["XLA_FLAGS"] = ""
os.environ["TF_XLA_FLAGS"] = ""
os.environ["XLA_FLAGS"] += " --xla_gpu_autotune_level=4"
os.environ["TF_XLA_FLAGS"] += " --tf_xla_auto_jit=2"

logger = logging.getLogger(__name__)
require_version(
    "datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt"
)
# MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_TYPES = (
    "fnet",
    "rembert",
    "roformer",
    "big_bird",
    "wav2vec2",
    "convbert",
    "ibert",
    "mobilebert",
    "distilbert",
    "albert",
    "camembert",
    "xlm-roberta",
    "mbart",
    "megatron-bert",
    "mpnet",
    "bart",
    "reformer",
    "longformer",
    "roberta",
    "deberta-v2",
    "deberta",
    "flaubert",
    "squeezebert",
    "bert",
    "xlm",
    "electra",
    "funnel",
    "layoutlm",
    "tapas",
)

# tf.config.optimizer.set_jit(True)

# region Command-line arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


# endregion


# region Helper classes
class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


# endregion

# region Data generator
def sample_generator(dataset, tokenizer, mlm_probability=0.15, pad_to_multiple_of=None):
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
        )
    # Trim off the last partial batch if present
    sample_ordering = np.random.permutation(len(dataset))
    for sample_idx in sample_ordering:
        example = dataset[int(sample_idx)]
        # Handle dicts with proper padding and conversion to tensor.
        example = tokenizer.pad(example, return_tensors="np", pad_to_multiple_of=pad_to_multiple_of)
        special_tokens_mask = example.pop("special_tokens_mask", None)
        example["input_ids"], example["labels"] = mask_tokens(
            example["input_ids"],
            mlm_probability,
            tokenizer,
            special_tokens_mask=special_tokens_mask,
        )
        if tokenizer.pad_token_id is not None:
            example["labels"][example["labels"] == tokenizer.pad_token_id] = -100
        example = {key: tf.convert_to_tensor(arr) for key, arr in example.items()}

        yield example, example["labels"]  # TF needs some kind of labels, even if we don't use them
    return


def mask_tokens(inputs, mlm_probability, tokenizer, special_tokens_mask):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = np.copy(inputs)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = np.random.random_sample(labels.shape)
    special_tokens_mask = special_tokens_mask.astype(np.bool_)

    probability_matrix[special_tokens_mask] = 0.0
    masked_indices = probability_matrix > (1 - mlm_probability)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (np.random.random_sample(labels.shape) < 0.8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        (np.random.random_sample(labels.shape) < 0.5) & masked_indices & ~indices_replaced
    )
    random_words = np.random.randint(
        low=0, high=len(tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
    )
    inputs[indices_random] = random_words

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


# endregion


def main():
    # region Argument Parsing
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sanity checks
    if (
        data_args.dataset_name is None
        and data_args.train_file is None
        and data_args.validation_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if data_args.train_file is not None:
            extension = data_args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, json or txt file."
        if data_args.validation_file is not None:
            extension = data_args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, json or txt file."

    if training_args.output_dir is not None:
        training_args.output_dir = Path(training_args.output_dir)
        os.makedirs(training_args.output_dir, exist_ok=True)

    if (
        isinstance(training_args.strategy, tf.distribute.TPUStrategy)
        and not data_args.pad_to_max_length
    ):
        logger.warning("We are training on TPU - forcing pad_to_max_length")
        data_args.pad_to_max_length = True
    # endregion

    # region Checkpoints
    # Detecting last checkpoint.
    checkpoint = None
    if len(os.listdir(training_args.output_dir)) > 0 and not training_args.overwrite_output_dir:
        config_path = training_args.output_dir / CONFIG_NAME
        weights_path = training_args.output_dir / TF2_WEIGHTS_NAME
        if config_path.is_file() and weights_path.is_file():
            checkpoint = training_args.output_dir
            logger.warning(
                f"Checkpoint detected, resuming training from checkpoint in {training_args.output_dir}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue regardless."
            )

    # endregion

    # region Setup logging
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    # endregion

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # region Load datasets
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # endregion

    # region Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if checkpoint is not None:
        config = AutoConfig.from_pretrained(checkpoint)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # endregion

    # region Dataset preprocessing
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can reduce that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset line_by_line",
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

    train_dataset = tokenized_datasets["train"]

    if data_args.validation_file is not None:
        eval_dataset = tokenized_datasets["validation"]
    else:
        logger.info(
            f"Validation file not found: using {data_args.validation_split_percentage}% of the dataset as validation as provided in data_args"
        )
        train_indices, val_indices = train_test_split(
            list(range(len(train_dataset))), test_size=data_args.validation_split_percentage / 100
        )

        eval_dataset = train_dataset.select(val_indices)
        train_dataset = train_dataset.select(train_indices)

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # endregion

    with training_args.strategy.scope():
        # region Prepare model
        if checkpoint is not None:
            model = TFAutoModelForMaskedLM.from_pretrained(checkpoint, config=config)
        elif model_args.model_name_or_path:
            model = TFAutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path, config=config
            )
        else:
            logger.info("Training new model from scratch")
            model = TFAutoModelForMaskedLM.from_config(config)

        model.resize_token_embeddings(len(tokenizer))
        # endregion

        # region TF Dataset preparation
        num_replicas = training_args.strategy.num_replicas_in_sync
        train_generator = partial(sample_generator, train_dataset, tokenizer)
        train_signature = {
            feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
            for feature in train_dataset.features
            if feature != "special_tokens_mask"
        }
        train_signature["labels"] = train_signature["input_ids"]
        train_signature = (train_signature, train_signature["labels"])
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        tf_train_dataset = (
            tf.data.Dataset.from_generator(train_generator, output_signature=train_signature)
            .with_options(options)
            .batch(
                batch_size=num_replicas * training_args.per_device_train_batch_size,
                drop_remainder=True,
            )
            .repeat(int(training_args.num_train_epochs))
        )
        eval_generator = partial(sample_generator, eval_dataset, tokenizer)
        eval_signature = {
            feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
            for feature in eval_dataset.features
            if feature != "special_tokens_mask"
        }
        eval_signature["labels"] = eval_signature["input_ids"]
        eval_signature = (eval_signature, eval_signature["labels"])
        tf_eval_dataset = (
            tf.data.Dataset.from_generator(eval_generator, output_signature=eval_signature)
            .with_options(options)
            .batch(
                batch_size=num_replicas * training_args.per_device_eval_batch_size,
                drop_remainder=True,
            )
        )
        # endregion

        # region Optimizer and loss
        batches_per_epoch = len(train_dataset) // (
            num_replicas * training_args.per_device_train_batch_size
        )
        # Bias and layernorm weights are automatically excluded from the decay
        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=int(training_args.num_train_epochs * batches_per_epoch),
            num_warmup_steps=training_args.warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
        )

        def dummy_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)

        model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
        # endregion

        # region Training and validation
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size = {training_args.per_device_train_batch_size * num_replicas}"
        )

        history = model.fit(
            tf_train_dataset,
            validation_data=tf_eval_dataset,
            epochs=int(training_args.num_train_epochs),
            steps_per_epoch=len(train_dataset)
            // (training_args.per_device_train_batch_size * num_replicas),
            callbacks=[SavePretrainedCallback(output_dir=training_args.output_dir)],
        )
        try:
            train_perplexity = math.exp(history.history["loss"][-1])
        except OverflowError:
            train_perplexity = math.inf
        try:
            validation_perplexity = math.exp(history.history["val_loss"][-1])
        except OverflowError:
            validation_perplexity = math.inf
        logger.warning(f"  Final train loss: {history.history['loss'][-1]:.3f}")
        logger.warning(f"  Final train perplexity: {train_perplexity:.3f}")
        logger.warning(f"  Final validation loss: {history.history['val_loss'][-1]:.3f}")
        logger.warning(f"  Final validation perplexity: {validation_perplexity:.3f}")
        # endregion

        if training_args.output_dir is not None:
            model.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        # You'll probably want to append some of your own metadata here!
        model.push_to_hub()


if __name__ == "__main__":
    main()
