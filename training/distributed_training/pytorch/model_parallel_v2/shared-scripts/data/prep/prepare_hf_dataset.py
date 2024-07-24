import argparse
import functools
import logging
import os
from itertools import chain

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger

# Either set token here or in the env
# login(token="", add_to_git_credential=True, new_session=False)


logger = logging.getLogger(__name__)

"""
Example commands
----
1. Wikicorpus for llama
python prepare_hf_dataset.py --dataset_name wikicorpus \
    --dataset_config_name raw_en \
    --val_split_percentage 20 \
    --hf_tokenizer_name meta-llama/Llama-2-7b-hf \
    --seq_len 4096 \
    --output_dir /fsx/datasets/wikicorpus__raw_en/llama/4096/

2. C4
# Had to delete a file which was incomplete
# rm ~/.cache/huggingface/datasets/downloads/extracted/741a4aaf04e7748f791ce4525c5876f13a45e8115d76b099c818cf7970972c48
python prepare_hf_dataset.py --dataset_path /fsx/datasets/c4/en/hf \
    --output_dir /fsx/datasets/c4/en/hf-tokenized/llama \
    --hf_tokenizer_name meta-llama/Llama-2-7b-hf \
    --seq_len 4096 \
    --val_split_percentage 20
"""

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--val_split_percentage", type=int, default=20)
parser.add_argument("--hf_tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--output_dir", default=None, type=str)
parser.add_argument("--num_proc", default=64, type=int)
parser.add_argument("--seq_len", type=int, default=4096)
args, _ = parser.parse_known_args()

if args.dataset_path is not None and (args.dataset_name is not None and args.dataset_config_name):
    raise ValueError("Set either (dataset_path) or (dataset_name, dataset_config_name)")
elif args.dataset_path is None:
    if args.dataset_name is None or args.dataset_config_name is None:
        raise ValueError(
            "If dataset_path is not set, then both dataset_name and dataset_config_name need to be set"
        )
do_train = True
do_eval = True


def tokenize_function(tokenizer, text_column_name, examples):
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    with CaptureLogger(tok_logger) as cl:
        output = _tokenize_function(tokenizer, text_column_name, examples)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
    return output


def _tokenize_function(tokenizer, text_column_name, examples):
    return tokenizer(examples[text_column_name])


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(block_size, examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
    else:
        result = {}
    return result


def tokenize_dataset(
    dataset_name,
    dataset_config_name,
    dataset_path,
    hf_tokenizer_name,
    output_dir,
    val_split_percentage=20,
    sequence_length=4096,
    num_proc=64,
    overwrite_cache=False,
):
    cache_dir = "/fsx/datasets/.cache/datasets/"
    if dataset_path is not None:
        raw_datasets = load_dataset(dataset_path, num_proc=num_proc, cache_dir=cache_dir)
    else:
        raw_datasets = load_dataset(
            dataset_name, dataset_config_name, num_proc=num_proc, cache_dir=cache_dir
        )

    os.makedirs(output_dir, exist_ok=True)
    train_split_percentage = 100 - val_split_percentage
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{val_split_percentage}%]",
            cache_dir=cache_dir,
        )

        raw_datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{train_split_percentage}%]",
            cache_dir=cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name, trust_remote_code=True)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    tokenized_datasets = raw_datasets.map(
        functools.partial(tokenize_function, tokenizer, text_column_name),
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    assert tokenizer.model_max_length >= sequence_length

    lm_datasets = tokenized_datasets.map(
        functools.partial(group_texts, sequence_length),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of {sequence_length}",
    )
    if do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        train_dataset.save_to_disk(f"{output_dir}/train/", num_proc=num_proc)

    if do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        eval_dataset.save_to_disk(f"{output_dir}/val/", num_proc=num_proc)

    torch.save({"arguments": args}, f"{output_dir}/args")


if __name__ == "__main__":
    tokenize_dataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_path=args.dataset_path,
        hf_tokenizer_name=args.hf_tokenizer_name,
        output_dir=args.output_dir,
        val_split_percentage=args.val_split_percentage,
        sequence_length=args.seq_len,
        num_proc=args.num_proc,
    )
