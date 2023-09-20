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
"""
Script evaluate a single sub-network given the shared weights of a fine-tuned
super-network. This script is meant to be executed by Automated Model tuner.
For more information about the neural architecture search methods see:

Structural Pruning of Large Language Models via Neural Architecture Search
Aaron Klein Jacek Golebiowski Xingchen Ma Valerio Perrone Cedric Archambeau
AutoML Conference 2023 Workshop Track
"""
import os
import logging
import sys
import tarfile

from dataclasses import dataclass, field

import numpy as np
import torch
import datasets

from transformers import (
    AutoModelForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from torch.utils.data import DataLoader

from evaluate import load

from estimate_efficency import compute_parameters
from task_data import GLUE_TASK_INFO
from mask import mask_bert
from hf_args import DataTrainingArguments, ModelArguments
from load_glue_datasets import load_glue_datasets
from load_imdb_dataset import load_imdb_dataset
from load_custom_dataset import load_custom_dataset

logger = logging.getLogger(__name__)


@dataclass
class SearchArguments:
    """
    Arguments to evaluate a sub-network given a fine-tuned super-network
    """

    num_layers: int = field(metadata={"help": "Number of layers of the sub-network"})
    num_heads: int = field(metadata={"help": "Number of heads per layer of the sub-network"})
    num_units: int = field(
        metadata={
            "help": "Number of units for each intermediate fully-connected layer of "
            "the sub-network"
        }
    )
    checkpoint_dir_model: str = field(
        metadata={"help": "Location of the checkpoint of the fine-tuned super-networks"},
        default=os.environ["SM_CHANNEL_MODEL"],
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, SearchArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        search_args,
    ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    # Set seed before initializing model
    if int(training_args.seed) == -1:
        training_args.seed = np.random.randint(2**32 - 1)
    set_seed(training_args.seed)

    # Extract tar ball
    tar = tarfile.open(search_args.checkpoint_dir_model + "/model.tar.gz")
    tar.extractall(training_args.output_dir)
    tar.close()

    model_type = model_args.model_name_or_path

    is_regression = data_args.task_name == "stsb"

    # Evaluation metric
    metric = load("glue", data_args.task_name)

    # Load validation data
    if data_args.task_name in GLUE_TASK_INFO:
        load_dataset = load_glue_datasets
    elif data_args.task_name == "imdb":
        load_dataset = load_imdb_dataset
    elif data_args.task_name == "custom":
        load_dataset = load_custom_dataset

    _, valid_dataset, _, data_collator = load_dataset(
        training_args=training_args, model_args=model_args, data_args=data_args
    )

    eval_dataloader = DataLoader(
        valid_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(training_args.output_dir)

    config = model.config
    if model_type.startswith("bert"):
        attention_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        mask = mask_bert

        n_params_emb = sum(p.numel() for p in model.bert.embeddings.parameters() if p.requires_grad)
        n_params_pooler = sum(p.numel() for p in model.bert.pooler.parameters() if p.requires_grad)
        n_params_classifier = sum(
            p.numel() for p in model.classifier.parameters() if p.requires_grad
        )
        n_params_classifier += n_params_pooler

    else:
        raise AttributeError(f"Model {model_type} is not supported at this point!")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    metric_name = GLUE_TASK_INFO[data_args.task_name]["metric"]

    head_mask = torch.ones((num_layers, num_attention_heads))
    neuron_mask = torch.ones((num_layers, intermediate_size))

    head_mask[search_args.num_layers :, :] = 0
    head_mask[: search_args.num_layers :, search_args.num_heads :] = 0
    neuron_mask[search_args.num_layers :, :] = 0
    neuron_mask[: search_args.num_layers :, search_args.num_units :] = 0

    head_mask = head_mask.to(device)
    neuron_mask = neuron_mask.to(device)
    n_params_model = compute_parameters(
        dmodel=attention_size,
        dhead=attention_head_size,
        num_heads_per_layer=head_mask.sum(dim=1),
        num_neurons_per_layer=neuron_mask.sum(dim=1),
    )
    n_params = n_params_emb + n_params_model + n_params_classifier

    _ = mask(model, neuron_mask, head_mask)

    model.eval()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(head_mask=head_mask, **batch)

        logits = outputs.logits
        predictions = torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)

        metric.add_batch(predictions=predictions, references=batch["labels"])

    eval_metric = metric.compute()

    # Return metrics to cloud watch
    print(f"number of parameters: {n_params}")
    print(f"validation error: {1 - eval_metric[metric_name]}")


if __name__ == "__main__":
    main()
