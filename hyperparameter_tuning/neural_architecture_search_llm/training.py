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
This file is adapted from the run_glue.py text classification example from HuggingFace
https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py

The key difference is that we replace HuggingFace Trainer with our NASTrainer to allow for
NAS super-network training as described in:

Structural Pruning of Large Language Models via Neural Architecture Search
Aaron Klein, Jacek Golebiowski, Xingchen Ma, Valerio Perrone, Cedric Archambeau
AutoML Conference 2023 Workshop Track
"""
import shutil
import logging
import numpy as np
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    EvalPrediction,
    set_seed,
    HfArgumentParser,
)
from transformers.trainer_callback import TrainerCallback

from nas_trainer import NASTrainer, NASTrainingArguments
from hf_args import DataTrainingArguments, ModelArguments
from task_data import GLUE_TASK_INFO
from load_glue_datasets import load_glue_datasets
from load_imdb_dataset import load_imdb_dataset
from load_custom_dataset import load_custom_dataset

logger = logging.getLogger(__name__)


class ReportBackMetrics(TrainerCallback):
    """
    Simple callback to report metric such that we can parse them with cloud-watch.
    We also report the number of parameters for visualization purposes.
    """

    def __init__(self, number_of_parameters):
        self.number_of_parameters = number_of_parameters

    def on_evaluate(self, args, state, control, **kwargs):

        print(f"number_parameters: {self.number_of_parameters}")

        metrics = kwargs["metrics"]
        for metric, metric_value in metrics.items():
            print(f"{metric}: {metric_value}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NASTrainingArguments))

    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    model_type = model_args.model_name_or_path
    task_name = data_args.task_name
    set_seed(training_args.seed)

    if data_args.task_name in GLUE_TASK_INFO:
        load = load_glue_datasets
    elif data_args.task_name == "imdb":
        load = load_imdb_dataset
    elif data_args.task_name == "custom":
        load = load_custom_dataset

    train_dataset, valid_dataset, _, data_collator = load(
        training_args=training_args, model_args=model_args, data_args=data_args
    )

    is_regression = data_args.task_name == "stsb" or data_args.is_regression

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    # Labels
    if is_regression:
        num_labels = 1
    elif data_args.task_name in GLUE_TASK_INFO or data_args.task_name == "imdb":
        label_list = train_dataset.features["label"].names
        num_labels = len(label_list)
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = train_dataset.unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    if task_name in GLUE_TASK_INFO:
        metric = evaluate.load("glue", task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if data_args.is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    config = AutoConfig.from_pretrained(
        model_type,
        num_labels=num_labels,
        finetuning_task=task_name,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        config=config,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = NASTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[ReportBackMetrics(number_of_parameters=num_params)],
    )
    trainer.train()

    logging.info(f"Store checkpoint in: {training_args.output_dir}")
    trainer.model.save_pretrained("checkpoint")

    shutil.make_archive(
        base_name=training_args.output_dir + "/model", format="gztar", root_dir="checkpoint"
    )


if __name__ == "__main__":
    main()
