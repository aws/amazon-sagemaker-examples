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
import os
import shutil
import logging
import numpy as np
import torch
import torch.nn.functional as F
import evaluate

from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn

from transformers import (
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_scheduler,
    HfArgumentParser
)

from datasets import load_dataset

from sampling import SmallSearchSpace
from mask import mask_bert 
from hf_args import DataTrainingArguments, ModelArguments
from task_data import TASKINFO

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    model_type = model_args.model_name_or_path
    task_name = data_args.task_name
    seed = training_args.seed
    per_device_train_batch_size = training_args.per_device_train_batch_size
    per_device_eval_batch_size = training_args.per_device_eval_batch_size

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    padding = "max_length"

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        return result

    raw_datasets = load_dataset("glue", task_name)
    sentence1_key, sentence2_key = ("sentence1", "sentence2")

    metric = evaluate.load("glue", task_name)

    preproc_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    label_list = preproc_datasets["train"].features["label"].names
    num_labels = len(label_list)

    train_dataset = preproc_datasets["train"]
    train_dataset = train_dataset.remove_columns(["idx"])

    split = train_dataset.train_test_split(train_size=0.7, seed=seed)
    train_dataset = split["train"]
    valid_dataset = split["test"]

    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=per_device_train_batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        valid_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    config = AutoConfig.from_pretrained(
        model_type,
        num_labels=num_labels,
        finetuning_task=task_name,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        config=config,
    )

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    num_training_steps = int(training_args.num_train_epochs * len(train_dataloader))
    warmup_steps = int(training_args.warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    step = 0

    if data_args.is_regression:
        distillation_loss = nn.MSELoss()
    else:
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        distillation_loss = lambda x, y: kl_loss(
            F.log_softmax(x, dim=-1), F.log_softmax(y, dim=-1)
        )

    model_type = model.config._name_or_path
    if model_type.startswith("bert"):
        mask = mask_bert
    else:
        raise AttributeError(f'Model {model_type} is not supported at this point!')

    sampler = SmallSearchSpace(
        model.config, rng=np.random.RandomState(seed=training_args.seed)
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # update largest sub-network (i.e super-network)
            outputs = model(**batch)
            loss = outputs.loss
            y_teacher = outputs.logits.detach()
            loss.backward()

            # update smallest sub-network
            head_mask, ffn_mask = sampler.get_smallest_sub_network()
            head_mask = head_mask.to(device=device, dtype=model.dtype)
            ffn_mask = ffn_mask.to(device=device, dtype=model.dtype)
            handles = mask(model, ffn_mask, head_mask)
            outputs = model(head_mask=head_mask, **batch)

            for handle in handles:
                handle.remove()

            loss = distillation_loss(outputs.logits, y_teacher)
            loss.backward()

            # update random sub-network
            head_mask, ffn_mask = sampler()
            head_mask = head_mask.to(device=device, dtype=model.dtype)
            ffn_mask = ffn_mask.to(device=device, dtype=model.dtype)

            handles = mask(model, ffn_mask, head_mask)
            outputs = model(head_mask=head_mask, **batch)
            for handle in handles:
                handle.remove()

            loss = distillation_loss(outputs.logits, y_teacher)
            loss.backward()

            # update random sub-network
            head_mask, ffn_mask = sampler()
            head_mask = head_mask.to(device=device, dtype=model.dtype)
            ffn_mask = ffn_mask.to(device=device, dtype=model.dtype)

            handles = mask(model, ffn_mask, head_mask)
            outputs = model(head_mask=head_mask, **batch)

            for handle in handles:
                handle.remove()

            loss = distillation_loss(outputs.logits, y_teacher)
            loss.backward()

            step += 1

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            train_loss += loss

        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            logits = outputs.logits
            predictions = (
                torch.squeeze(logits)
                if data_args.is_regression
                else torch.argmax(logits, dim=-1)
            )

            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        metric_name = TASKINFO[data_args.task_name]["metric"]

        print(f"epoch: {epoch}")
        print(f"training loss: {train_loss / len(train_dataloader)}")
        print(f"number of parameters: {n_params}")
        print(f"validation error: {1 - eval_metric[metric_name]}")

        if training_args.save_strategy == "epoch":
            os.makedirs(training_args.output_dir, exist_ok=True)

            logging.info(f"Store checkpoint in: {training_args.output_dir}")
            model.save_pretrained('checkpoint')

            shutil.make_archive(
                base_name=training_args.output_dir + '/model',
                format='gztar',
                root_dir='checkpoint'
            )


if __name__ == "__main__":
    main()
