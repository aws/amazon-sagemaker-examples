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
import time
import torch
import logging
import numpy as np

from tqdm.auto import tqdm

from transformers import get_scheduler
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F

import accelerate

from sampling import SmallSearchSpace
from mask import mask_bert, mask_gpt

accelerator = accelerate.Accelerator()

logger = logging.getLogger(__name__)


def train_supernetwork(model, train_dataloader, eval_dataloader, metric, training_args):

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

    start_time = time.time()
    step = 0

    if training_args.is_regression:
        distillation_loss = nn.MSELoss()
    else:
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        distillation_loss = lambda x, y: kl_loss(
            F.log_softmax(x, dim=-1), F.log_softmax(y, dim=-1)
        )

    model_type = model.config._name_or_path
    if model_type.startswith("gpt2"):
        mask = mask_gpt
    elif model_type.startswith("bert"):
        mask = mask_bert

    if training_args.use_accelerate:
        (
            train_dataloader,
            eval_dataloader,
            model,
            optimizer,
        ) = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

    sampler = SmallSearchSpace(
        model.config, rng=np.random.RandomState(seed=training_args.seed)
    )

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            if not training_args.use_accelerate:
                batch = {k: v.to(device) for k, v in batch.items()}

            # update largest sub-network (i.e super-network)
            outputs = model(**batch)
            loss = outputs.loss
            y_teacher = outputs.logits.detach()
            accelerator.backward(
                loss
            ) if training_args.use_accelerate else loss.backward()

            # update smallest sub-network
            head_mask, ffn_mask = sampler.get_smallest_sub_network()
            head_mask = head_mask.to(device="cuda", dtype=model.dtype)
            ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)
            handles = mask(model, ffn_mask, head_mask)
            outputs = model(head_mask=head_mask, **batch)

            for handle in handles:
                handle.remove()

            loss = distillation_loss(outputs.logits, y_teacher)
            accelerator.backward(
                loss
            ) if training_args.use_accelerate else loss.backward()

            # update random sub-network
            head_mask, ffn_mask = sampler()
            head_mask = head_mask.to(device="cuda", dtype=model.dtype)
            ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

            handles = mask(model, ffn_mask, head_mask)
            outputs = model(head_mask=head_mask, **batch)
            for handle in handles:
                handle.remove()

            loss = distillation_loss(outputs.logits, y_teacher)
            accelerator.backward(
                loss
            ) if training_args.use_accelerate else loss.backward()

            # update random sub-network
            head_mask, ffn_mask = sampler()
            head_mask = head_mask.to(device="cuda", dtype=model.dtype)
            ffn_mask = ffn_mask.to(device="cuda", dtype=model.dtype)

            handles = mask(model, ffn_mask, head_mask)
            outputs = model(head_mask=head_mask, **batch)

            for handle in handles:
                handle.remove()

            loss = distillation_loss(outputs.logits, y_teacher)
            accelerator.backward(
                loss
            ) if training_args.use_accelerate else loss.backward()

            step += 1

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            train_loss += loss

        model.eval()
        for batch in eval_dataloader:
            if not training_args.use_accelerate:
                batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
            predictions = (
                torch.squeeze(logits)
                if training_args.is_regression
                else torch.argmax(logits, dim=-1)
            )

            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        runtime = time.time() - start_time
        logger.info(
            f"epoch {epoch}: training loss = {train_loss / len(train_dataloader)}, "
            f"evaluation metrics = {eval_metric}, "
            f"runtime = {runtime}"
        )

        if training_args.save_strategy == "epoch":
            os.makedirs(training_args.output_dir, exist_ok=True)
            logger.info(f"Store checkpoint in: {training_args.output_dir}")
            if training_args.use_accelerate:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    training_args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
            else:
                model.save_pretrained(training_args.output_dir)
