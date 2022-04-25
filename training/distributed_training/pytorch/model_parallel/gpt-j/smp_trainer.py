import datasets
from datasets import load_dataset, load_from_disk, load_metric
import smdistributed.modelparallel.torch as smp

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import time
import os

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.testing_utils import CaptureLogger
from transformers.file_utils import is_sagemaker_dp_enabled, is_sagemaker_mp_enabled


from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.trainer_utils import (
    EvalPrediction,
    get_last_checkpoint,
)

from transformers.trainer_callback import (
    TrainerCallback,
)

import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)


class SMPTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
    ):

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    @smp.step
    def train_step(model, optimizer, input_ids, attention_mask, args):

        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]

        model.backward(loss)

        return loss

    def train_smp(
        self,
        model,
        optimizer,
        lr_scheduler,
        start_train_path_index,
        start_batch_index,
        num_params,
        total_steps,
        args,
        prescaled_batch,
    ):

        model.train()

        dp_rank = smp.dp_rank() if not prescaled_batch else smp.rdp_rank()
        dp_size = smp.dp_size() if not prescaled_batch else smp.rdp_size()

        start = time.time()
        throughput = None
        to_save = {"loss": [], "val_loss": []}
        loss_metric = 0

        def should_record():

            # only record the ranks that in the tp group that contains global rank 0
            if smp.tp_size() > 1:
                tp_group = smp.get_tp_group()
                return 0 in tp_group
            else:
                return smp.rank() == 0

        # Set the same seed for computation
        set_seed(args.seed)

        sampler = torch.utils.data.DistributedSampler(
            self.train_dataset,
            shuffle=True,
            seed=args.seed,
            rank=dp_rank,
            num_replicas=dp_size,
            drop_last=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        total_steps = 0
        for batch_idx, input_data in enumerate(train_dataloader):

            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)

            input_ids = input_data["input_ids"]
            attention_mask = input_data["attention_mask"]

            loss_mb = self.train_step(model, optimizer, input_ids, attention_mask, args)

            loss = loss_mb.reduce_mean()

            lr_scheduler.step()

            total_steps += 1

            total_steps += 1
            time_elapsed = time.time() - start
            step_time = time.time() - step_start

            if smp.rank() == 0 and not total_steps % 10:
                print(
                    f"({int(time_elapsed)}s), Batch {total_steps - 1} Loss: {loss.item()}, Speed: {''} samples/sec"
                )
            if total_steps == args.max_steps:
                break
