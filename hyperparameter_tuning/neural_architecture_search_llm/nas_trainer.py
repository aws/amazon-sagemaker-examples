from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.data.data_collator import DataCollator
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction

from sampling import SmallSearchSpace
from mask import mask_bert


@dataclass
class NASTrainingArguments(TrainingArguments):
    num_sub_networks: Optional[int] = field(
        default=4,
        metadata={
            "help": "Total number sub-networks that are updated in each steps. Includes the lower and upper bound "
            "of the sandwich rule and n-2 random sub-networks."
        },
    )


class NASTrainer(Trainer):

    """
    Overwrites HuggingFace Trainer to run NAS super-network training, which updates in each steps part of the network
    according to the sandwich rule. For more information, see:

    Structural Pruning of Large Language Models via Neural Architecture Search
    Aaron Klein Jacek Golebiowski Xingchen Ma Valerio Perrone Cedric Archambeau
    AutoML Conference 2023 Workshop Track
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):

        self.sampler = SmallSearchSpace(model.config, seed=args.seed)

        model_type = model.config._name_or_path
        if model_type.startswith("bert"):
            self.mask = mask_bert
        else:
            raise AttributeError(f"Model {model_type} is not supported at this point!")

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False, head_mask=None, ffn_mask=None):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if head_mask is not None and ffn_mask is not None:
            handles = self.mask(model, ffn_mask, head_mask)
            outputs = model(head_mask=head_mask, **inputs)

            for handle in handles:
                handle.remove()
        else:
            outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:

            model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step according to the sandwich rule on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        sum_of_sub_networks_losses = 0
        device = self.accelerator.device
        for i in range(self.args.num_sub_networks):
            #
            # if is_sagemaker_mp_enabled():
            #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            #     return loss_mb.reduce_mean().detach().to(self.args.device)
            #
            # with self.compute_loss_context_manager():
            #     loss = self.compute_loss(model, inputs)

            if i == 0:
                # first update full network
                head_mask, ffn_mask = None, None
            elif i == 1:
                # update smallest network
                head_mask, ffn_mask = self.sampler.get_smallest_sub_network()
                head_mask = head_mask.to(device=device, dtype=model.dtype)
                ffn_mask = ffn_mask.to(device=device, dtype=model.dtype)
            else:
                # update random sub-network
                head_mask, ffn_mask = self.sampler()
                head_mask = head_mask.to(device=device, dtype=model.dtype)
                ffn_mask = ffn_mask.to(device=device, dtype=model.dtype)

            loss = self.compute_loss(model, inputs, head_mask=head_mask, ffn_mask=ffn_mask)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            sum_of_sub_networks_losses += loss.detach() / self.args.gradient_accumulation_steps
        return sum_of_sub_networks_losses
