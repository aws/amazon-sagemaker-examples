import argparse
import json
import logging
import math
from pathlib import Path
import time

import evaluate
import torch
import os
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler


### implementation - optional
from smart_sifting.data_model.data_model_interface import SiftingBatch, SiftingBatchTransform
from smart_sifting.data_model.list_batch import ListBatch
from smart_sifting.sift_config.sift_configs import RelativeProbabilisticSiftConfig, LossConfig, SiftingBaseConfig


#### interface
from smart_sifting.dataloader.sift_dataloader import SiftingDataloader
from smart_sifting.loss.abstract_sift_loss_module import Loss


from typing import Any

logging.basicConfig(level=logging.INFO)

############################################# ImageLoss ####################################################
class ImageLoss(Loss):
    """
    This is an implementation of the Loss interface for the model 
    required for Smart Sifting. 
    """
    def __init__(self):
        self.celoss = torch.nn.CrossEntropyLoss(reduction='none')

    def loss(
            self,
            model: torch.nn.Module,
            transformed_batch: SiftingBatch,
            original_batch: Any = None,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v, in original_batch.items()}

        # compute loss
        outputs = model(**batch)
        return self.celoss(outputs.logits, batch["labels"])

############################################# ImageListBatchTransform ################################################
class ImageListBatchTransform(SiftingBatchTransform):
    """
    This is an implementation of the data transforms for the model 
    required for Smart Sifting. Transform to and from ListBatch
    """
    def transform(self, batch: Any):
        inputs = []
        labels = []

        for i in range(len(batch["pixel_values"])):
            inputs.append(batch["pixel_values"][i])

        for i in range(len(batch["labels"])):
            labels.append(batch["labels"][i])

        return ListBatch(inputs, labels)
    
    def reverse_transform(self, list_batch: ListBatch):
        a_batch = {}
        a_batch["pixel_values"] = self.stack_tensors(list_batch.inputs)
        a_batch["labels"] = self.stack_tensors(list_batch.labels)
  
        return a_batch

    def stack_tensors(self,list_of_tensors):
        if list_of_tensors:
            t = torch.stack(list_of_tensors)
        else:
            t = torch.tensor([])
        return t
    

def parse_args():

    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")

    
    ##################### SIFTING FLAG #############################
    parser.add_argument(
        "--use_sifting",
        type=int,
        default=None,
        help="Flag for whether to use sifting or not. 0 serves as false",
    )
    #################################################################
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset)."
        ),
    )
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_false",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    args = parser.parse_args()
    # Sanity checks
    if args.dataset_name is None and args.train_dir is None and args.validation_dir is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")


    return args


def main():
    args = parse_args()

    ##################### SIFTING FLAG #############################
    sifting = (args.use_sifting > 0) 
    ################################################################



    # If passed along, set the training seed now.
    if args.seed is not None:
        torch.manual_seed(args.seed)


    dataset = load_from_disk(args.train_dir)
    # If we don't have a validation split, split off a percentage of train as validation.
    args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["label"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and image processor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        i2label=id2label,
        label2id=label2id,
        finetuning_task="image-classification",
    )
    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    # Preprocessing the datasets

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    #with accelerator.main_process_first():
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)
    if args.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(args.max_eval_samples))
    # Set the validation transforms
    eval_dataset = dataset["validation"].with_transform(preprocess_val)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_training_steps = args.num_train_epochs * len(train_dataloader)

    if (sifting):    
        print("******* will run the training using sifting*********")
        sift_config = RelativeProbabilisticSiftConfig(
            beta_value=3,
            loss_history_length=500,
            loss_based_sift_config=LossConfig(
                 sift_config=SiftingBaseConfig(sift_delay=10)
            )
        )
        train_dataloader = SiftingDataloader(
                sift_config=sift_config,
                orig_dataloader=train_dataloader,
                batch_transforms=ImageListBatchTransform(),
                loss_impl=ImageLoss(),
                model=model
        )        
       
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=num_training_steps * args.gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    #num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    #args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Get the metric function
    metric = evaluate.load("accuracy")
    clf_metrics = evaluate.combine([
        evaluate.load("accuracy",average="weighted"),
        evaluate.load("f1",average="weighted"),
        evaluate.load("precision", average="weighted"),
        evaluate.load("recall", average="weighted")
        ])
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(num_training_steps))
    completed_steps = 0
    starting_epoch = 0

    device = torch.device("cuda")

    model = model.to(device)
    train_step_count = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        total_loss = 0
        for  batch in train_dataloader:
            train_start = time.perf_counter()

            batch = {k: v.to(device) for k, v, in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
                # We keep track of the loss at each epoch
 
            total_loss += loss.detach().float()
            train_bp_start = time.perf_counter()
            print((f'train forward pass latency: {train_bp_start - train_start}'))
            loss.backward()
            print(f'train backprop latency: {time.perf_counter() - train_bp_start}')
            train_optim_start = time.perf_counter()
            optimizer.step() #gather gradient updates from all cores and apply them
            lr_scheduler.step()
            optimizer.zero_grad()
            print(f'train optimizer step latency: {time.perf_counter() - train_optim_start}')
            print(f'train total step latency: {time.perf_counter() - train_start}')
            train_step_count += 1
            print(f'train step count: {train_step_count}')

            progress_bar.update(1)
            completed_steps += 1


            if completed_steps >= args.max_train_steps:
                break
        print(
            "Epoch {}, Loss {:0.4f}".format(epoch, loss.detach().to("cpu"))
            )       
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v, in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)
            references = batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"epoch {epoch}: {eval_metric}")
        print(f"epoch {epoch}: eval loss {loss}")


    if args.output_dir is not None:     
        image_processor.save_pretrained(args.output_dir)
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    main()
