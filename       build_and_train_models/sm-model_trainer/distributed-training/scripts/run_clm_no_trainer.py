# flake8: noqa
import os
import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_scheduler, SchedulerType
from datasets import load_from_disk
import torch
from utils import create_dataloaders, get_module_class_from_name, save_model
import time
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
import functools


def training_function(args):
    # set seed
    set_seed(args['seed'])

    from huggingface_hub.hf_api import HfFolder

    print(f"Loading dataset from {args['dataset_path']}")
    dataset = load_from_disk(f"file://{args['dataset_path']}")

    dist.barrier()

    # load model from the hub
    model = AutoModelForCausalLM.from_pretrained(
        args['model_id'],
        cache_dir=args['cache_dir'],
        use_cache=(
            False if args['gradient_checkpointing'] else True
        ),  # this is needed for gradient checkpointing
    )

    tokenizer = AutoTokenizer.from_pretrained(args['model_id'])

    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    # Create dataloaders for training and evaluation
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset,
        eval_dataset,
        args['rank'],
        args['world_size'],
        args['seed'],
        args['per_device_train_batch_size'],
        args['per_device_train_batch_size'],
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            get_module_class_from_name(model, args['fsdp_transformer_layer_cls_to_wrap'])
        },
    )

    torch.cuda.set_device(args['local_rank'])

    dtype = torch.bfloat16

    mixed_precision_policy = MixedPrecision(
        param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3,
        cpu_offload=CPUOffload(offload_params=True),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # BACKWARD_POST, BACKWARD_PRE
        forward_prefetch=args['forward_prefetch'],
        limit_all_gathers=args['limit_all_gathers'],
        device_id=torch.cuda.current_device(),
    )

    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper, offload_to_cpu=True, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    check_fn_gpt = lambda submodule: isinstance(
        submodule, get_module_class_from_name(model, args['fsdp_transformer_layer_cls_to_wrap'])
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn_gpt
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'])

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args['gradient_accumulation_steps']
    )
    if args['rank'] == 0:
        print(f"Number of update steps per epoch {num_update_steps_per_epoch}")
    if args['max_train_steps'] is None:
        args['max_train_steps'] = args['num_train_epochs'] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=args['num_warmup_steps'] * args['gradient_accumulation_steps'],
        num_training_steps=args['max_train_steps'] * args['gradient_accumulation_steps'],
    )

    start = time.time()
    device = torch.device(f"cuda:{args['local_rank']}")

    # Perform Training Loop for num_train_epochs times
    for epoch in range(args['num_train_epochs']):

        model.train()
        total_steps = 0
        fsdp_loss = torch.zeros(2).to(args['local_rank'])

        # Use train_dataloader to get the batch data
        for _, batch in enumerate(tqdm(train_dataloader, disable=not (args['rank'] == 0))):

            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output["loss"]
            loss.backward()
            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(batch["input_ids"])

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_steps += 1
            if total_steps > args['max_steps']:
                break

        # Reduce the loss across all processes
        torch.distributed.all_reduce(fsdp_loss, op=torch.distributed.ReduceOp.SUM)
        train_loss = fsdp_loss[0] / fsdp_loss[1]
        train_ppl = torch.exp(train_loss)

        if args['rank'] == 0:
            print(f"******{epoch=}: {train_ppl=} {train_loss=}******")

        model.eval()
        eval_loss = 0
        fsdp_eval_loss = torch.zeros(2).to(args['local_rank'])
        for steps, batch in enumerate(tqdm(eval_dataloader, disable=not (args['rank'] == 0))):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs["loss"]

            fsdp_eval_loss[0] += loss.item()
            fsdp_eval_loss[1] += len(batch["input_ids"])
            if steps > args['max_steps']:
                break

        torch.distributed.all_reduce(fsdp_eval_loss, op=torch.distributed.ReduceOp.SUM)
        eval_loss = fsdp_eval_loss[0] / fsdp_eval_loss[1]
        eval_ppl = torch.exp(eval_loss)

        if args['rank'] == 0:
            print(f"*******{epoch=}: {eval_ppl=} {eval_loss=}*******")

    save_model(model, tokenizer, args['model_dir'], args['rank'])
    if args['rank'] == 0:
        print("Training done!")
    dist.barrier()


def main():
    args = json.loads(os.environ["SM_HPS"])

    if "LOCAL_RANK" in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        WORLD_RANK = int(os.environ["RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        # Environment variables set by mpirun
        LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
    else:
        import sys

        sys.exit("Can't find the evironment variables for local rank")

    args["local_rank"] = LOCAL_RANK
    args["rank"] = WORLD_RANK
    args["world_size"] = WORLD_SIZE

    print(f"Running with args: {args}")
    # Initialize the distributed environment
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=args['rank'], world_size=args['world_size']
    )
    training_function(args)


if __name__ == "__main__":
    main()
