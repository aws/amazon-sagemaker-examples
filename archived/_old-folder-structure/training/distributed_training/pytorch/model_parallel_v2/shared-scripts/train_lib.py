"""Train lib function."""
import datetime
import functools
import math
import re
import time
from contextlib import nullcontext

# pylint: disable=fixme,import-error,import-outside-toplevel,invalid-name,no-name-in-module,wrong-import-order
import numpy as np
import torch
import torch.distributed as dist
import torch.sagemaker as tsm
import torch.utils.data

import transformer_engine
from transformer_engine.common.recipe import Format, DelayedScaling

import transformers
from accelerate import init_empty_weights
from checkpoints import (
    _CHECKPOINT_DIR_REGEX,
    _DEFAULT_STATE_DICT_TYPE,
    CheckpointingMethod,
    get_coordinator_rank,
    is_action_rank,
    load_checkpoint,
    save_checkpoint,
)
from data.pipelines import GPTDataPipeline, create_data_pipeline
from fsdp_utils import get_backward_fetch_policy, get_sharding_strategy, get_transformer_layer
from logging_utils import (
    create_args_table,
    get_logger,
    log_and_write_eval_metrics,
    log_train_metrics,
    show_env_vars,
    write_nccl_test_stats,
    write_metrics_train_step,
)
from memory_tracker import memory_status, memory_status_cpu
from packaging import version as pversion
from torch import optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.sagemaker import transform
from torch.sagemaker.delayed_param import DelayedParamIniter
from torch.sagemaker.grad_norm import clip_grad_norm_
from torch.sagemaker.utils import utils as tsm_utils  # pylint: disable=no-name-in-module
from train_utils import (
    apply_activation_checkpoint,
    compute_num_params,
    compute_tflops,
    create_model,
    get_learning_rate_scheduler,
    get_model_config,
    get_param_groups_by_weight_decay,
    patch_neox_rope,
)
from transformers import set_seed
import utils

logger = get_logger()


def finetune_with_pretrained_weights_check(args) -> bool:
    # returns True for start of finetuning only
    return args.hf_pretrained_model_name_or_dir is not None and args.resume_from_checkpoint is None


def finetune_check(args):
    # returns True for start of finetuning as well as resuming
    return args.hf_pretrained_model_name_or_dir is not None


def eval_model(model, data_pipeline, num_batches):
    """Eval step."""
    model = model.eval()
    n_batches = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, input_data in enumerate(data_pipeline.val_dataloader):
            input_ids, _ = data_pipeline.get_val_batch(input_data)

            if batch_idx >= num_batches:
                break

            loss += model(input_ids=input_ids, attention_mask=None, labels=input_ids)["loss"]
            n_batches += 1

    if n_batches > 0:
        detached_loss = loss.detach()
        torch.distributed.all_reduce(detached_loss)
        loss = detached_loss.item() / dist.get_world_size()
        loss /= n_batches
        ppl = math.exp(loss)
    else:
        loss = -1.0
        ppl = -1.0

    return loss, ppl


def reduce_loss(loss):
    loss_detached = loss.detach()
    dist.all_reduce(loss_detached)
    loss_scalar = loss_detached.item() / dist.get_world_size()
    return loss_scalar


def train_step(  # pylint: disable=too-many-arguments,too-many-branches,too-many-locals
    args, display_step: int, batch_idx: int, nvtx_warmup_iters,
    data_pipeline, input_data, model, optimizer, lr_scheduler, writers, fp8_recipe
):
    if batch_idx >= nvtx_warmup_iters:
        torch.cuda.nvtx.range_push(f"iteration{batch_idx}")

    input_ids, _, labels = data_pipeline.get_batch(input_data)

    if batch_idx == 0:
        # checking only on batch 0 to reduce checks during runtime
        assert (
            input_ids.shape[1] == args.max_context_width
        ), f"Input data passed {input_ids.shape} does not respect max_context_width set. Note that this is not strictly necessary, but added to prevent mistakes. If you intend to do this, please remove this check."
        assert (
            input_ids.shape[1] <= args.max_context_width
        ), "Input data passed is larger than max_context_width for model. You need to change max_context_width so model can expect larger sequences"

    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    step_start = time.time()

    if batch_idx >= nvtx_warmup_iters:
        torch.cuda.nvtx.range_push("forward")

    # uses default causal mask
    if args.fp8==1 and args.use_smp_implementation==1:
        with transformer_engine.pytorch.fp8_autocast(enabled=args.fp8, fp8_recipe=fp8_recipe, fp8_group=tsm.state.world_process_group):
            loss = model(input_ids=input_ids, attention_mask=None, labels=labels)["loss"]
    else:
        loss = model(input_ids=input_ids, attention_mask=None, labels=labels)["loss"]

    if batch_idx >= nvtx_warmup_iters:
        # for forward
        torch.cuda.nvtx.range_pop()

    if args.enable_memory_profiling > 0 and batch_idx < 5:
        memory_status_cpu("After forward", writers=writers, step=display_step)
        memory_status(tag="After forward", writers=writers, step=display_step)

    if batch_idx >= nvtx_warmup_iters:
        torch.cuda.nvtx.range_push("backward")

    loss.backward()

    if batch_idx >= nvtx_warmup_iters:
        # for backward
        torch.cuda.nvtx.range_pop()

    if args.enable_memory_profiling > 0 and batch_idx < 5:
        memory_status_cpu("After train step", writers=writers, step=display_step)
        memory_status(tag="After train step", writers=writers, step=display_step)

    if batch_idx >= nvtx_warmup_iters:
        torch.cuda.nvtx.range_push("opt_step")

    grad_norm = clip_grad_norm_(model, args.grad_clip)
    optimizer.step()
    lr_scheduler.step()

    if batch_idx >= nvtx_warmup_iters:
        # for opt step
        torch.cuda.nvtx.range_pop()

    if args.clean_cache > 0:
        # empty the cache to avoid OOM
        torch.cuda.empty_cache()

    if batch_idx >= nvtx_warmup_iters:
        # for step
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    step_time = time.time() - step_start

    if args.enable_memory_profiling > 0 and batch_idx < 5:
        memory_status(tag="After opt step", writers=writers, step=display_step)

    batch_num_sequences = input_ids.shape[0]
    batch_seqlen = input_ids.shape[1]
    return loss, step_time, batch_num_sequences, batch_seqlen, grad_norm


# pylint: disable=no-member,too-many-arguments,too-many-branches,too-many-locals,too-many-statements
def train(
    model,
    optimizer,
    lr_scheduler,
    writers,
    model_config,
    start_epoch,
    start_train_path_index,
    resume_from_sequence_number,
    val_resume_from_sequence_number,
    num_params,
    total_steps,
    args,
    global_rank,
    world_size,
    checkpointing_pg_metadata,
    fp8_recipe,
):
    """Train."""
    if args.enable_memory_profiling > 0:
        memory_status_cpu(tag="Before train step", writers=writers, step=total_steps - 1)

    model.train()
    dp_rank = global_rank
    dp_size = world_size

    if tsm.state.tensor_parallel_degree > 1:
        dp_rank //= tsm.state.tensor_parallel_degree
        dp_size //= tsm.state.tensor_parallel_degree

    if global_rank == 0:
        logger.info("Creating train dataloader")

    throughputs = []
    # Set the same seed for computation
    set_seed(args.seed)

    data_pipeline = create_data_pipeline(
        args, start_train_path_index, resume_from_sequence_number, val_resume_from_sequence_number, dp_rank, dp_size
    )
    cur_seq_index = resume_from_sequence_number
    cur_val_seq_index = val_resume_from_sequence_number
    epoch = start_epoch
    while total_steps < args.max_steps:
        nvtx_warmup_iters = 3
        if global_rank == 0:
            logger.info("Starting training with epoch %s.", epoch)

        # additional loop around is for GPTDataset as there can be multiple dataloaders
        if isinstance(data_pipeline, GPTDataPipeline):
            # with new path if incremented at the end of this for loop
            data_pipeline.create_train_dataset()

        for batch_idx, input_data in enumerate(data_pipeline.train_dataloader):
            if total_steps >= args.max_steps:
                break

            if args.profile_nsys > 0 and batch_idx == nvtx_warmup_iters:
                torch.cuda.cudart().cudaProfilerStart()

            loss, step_time, batch_num_sequences, batch_seqlen, grad_norm = train_step(
                args,
                total_steps,
                batch_idx,
                nvtx_warmup_iters,
                data_pipeline,
                input_data,
                model,
                optimizer,
                lr_scheduler,
                writers,
                fp8_recipe,
            )
            total_steps += 1
            cur_seq_index += batch_num_sequences
            sample_processed = batch_num_sequences * dp_size
            throughput = sample_processed / step_time
            throughputs.append(throughput)

            tflops_per_gpu = compute_tflops(args, sample_processed, step_time, world_size)

            if not total_steps % args.logging_freq and args.log_reduced_training_loss > 0:
                loss_scalar = reduce_loss(loss)
            else:
                loss_scalar = loss.item()

            current_lr = lr_scheduler.get_lr()
            display_step = total_steps - 1
            if global_rank == 0:
                write_metrics_train_step(
                    writers,
                    display_step,
                    loss_scalar,
                    throughput,
                    tflops_per_gpu,
                    current_lr,
                    grad_norm,
                )
                if not total_steps % args.logging_freq:
                    log_train_metrics(
                        args,
                        total_steps,
                        display_step,
                        loss_scalar,
                        throughput,
                        tflops_per_gpu,
                        current_lr,
                        grad_norm,
                        throughputs,
                        num_params,
                        dp_size,
                        batch_seqlen,
                    )

            # evaluate on validation
            if args.validation_freq and not total_steps % args.validation_freq:
                cur_state = np.random.get_state()
                torch.cuda.empty_cache()
                val_loss, val_ppl = eval_model(model, data_pipeline, args.validation_batches)
                cur_val_seq_index += args.val_batch_size * args.validation_batches
                if global_rank == 0:
                    log_and_write_eval_metrics(writers, display_step, val_loss, val_ppl)
                model = model.train()
                if args.preserve_np_state > 0:
                    np.random.set_state(cur_state)

            # checkpoint
            if not total_steps % args.checkpoint_freq[0]:

                if isinstance(data_pipeline, GPTDataPipeline):
                    save_train_path_index = data_pipeline.cur_train_path
                else:
                    save_train_path_index = 0
                save_train_seq_index = cur_seq_index
                save_val_seq_index = cur_val_seq_index
                # technically we have processed save_train_seq_index sequences in this file
                # and so index to start from is save_train_seq_index
                user_content = {
                    "cli_args": args.__dict__,
                    "model_config": model_config,
                    "num_params": num_params,
                    "total_steps": total_steps,
                    "epoch": epoch,
                    "start_train_path_index": save_train_path_index,
                    "resume_from_sequence_number": save_train_seq_index,
                    "val_resume_from_sequence_number": save_val_seq_index,
                }

                subdir = f"{args.model_type}-{total_steps}steps"
                if global_rank == 0 and not re.match(_CHECKPOINT_DIR_REGEX, subdir):
                    raise ValueError(
                        f"Please double check hard-coded checkpoint subdir pattern: `{subdir}` "
                        f"not matching `{_CHECKPOINT_DIR_REGEX}`."
                    )

                if args.enable_memory_profiling > 0:
                    msg = f"({_DEFAULT_STATE_DICT_TYPE})"
                    memory_status(tag=f"Before ckpt {msg}", writers=writers, step=display_step)
                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    user_content,
                    get_sharding_strategy(args.sharding_strategy),
                    args.checkpoint_dir[0],
                    subdir,
                    args.num_kept_checkpoints[0],
                    checkpointing_pg_metadata,
                    tensor_parallel_degree=int(tsm.state.tensor_parallel_degree),
                    expert_parallel_degree=int(tsm.state.expert_parallel_degree),
                    checkpoint_type=args.checkpoint_type,
                )
                if args.enable_memory_profiling > 0:
                    msg = f"({_DEFAULT_STATE_DICT_TYPE})"
                    memory_status(tag=f"After ckpt {msg}", writers=writers, step=display_step)

        if isinstance(data_pipeline, GPTDataPipeline):
            incremented_in_epoch = data_pipeline.increment_path_in_epoch()
            if not incremented_in_epoch:
                # path index set to 0
                epoch += 1
        else:
            epoch += 1
    # Using median throughput across all steps, could be more robust.
    return total_steps, np.median(throughputs) if throughputs else 0


@record
def main(args):
    """Main function to train GPT."""
    global_start_time = time.time()

    # Sanity check for args.
    # - Checkpoints.
    # TODO(sliuxl): Supporting one single checkpoint dir now, and multiple dirs support is missing.
    ckpt_lens = (
        len(args.checkpoint_dir),
        len(args.checkpoint_freq),
        len(args.num_kept_checkpoints),
    )
    if len(set(ckpt_lens)) != 1:
        raise ValueError(f"Len mismtach for checkpoint dir, freq vs num to keep:  {ckpt_lens}.")

    if args.distributed_backend == "smddp":
        import smdistributed.dataparallel.torch.torch_smddp  # pylint: disable=unused-import

    dist.init_process_group(args.distributed_backend, timeout=datetime.timedelta(seconds=7200))
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    world_size = dist.get_world_size()

    if args.tensorboard_dir and global_rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        logger.info("Writing metrics for tensorboard to %s.", args.tensorboard_dir)
        writers = tuple(SummaryWriter(log_dir=tb_dir) for tb_dir in args.tensorboard_dir)
        table_str = create_args_table(args.__dict__)
        for writer in writers:
            writer.add_text("Arguments", table_str)
    else:
        writers = ()

    if args.nccl_test_log:
        report = utils.get_nccl_test_report(utils.parse_nccl_test_log(args.nccl_test_log))
        if report is not None and global_rank == 0:
            write_nccl_test_stats(writers, report)

    tsm.init()

    if args.use_smp_implementation < 1 < tsm.state.tensor_parallel_degree:
        args.use_smp_implementation = 1
        if global_rank == 0:
            logger.info(
                "Tensor parallelism is enabled as tensor_parallel_degree is set to %d > 1. "
                "Switching use_smp_implementation to 1 so we can use SMP optimized implementation.",
                tsm.state.tensor_parallel_degree
            )
    if args.use_smp_implementation:
        # For our Mem usage fix to TE, this needs to be True
        args.use_orig_params = 1

    if args.use_synthetic_data and args.validation_freq is not None:
        # Overriding validation freq to None as synthetic data
        args.validation_freq = None

    show_env_vars(0)

    if global_rank == 0:
        for index, (key, value) in enumerate(sorted(args.__dict__.items()), 1):
            logger.info("Arguments [%03d/%03d] %-30s: %s", index, len(args.__dict__), key, value)
        logger.info("Transformers version: %s", transformers.__version__)
        logger.info("World size = %d: # nodes = %d.", world_size, world_size / 8)

        gbs = (
            world_size
            * args.max_context_width
            * args.train_batch_size
            / tsm.state.tensor_parallel_degree
        )
        logger.info("Global batch size in tokens: %10d (%5.2fM).", gbs, gbs / 1024 ** 2)

    set_seed(args.seed)

    if args.enable_memory_profiling > 0:
        memory_status_cpu(tag="Before model creation", writers=writers)

    if args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.get_default_dtype()

    if finetune_check(args):
        from transformers import AutoConfig

        # Using config for finetune mode, else uses args to create model
        model_config = AutoConfig.from_pretrained(args.hf_pretrained_model_name_or_dir)
        if hasattr(model_config, "use_cache"):
             model_config.use_cache = False
    else:
        model_config = get_model_config(args)

    delayed_param_initer = None
    with tsm_utils.timeit(True, "Model creation", global_rank):
        if args.delayed_param:
            if finetune_with_pretrained_weights_check(args) and dist.get_rank() == 0:
                # create model with pretrained weights on one rank even if we want to use
                # delayed param, param init on other ranks will still be delayed
                model = create_model(
                    args,
                    model_config=model_config,
                    dtype=dtype,
                    pretrained_model_weights=args.hf_pretrained_model_name_or_dir
                    if finetune_with_pretrained_weights_check(args)
                    else None,
                )
                num_params = compute_num_params(model)
            else:
                with init_empty_weights():
                    model = create_model(
                        args,
                        model_config=model_config,
                        dtype=dtype,
                    )
                num_params = compute_num_params(model)
            if finetune_check(args):
                dist.barrier()
        else:
            model = create_model(
                args,
                model_config=model_config,
                dtype=dtype,
                pretrained_model_weights=args.hf_pretrained_model_name_or_dir
                if finetune_with_pretrained_weights_check(args) and dist.get_rank() == 0
                else None,
            )
            num_params = compute_num_params(model)

        if args.use_smp_implementation:
            if args.moe:
                from torch.sagemaker.moe.moe_config import MoEConfig
                moe_config = MoEConfig(
                    smp_moe=args.use_smp_implementation > 0,
                    random_seed=args.seed,
                    moe_load_balancing=args.moe_load_balancing,
                    global_token_shuffle=args.global_token_shuffle > 0,
                    moe_all_to_all_dispatcher=args.moe_all_to_all_dispatcher > 0,
                    use_cpu_initialization=finetune_with_pretrained_weights_check(args) and dist.get_rank() == 0,
                )
            else:
                moe_config = None
            load_state_dict_from_rank0 = finetune_with_pretrained_weights_check(args)
            if args.moe and args.delayed_param and (not load_state_dict_from_rank0 or dist.get_rank() != 0):
                with init_empty_weights():
                    model = transform(model, config=moe_config, load_state_dict_from_rank0=load_state_dict_from_rank0)
            else:
                model = transform(model, config=moe_config, load_state_dict_from_rank0=load_state_dict_from_rank0)

        if args.delayed_param:
            # param init fn for delayed param creation
            if finetune_with_pretrained_weights_check(args):
                if dist.get_rank() != 0:
                    delayed_param_initer = DelayedParamIniter(model)
            else:
                delayed_param_initer = DelayedParamIniter(model)

    assert set(x.dtype for x in model.parameters()) == set(
        [torch.float32]
    ), "Model parameters should be in fp32 for FSDP mixed precision"

    if global_rank == 0:
        logger.info(
            "Created model with total parameters: %d (%.2f B)", num_params, num_params * 1e-9
        )

    transformer_layer = get_transformer_layer(args.model_type, args.use_smp_implementation,
                                              args.moe)

    if args.auto_wrap_policy == "transformer_auto_wrap_policy":
        gpt_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                transformer_layer,
            },
        )
    elif args.auto_wrap_policy == "size_based_auto_wrap_policy":
        gpt_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
        )

    torch.cuda.set_device(device)
    if args.bf16:
        # buffer set to fp32 as some models in HF such as llama hard code buffers to fp32
        # to be similar with that we set this to fp32
        buffer_dtype = torch.float32 if args.use_smp_implementation else dtype
        mixed_precision_policy = MixedPrecision(
            param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=buffer_dtype
        )
    else:
        mixed_precision_policy = None

    if args.enable_memory_profiling > 0:
        memory_status_cpu(tag="Before FSDP wrapper", writers=writers)

    sharding_strategy = get_sharding_strategy(args.sharding_strategy)

    with (
        delayed_param_initer.validate_params_and_buffers_inited()
        if (delayed_param_initer and not finetune_with_pretrained_weights_check(args))
        else nullcontext(),
        tsm_utils.timeit(True, "FSDP constructor", global_rank),
    ):
        model = FSDP(  # pylint: disable=unexpected-keyword-arg
            model,
            auto_wrap_policy=gpt_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            backward_prefetch=get_backward_fetch_policy(args.backward_fetch_policy),
            forward_prefetch=args.forward_prefetch,
            limit_all_gathers=args.limit_all_gathers,
            device_id=torch.cuda.current_device(),
            use_orig_params=args.use_orig_params > 0,
            param_init_fn=delayed_param_initer.get_param_init_fn()
            if delayed_param_initer
            else None,
            post_param_init_fn=delayed_param_initer.get_post_param_init_fn()
            if delayed_param_initer
            else None,
            sync_module_states=finetune_with_pretrained_weights_check(args),
        )
    # Barrier is a workaround to reduce extra memory usage with SMDDP backend
    # after the broadcast that happens when we use sync_module_states
    # This can be removed once the SMDDP issue is fixed
    dist.barrier()

    if global_rank == 0:
        logger.info("Wrapped model with FSDP")

    if args.enable_memory_profiling > 0:
        memory_status(tag="After FSDP wrapper", writers=writers)

    fp8_recipe = None
    if args.fp8==1 and args.use_smp_implementation==1:
        fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=args.fp8_amax_history_len, amax_compute_algo=args.fp8_amax_compute_algo)

    if args.activation_checkpointing > 0:
        apply_activation_checkpoint(args, model=model)

    if tsm.state.sm_activation_offloading > 0:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper

        model = offload_wrapper(model)

        # Patch RoPE for GPT NEoX where they are created on Host to move them to Device
        if args.use_smp_implementation == 0 and args.model_type == "gpt_neox" and args.patch_neox_rope > 0:
            patch_neox_rope(model)

    param_groups = get_param_groups_by_weight_decay(model)

    optimizer = optim.AdamW(
        param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay
    )

    if global_rank == 0:
        logger.info("Created optimizer")

    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    checkpointing_pg_metadata = (
        model.process_group,
        get_coordinator_rank(model.process_group),
        is_action_rank(global_rank),
    )

    if args.resume_from_checkpoint:
        (
            model,
            optimizer,
            lr_scheduler,
            epoch,
            total_steps,
            start_train_path_index,
            resume_from_sequence_number,
            val_resume_from_sequence_number,
        ) = load_checkpoint(
            args,
            model,
            optimizer,
            lr_scheduler,
            args.resume_from_checkpoint,
            sharding_strategy,
            checkpointing_pg_metadata,
            tensor_parallel_degree=int(tsm.state.tensor_parallel_degree),
            expert_parallel_degree=int(tsm.state.expert_parallel_degree),
            checkpoint_type=args.checkpoint_type,
        )
        torch.cuda.empty_cache()

    else:
        total_steps = 0
        epoch = 0
        start_train_path_index = 0
        resume_from_sequence_number = 0
        val_resume_from_sequence_number = 0

    train_start_time = time.time()
    # total_steps, throughput, loss
    total_steps, _ = train(
        model,
        optimizer,
        lr_scheduler,
        writers,
        model_config,
        epoch,
        start_train_path_index,
        resume_from_sequence_number,
        val_resume_from_sequence_number,
        num_params,
        total_steps,
        args,
        global_rank,
        world_size,
        checkpointing_pg_metadata,
        fp8_recipe,
    )
    time_now = time.time()
    total_sec = time_now - global_start_time
    train_sec = time_now - train_start_time

    dist.barrier()

    if args.save_final_model:
        save_checkpoint(
            model,
            None,
            None,
            {"model_config": model_config},
            None,
            args.model_dir if args.model_dir is not None else args.checkpoint_dir[0],
            "" if args.model_dir is not None else "model",
            1,
            None,
            int(tsm.state.tensor_parallel_degree),
            int(tsm.state.expert_parallel_degree),
            checkpoint_type=CheckpointingMethod.FULL,
        )

    if global_rank == 0:
        train_min = train_sec / 60.0
        total_min = total_sec / 60.0

        for writer in writers:
            runtime = {
                "total": total_min,
                "train": train_min,
            }
            writer.add_scalars("Perf/runtime", runtime, total_steps - 1)

        logger.info(
            "FSDP training finished successfully %fs (%fmin) out of (%fmin).",
            train_sec, train_min, total_min
        )

    dist.destroy_process_group()
