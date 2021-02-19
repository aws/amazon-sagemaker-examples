# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time

import torch

import maskrcnn_benchmark.utils.comm as comm
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
import smdistributed.modelparallel.torch as smp

from apex import amp

class Targets:
    def __init__(self, targ):
        self.targets = targ
    def smp_slice(self, num_mb, mb, axis):
        slice_size = len(self.targets) // num_mb
        return self.targets[mb*slice_size : (mb+1)*slice_size]


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = comm.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        comm.reduce(all_losses, dst=0)
        if comm.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


# Instead of zeroing, set parameter grads to None
# Prevents extraneous copy as we're not accumulating
def set_grads_to_none(model):
    for param in model.parameters():
        param.grad = None

@smp.step
def forward_backward(model, optimizer, images, targets):
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.backward(losses)
    return loss_dict


def do_train(
    model,
    iters_per_epoch,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    disable_allreduce_for_logging,
    per_iter_start_callback_fn=None,
    per_iter_end_callback_fn=None,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    def prefetcher(load_iterator):
        prefetch_stream = torch.cuda.Stream()
        pad_batches = []

        def _prefetch():
            try:
                # I'm not sure why the trailing _ is necessary but the reference used
                # "for i, (images, targets, _) in enumerate(data_loader):" so I'll keep it.
                images, targets, _ = next(load_iterator)
            except StopIteration:
                return None, None

            with torch.cuda.stream(prefetch_stream):
                # TODO:  I'm not sure if the dataloader knows how to pin the targets' datatype.
                targets = [target.to(device, non_blocking=True) for target in targets]
                images = images.to(device, non_blocking=True)

            return images, targets

        next_images, next_targets = _prefetch()

        while next_images is not None:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
            current_images, current_targets = next_images, next_targets
            next_images, next_targets = _prefetch()
            yield current_images, current_targets
    
    torch.cuda.set_device(torch.device("cuda", smp.local_rank()))
    comm.synchronize()
    optimizer.zero_grad()
    for iteration, (images, targets) in enumerate(prefetcher(iter(data_loader)), start_iter):
        if per_iter_start_callback_fn is not None:
            per_iter_start_callback_fn(iteration=iteration)

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = forward_backward(model, optimizer, images, Targets(targets))
        loss_dict = {k: v.reduce_mean() for k, v in loss_dict.items()}        

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        if not disable_allreduce_for_logging:
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
        else:
            meters.update(loss=losses, **loss_dict)

        # optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with optimizer.scale_loss(losses) as scaled_losses:
        #optimizer.backward(losses)
        optimizer.update_master_grads()
        optimizer.step()
        # set_grads_to_none(model)
        optimizer.zero_grad()
        scheduler.step()

        batch_time = time.time() - end
        throughput = arguments["global_batch_size"] / batch_time
        end = time.time()
        meters.update(time=batch_time, data=data_time, throughput=throughput)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            next_epoch = (iteration // iters_per_epoch + 1) * iters_per_epoch - iteration
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter/next_eval: {iter}/{next_epoch}",
                        "throughput: {throughput} img/sec",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    next_epoch=next_epoch,
                    meters=str(meters),
                    throughput=meters.throughput.global_avg,
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        # TODO add saving/checkpointing
        if iteration % checkpoint_period == 0 and arguments["save_checkpoints"]:
            arguments["save_partial"] = True 
            checkpointer.save("model_{:07d}_partial".format(iteration), **arguments)
            comm.synchronize()

        if iteration == max_iter and arguments["save_checkpoints"]:
            checkpointer.save("model_final", save_partial=False, **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            # Note: iteration has been incremented previously for
            # human-readable checkpoint names (i.e. 60000 instead of 59999)
            # so need to adjust again here
            early_exit = per_iter_end_callback_fn(iteration=iteration-1)
            if early_exit:
                break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if per_iter_end_callback_fn is not None:
        if early_exit:
            return True
        else:
            return False
    else:
        return None
