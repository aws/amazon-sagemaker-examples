"""Logging utils."""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch.distributed as dist

_logger = None


def create_args_table(args: Dict) -> str:
    """Create args table."""
    table_str = ""
    table_header = "|" + "#" + "|" + "Arguments" + "|" + "Value" + "|" + "\n"
    separator = "|-----" * 3 + '|' + "\n"
    table_str += table_header + separator
    for idx, (key, col) in enumerate(sorted(args.items())):
        table_row = f"| {idx} | {key} | {col} |\n"
        table_str += table_row
    return table_str


def get_logger():
    """Get logger."""
    global _logger
    if _logger is None:
        logging.getLogger("torch.distributed.checkpoint._dedup_tensors").setLevel(logging.ERROR)
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
        _logger = logging.getLogger(__name__)
        _logger.setLevel(logging.INFO)
        _logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname).1s " "[%(filename)s:%(lineno)d] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        _logger.addHandler(ch)
        _logger.propagate = False
    return _logger


def show_env_vars(rank: Optional[int] = 0):
    """Show env vars."""
    my_rank = dist.get_rank()

    env_var = os.environ
    if rank is None or my_rank == rank:
        _logger.info("Env variables (len = %d):", len(env_var))

        count = 0
        for key, value in sorted(env_var.items()):
            _logger.info(
                "[%03d] env [%03d/%03d] %-20s: `%s`", my_rank, count, len(env_var), key, value
            )
            count += 1

    keys = (
        "HOSTNAME",
        "SLURM_PROCID",
    )
    values = tuple(str(env_var.get(key)) for key in keys)
    if my_rank % 8 == 0:  # Print from each node exactly once.
        _logger.info("[%03d] env from all nodes `%s`: `%s`", my_rank, keys, values)


def write_nccl_test_stats(
    writers, report: Optional[Dict[str, Any]], prefix: str = "", step: int = -1
) -> None:
    """Write NCCL test stats."""

    # 1. Different units and scale.
    separate_fields = ("len",)
    # 2. Bandwidth: Scalars and vectors.
    stats_fields = (
        "min", "min2", "min3", "min4", "min5", "max", "max2", "max3", "max4", "max5",
        "mean", "median", "std"
    )
    vector_fields = ("data", "data_sorted")

    for writer in writers:
        for field in separate_fields:
            if field in report:
                writer.add_scalar(f"NCCLTest/{prefix}{field}", report[field], step)

        # Bandwidth.
        # - Scalars.
        stats = {field: report[field] for field in stats_fields if field in report}
        if stats:
            _logger.info("NCCL test stats: `%s`.", stats)
            writer.add_scalars(f"NCCLTest/{prefix}stats", stats, step)

        # - Vectors.
        for field in vector_fields:
            if field not in report:
                continue
            vector = report[field]
            _logger.info("NCCL test vectors (`%s`, len = %02d): `%s`.", field, len(vector), vector)

            writer.add_histogram(f"NCCLTest/{prefix}{field}-hist", vector, step)
            # When written as a scalar, its *max* step is written at the given step.
            for index, value in enumerate(np.flip(vector)):
                writer.add_scalar(f"NCCLTest/{prefix}{field}", value, step - index)


def write_metrics_train_step(
    writers, display_step, loss_scalar, throughput, tflops_per_gpu, current_lr, grad_norm
):
    """Write train metrics."""
    for writer in writers:
        writer.add_scalar("Loss/train", loss_scalar, display_step)
        writer.add_scalar("Perf/SeqPerSec", throughput, display_step)
        writer.add_scalar("Perf/ModelTFLOPs", tflops_per_gpu, display_step)
        writer.add_scalar("LR/learning_rate", current_lr, display_step)
        writer.add_scalar("Norms/grad_norm", grad_norm, display_step)


def log_train_metrics(
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
    world_size,
    batch_seqlen,
):
    """Log train metrics."""
    _logger.info(
        "Batch %d Loss: %s, Speed: %.2f samples/sec, Model TFLOPS/GPU: %.2f, lr: %.6f, gradnorm: %.4f",  # pylint: disable=line-too-long
        display_step,
        loss_scalar,
        throughput,
        tflops_per_gpu,
        current_lr,
        grad_norm,
    )

    # Compute average throughput and tflops after 30 steps to remove
    # high variance in initial steps
    if len(throughputs) > 30 and not total_steps % args.logging_freq_for_avg:
        avg_throughput = np.average(throughputs[30:])
        from train_utils import compute_tflops

        avg_tflops = compute_tflops(avg_throughput, num_params, world_size, batch_seqlen)
        _logger.info(
            "Batch %d Running Avg Speed: %.2f samples/sec, Running Avg Model TFLOPS/GPU: %.2f",  # pylint: disable=line-too-long
            display_step,
            avg_throughput,
            avg_tflops,
        )


def log_and_write_eval_metrics(writers, display_step, val_loss, val_ppl):
    """Log and write eval metrics."""
    for writer in writers:
        writer.add_scalar("Loss/val", val_loss, display_step)
        writer.add_scalar("Loss/perplexity", val_ppl, display_step)

    _logger.info(
        "Batch %d Validation loss: %s",
        display_step,
        val_loss,
    )
    _logger.info(
        "Batch %d Validation perplexity: %s",
        display_step,
        val_ppl,
    )
