"""Memory tracker."""

import os
from typing import Any, Tuple

import psutil
import torch
import torch.distributed as dist

try:
    from py3nvml import py3nvml
except ImportError:
    py3nvml = None

# pylint: disable=global-statement
dtype_to_bit = {
    torch.float32: 32,
    torch.float64: 64,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.bool: 1,
}

process = psutil.Process(os.getpid())
base_mem_usage = process.memory_info().data
last_mem_usage = base_mem_usage

_GB = 1024**3
_FORMAT = "7.4f"


def memory_status(  # pylint: disable=too-many-locals
    tag: str = "",
    reset_max: bool = True,
    sync: bool = True,
    writers: Tuple[Any] = (),
    step: int = 0,
) -> Tuple[float]:
    """Memory status gpu."""
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()

    if rank > 0:
        return 0., 0., 0., 0.

    if sync:
        torch.cuda.synchronize()

    if py3nvml is not None:
        py3nvml.nvmlInit()
        handle = py3nvml.nvmlDeviceGetHandleByIndex(local_rank)
        info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        total_used = info.used / _GB
        total_used_str = f"Totally used GPU memory: {total_used} GB."
    else:
        total_used_str = ""

    # Convert to GB for printing.
    alloced = torch.cuda.memory_allocated(device=local_rank) / _GB
    max_alloced = torch.cuda.max_memory_allocated(device=local_rank) / _GB
    cached = torch.cuda.memory_reserved(device=local_rank) / _GB
    max_cached = torch.cuda.max_memory_reserved(device=local_rank) / _GB

    print(
        f"[GPU MEMORY]@{step:04d} "
        f"(torch, rank, device) = ({torch.__version__}, {rank}, {local_rank}), "
        f"(alloc, max_alloc, cache, max_cache) = ({alloced:{_FORMAT}}, {max_alloced:{_FORMAT}}, "
        f"{cached:{_FORMAT}}, {max_cached:{_FORMAT}}) GB. "
        f"{total_used_str} [{tag:10s}]",
    )

    if reset_max:
        torch.cuda.reset_peak_memory_stats()

    if py3nvml is not None:
        py3nvml.nvmlShutdown()

    usage = {
        "allocated": alloced,
        "max_allocated": max_alloced,
        "max_reserved": max_cached,
        "reserved": cached,
    }
    for writer in writers:
        writer.add_scalars(f"GPUMemoryGB/{tag}", usage, step)

    return alloced, max_alloced, cached, max_cached


def memory_status_cpu(  # pylint: disable=too-many-locals
    tag: str = "", writers: Tuple[Any] = (), step: int = 0
) -> Tuple[float]:
    """Memory status cpu."""
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()

    if rank > 0:
        return 0., 0., 0., 0.

    import gc  # pylint: disable=import-outside-toplevel

    global last_mem_usage
    global base_mem_usage  # pylint: disable=global-variable-not-assigned

    gc.collect()
    gc.collect()
    gc.collect()
    objects = gc.get_objects()
    tensors = [obj for obj in objects if isinstance(obj, torch.Tensor) and not obj.is_cuda]
    torch_usage = 0
    for t in tensors:  # pylint: disable=invalid-name
        torch_usage += t.numel() * dtype_to_bit[t.dtype]
    # total_usage = psutil.virtual_memory()[3] # This will get the total usage for all processes
    current_usage = process.memory_info().data
    total_usage = current_usage - base_mem_usage
    usage_change = current_usage - last_mem_usage
    last_mem_usage = current_usage

    torch_usage /= _GB
    total_usage /= _GB
    usage_change /= _GB
    base_usage = base_mem_usage / _GB

    print(
        f"[CPU MEMORY]@{step:04d} "
        f"(torch, rank, device) = ({torch.__version__}, {rank}, {local_rank}), "
        f"(torch tensor, mem, change since last measurement, base) = ({torch_usage:{_FORMAT}}, "
        f"{total_usage:{_FORMAT}}, {usage_change:{_FORMAT}}, {base_usage:{_FORMAT}}): "
        f"{tag}"
    )

    usage = {
        "base": base_usage,
        "delta": usage_change,
        "torch": torch_usage,
        "total": total_usage,
    }
    for writer in writers:
        writer.add_scalars(f"CPUMemoryGB/{tag}", usage, step)

    return torch_usage, total_usage, usage_change, base_usage
