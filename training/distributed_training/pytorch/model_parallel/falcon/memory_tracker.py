import os

import psutil
import smdistributed.modelparallel.torch as smp
import torch

try:
    from py3nvml import py3nvml
except ImportError:
    py3nvml = None

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


def memory_status(msg="", reset_max=True, sync=True):

    rank = smp.rank()
    tp_rank = smp.tp_rank()
    pp_rank = smp.pp_rank()
    rdp_rank = smp.rdp_rank()
    local_rank = smp.local_rank()

    if sync:
        torch.cuda.synchronize()

    if rdp_rank != 0:
        return

    if py3nvml != None:
        py3nvml.nvmlInit()
        handle = py3nvml.nvmlDeviceGetHandleByIndex(local_rank)
        info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        total_used = info.used / 1024**3
        total_used_str = f"Totally used GPU memory: {total_used}"
    else:
        total_used_str = ""

    alloced = torch.cuda.memory_allocated(device=local_rank)
    max_alloced = torch.cuda.max_memory_allocated(device=local_rank)
    cached = torch.cuda.memory_reserved(device=local_rank)
    max_cached = torch.cuda.max_memory_reserved(device=local_rank)

    # convert to GB for printing
    alloced /= 1024**3
    cached /= 1024**3
    max_alloced /= 1024**3
    max_cached /= 1024**3

    print(
        f"[{msg}] rank {rank} tp_rank {tp_rank} pp_rank {pp_rank} TORCH {torch.__version__}",
        f"device={local_rank} "
        f"alloc {alloced:0.4f} max_alloced {max_alloced:0.4f} "
        f"cache {cached:0.4f} max_cached {max_cached:0.4f} "
        f"{total_used_str}",
    )
    if reset_max:
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()
    if py3nvml != None:
        py3nvml.nvmlShutdown()


def memory_status_cpu(msg=""):
    import gc

    global last_mem_usage
    global base_mem_usage
    rdp_rank = smp.rdp_rank()
    gc.collect()
    gc.collect()
    gc.collect()
    objects = gc.get_objects()
    tensors = [obj for obj in objects if isinstance(obj, torch.Tensor) and not obj.is_cuda]
    torch_usage = 0
    for t in tensors:
        torch_usage += t.numel() * dtype_to_bit[t.dtype]
    # total_usage = psutil.virtual_memory()[3] # This will get the total usage for all processes
    current_usage = process.memory_info().data
    total_usage = current_usage - base_mem_usage
    usage_change = current_usage - last_mem_usage
    last_mem_usage = current_usage

    torch_usage /= 1024**3
    total_usage /= 1024**3
    usage_change /= 1024**3
    base_usage = base_mem_usage / 1024**3

    rank = smp.rank()
    tp_rank = smp.tp_rank()
    pp_rank = smp.pp_rank()
    rdp_rank = smp.rdp_rank()
    local_rank = smp.local_rank()
    if rdp_rank != 0:
        return

    print(
        f"[{msg}] rank {rank} tp_rank {tp_rank} pp_rank {pp_rank} TORCH {torch.__version__}",
        f"device={local_rank} "
        f"torch cpu tensor usage {torch_usage:0.4f} cpu mem usage {total_usage:0.4f} change since last measurement {usage_change:0.4f} base cpu mem usage {base_usage:0.4f}",
    )
