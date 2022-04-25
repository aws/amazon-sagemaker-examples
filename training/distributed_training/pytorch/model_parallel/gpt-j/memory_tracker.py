import smdistributed.modelparallel.torch as smp
import torch


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
        f"cache {cached:0.4f} max_cached {max_cached:0.4f}",
    )
    if reset_max:
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()
