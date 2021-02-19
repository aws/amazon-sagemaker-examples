"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle
import time
import torch
import os
import torch.distributed as dist
#from maskrcnn_benchmark.utils.herring_env import is_herring
import smdistributed.modelparallel.torch as smp

run_herring = False
#if is_herring():
#    import herring.torch as herring
#    run_herring = True

def get_world_size():
    """ Note: Post-SMP-integration, this is actually the dp size """
    return smp.dp_size()

    #if run_herring:
    #    return herring.get_world_size()
    #else:
    #    if not dist.is_available():
    #        return 1
    #    if not dist.is_initialized():
    #        return 1
    #    return dist.get_world_size()


def reduce(tensor, dst_rank=0):
    dist.reduce(tensor, dst_rank, group=smp.get_dp_process_group())


def broadcast(tensor, src_rank=0):
    dist.broadcast(tensor, src_rank, group=smp.get_dp_process_group())


def get_local_rank():
    return smp.local_rank()

    #if not run_herring:
    #    local_rank = os.getenv('LOCAL_RANK', 0)
    #    return local_rank
    #return dist.get_local_rank()


def get_rank():
    """ Note: Post-SMP-integration, this is actually the dp rank """

    return smp.dp_rank()
    #if run_herring:
    #    return herring.get_rank()
    #else:
    #    if not dist.is_available():
    #        return 0
    #    if not dist.is_initialized():
    #        return 0
    #    return dist.get_rank()

def get_mp_rank():
    return smp.mp_rank()

def is_main_process():
    return smp.rank() == 0

    #if run_herring:
    #    herring.barrier()
    #else:
    #    if not dist.is_available():
    #        return
    #    if not dist.is_initialized():
    #        return
    #    world_size = dist.get_world_size()
    #    if world_size == 1:
    #        return
    #    dist.barrier()




def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    smp.barrier()

    #if not run_herring:
    #    if not dist.is_available():
    #        return
    #    if not dist.is_initialized():
    #        return
    #    world_size = dist.get_world_size()
    #    if world_size == 1:
    #        return
    #dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if run_herring:
        return data
    else:
        world_size = get_world_size()
        if world_size == 1:
            return [data]

        # serialized to a Tensor
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")

        # obtain Tensor size of each rank
        local_size = torch.IntTensor([tensor.numel()]).to("cuda")
        size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        # receiving Tensor from all ranks
        # we pad the tensor because torch all_gather does not support
        # gathering tensors of different shapes
        tensor_list = []
        for _ in size_list:
            tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
        if local_size != max_size:
            padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
            tensor = torch.cat((tensor, padding), dim=0)
        dist.all_gather(tensor_list, tensor, group=smp.get_dp_process_group())

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))

        return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    if run_herring:
        return input_dict
    else:
        world_size = get_world_size()
        if world_size < 2:
            return input_dict
        with torch.no_grad():
            names = []
            values = []
            # sort the keys so that they are consistent across processes
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            dist.reduce(values, dst=0, group=smp.get_dp_process_group())
            if dist.get_rank() == 0 and average:
                # only main process gets accumulated, so only divide by
                # world_size in this case
                values /= world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

