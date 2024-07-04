import torch


def build_param_id_to_offset(param_groups):
    param_id_to_offset = []
    for i, group in enumerate(param_groups):
        offset = 0
        group_offsets = {}
        for p in group["params"]:
            size = p.ds_tensor.ds_numel
            group_offsets[id(p)] = (offset, size)
            offset += size
        param_id_to_offset.append(group_offsets)
    return param_id_to_offset


def build_param_id_to_buffer(optimizer, param_id_to_offset):
    param_id_to_buffer = {}
    for i, group in enumerate(optimizer.param_groups):
        for _id, (offset, sz) in param_id_to_offset[i].items():
            buf = optimizer.fp32_partitioned_groups_flat[i].narrow(0, offset, sz)
            param_id_to_buffer[_id] = buf
    return param_id_to_buffer


def log_param_norms(model, optimizer, param_id_to_buffer):
    weight_norms = {}
    other_norms = {}
    for name, param in model.named_parameters():
        buf = param_id_to_buffer[id(param)]
        param_norm = torch.linalg.norm(buf) ** 2
        other_norm = torch.linalg.norm(param.ds_tensor.data) ** 2
        torch.distributed.all_reduce(param_norm, group=optimizer.ds_param_shard_group)
        torch.distributed.all_reduce(other_norm, group=optimizer.ds_param_shard_group)
        weight_norms[name] = torch.sqrt(param_norm).item()
        other_norms[name] = torch.sqrt(other_norm).item()
        if smp.rank() == 0:
            print(f"{name}: {weight_norms[name]} {other_norms[name]}")
