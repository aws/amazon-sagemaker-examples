import torch
import glob
import math
import os
import re
import gc
from collections import OrderedDict

# load to cpu
device = torch.device('cpu')
smp_prefix = "module."

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_model_state_file(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")
    file = os.path.join(checkpoint_dir, "model_0.pt")

    if not os.path.exists(file):
        raise FileNotFoundError(f"can't find model states file at '{file}'")

    return file

def get_optim_files(checkpoint_dir):
    optim_files = sorted(glob.glob(os.path.join(checkpoint_dir, "optimizer_*.pt")), key=natural_keys)

    if len(optim_files) == 0:
        raise FileNotFoundError(
            f"can't find '*_optim_states.pt' files in directory '{checkpoint_dir}'")

    return optim_files

def get_user_content_file(checkpoint_dir):
    file = os.path.join(checkpoint_dir, "user_content.pt")
    if not os.path.exists(file):
        raise FileNotFoundError(f"can't find user content file at '{file}'")
    return file

def parse_model_state(model_file, user_content_file, dtype):
    state_dict = torch.load(model_file, map_location=device)
    user_content = torch.load(user_content_file, map_location=device)

    if "buffer_names" not in user_content:
        raise ValueError(f"{user_content_file} miss buffer_names to reconstruct the full state")
    if "param_shapes" not in user_content:
        raise ValueError(f"{user_content_file} miss param_shapes to reconstruct the full state")
    buffer_names = user_content["buffer_names"]
    param_shapes = user_content["param_shapes"]

    # recover just the buffers while restoring them to the specified dtype
    buffers = {
        k: v.to(dtype)
        for k,
        v in state_dict["module"].items() if k in buffer_names
    }

    return buffers, param_shapes

def parse_optim_states(files, checkpoint_dir, dtype):
    total_files = len(files)
    state_dicts = []
    sharded_data_parallel_size = None
    # param_shapes = None
    fp32_groups_key = None
    for i, f in enumerate(files):
        states = torch.load(f, map_location=device)
        if i == 0:
            sharded_data_parallel_size = states["partition_count"]
        states["fp32_flat_groups"] = [group.to(dtype) for group in states["fp32_flat_groups"]]
        state_dicts.append(states["fp32_flat_groups"])

    if type(sharded_data_parallel_size) is list:
        sharded_data_parallel_size = max(sharded_data_parallel_size)

    if sharded_data_parallel_size != total_files:
        raise ValueError(
            f"Expected {sharded_data_parallel_size} of 'optimizer_*.pt' under '{checkpoint_dir}' but found {total_files} files. "
            "Possibly due to an overwrite of an old checkpoint, or a checkpoint didn't get saved by one or more processes."
        )

    flat_groups = [
        torch.cat(state_dicts[i],
                  0) for i in range(len(state_dicts))
    ]    

    return sharded_data_parallel_size, flat_groups

def partitioned_param_info(unpartitioned_numel, sharded_data_parallel_size):
    remainder = unpartitioned_numel % sharded_data_parallel_size
    padding_numel = (sharded_data_parallel_size - remainder) if remainder else 0
    partitioned_numel = math.ceil(unpartitioned_numel / sharded_data_parallel_size)
    return partitioned_numel, padding_numel
    
def get_full_state_dict_from_sharded_data_parallel_checkpoint(checkpoint_dir, dtype=torch.float32, tag=None, remove_smp_prefix=True):
    """
    Returns full state_dict reconstructed from sharded data parallel checkpoint

    Args:
        - checkpoint_dir: path to the sharded data parallel checkpoint folder (where the optimizer files are)
        - dtype: the dtype of the output full checkpoint
        - tag: the checkpoint tag, if not specified will read the newest checkpoint
        - remove_smp_prefix: remove the "module." prefix created by smp

    """
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, 'newest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'newest' file at {latest_path}")

    checkpoint_dir = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")

    print(f"Processing checkpoint '{checkpoint_dir}'")

    optim_files = get_optim_files(checkpoint_dir)
    sharded_data_parallel_size, flat_groups = parse_optim_states(optim_files, checkpoint_dir, dtype)

    model_file = get_model_state_file(checkpoint_dir)
    user_content_file = get_user_content_file(checkpoint_dir)
    buffers, param_shapes = parse_model_state(model_file, user_content_file, dtype)
    
    gc.collect()
    avail_numel = flat_groups[0].numel() * sharded_data_parallel_size
    # merge list of dicts, preserving order
    param_shapes = {k: v for d in param_shapes for k, v in d.items()}
    
    # params
    offset = 0
    total_numel = 0
    total_params = 0

    state_dict = OrderedDict()
    state_dict.update(buffers)

    for name, shape in param_shapes.items():
        if remove_smp_prefix and name.startswith(smp_prefix):
            name = name[len(smp_prefix):]

        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        total_params += 1

        partitioned_numel, partitioned_padding_numel = partitioned_param_info(unpartitioned_numel, sharded_data_parallel_size)

        print(
            f"{total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
        )

        # memory usage doubles here
        state_dict[name] = torch.cat(
                tuple(flat_groups[i].narrow(0,
                                                 offset,
                                                 partitioned_numel) 
                    for i in range(sharded_data_parallel_size)),
                0).narrow(0,
                        0,
                        unpartitioned_numel).view(shape)
        offset += partitioned_numel

    offset *= sharded_data_parallel_size

    # Sanity check
    if offset != avail_numel:
        raise ValueError(
            f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(
        f"Reconstructed state dict with {total_params} params {total_numel} elements"
    )

    return state_dict

def get_param_shapes(model, optimizer):
    """Returns a dict of name to shape mapping, only for the flattened weights saved by the
    optimizer. the names are exactly as in state_dict. The order is absolutely important, since
    the saved data is just flattened data with no identifiers and requires reconstruction in the
    same order it was saved.

    We can't rely on module.named_parameters() to get the saved tensors, as some params
    will be missing and others unsaved and then it'd be impossible to reconstruct state_dict
    from the flattened weights.
    """
    param_group_shapes = []
    cnt = 0
    numel = 0

    bit16_groups = optimizer.fp16_groups
    param_names = {param: name for name, param in model.module.named_parameters()}

    for bit16_group in bit16_groups:
        param_shapes = OrderedDict()
        for param in bit16_group:
            cnt += 1
            numel += param.ds_numel if hasattr(param, "ds_numel") else param.numel()
            shape = param.ds_shape if hasattr(param, "ds_shape") else param.shape
            if param not in param_names:
                raise ValueError(f"failed to find optimizer param in named params")
            name = param_names[param]
            param_shapes[name] = shape

        param_group_shapes.append(param_shapes)

    return param_group_shapes

def get_buffer_names(model):
    buffer_names = []

    # we save buffer names so that we could extract later the real buffers from the saved
    # state_dict["module"] in the non-zero checkpoint - the buffers are already there but they
    # are intermixed with param placeholders

    # have to traverse the tree to be able to skip non-persistent buffers
    def get_layer_named_buffers(module, prefix=""):
        for name, buf in module.named_buffers(recurse=False):
            if buf is not None and name not in module._non_persistent_buffers_set:
                buffer_names.append(prefix + name)

        for name, child in module.named_children():
            if child is not None:
                get_layer_named_buffers(child, prefix + name + ".")

    get_layer_named_buffers(model.module, prefix="")

    return buffer_names