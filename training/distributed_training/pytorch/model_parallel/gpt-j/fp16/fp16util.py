# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.state_mod import state as smp_state


class tofp16(nn.Module):
    """
    Utility module that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def BN_convert_float(module):
    """
    Utility function for network_to_half().

    Retained for legacy purposes.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.

    Retained for legacy purposes. It is recommended to use FP16Model.
    """
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))


def convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data.to(dtype=dtype)
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data.to(dtype=dtype)

    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data.to(dtype=dtype)


def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
    return network


class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


def backwards_debug_hook(grad):
    raise RuntimeError("master_params recieved a gradient in the backward pass!")


def prep_param_lists(model, flat_master=False):
    """
    Creates a list of FP32 master parameters for a given model, as in
    `Training Neural Networks with Mixed Precision:  Real Examples`_.

    Args:
        model (torch.nn.Module): Existing Pytorch model
        flat_master (bool, optional, default=False):  Flatten the master parameters into a single tensor, as a performance optimization.
    Returns:
        A tuple (``model_params``, ``master_params``). ``model_params`` is a list of the model's parameters for later use with :func:`model_grads_to_master_grads` and :func:`master_params_to_model_params`.  ``master_params`` is a list of FP32 master gradients.  If ``flat_master=True``, ``master_params`` will be a list with one element.

    Example::

        model_params, master_params = prep_param_lists(model)

    .. warning::
        Currently, if ``flat_master=True``, all the model's parameters must be the same type.  If the model has parameters of different types, use ``flat_master=False``, or use :class:`FP16_Optimizer`.

    .. _`Training Neural Networks with Mixed Precision:  Real Examples`:
        http://on-demand.gputechconf.com/gtc/2018/video/S81012/
    """
    model_params = [param for param in model.parameters() if param.requires_grad]

    if flat_master:
        # Give the user some more useful error messages
        try:
            # flatten_dense_tensors returns a contiguous flat array.
            # http://pytorch.org/docs/master/_modules/torch/_utils.html
            master_params = _flatten_dense_tensors([param.data for param in model_params]).float()
        except BaseException:
            print("Error in prep_param_lists:  model may contain a mixture of parameters "
                  "of different types.  Use flat_master=False, or use F16_Optimizer.")
            raise
        master_params = torch.nn.Parameter(master_params)
        master_params.requires_grad = True
        # master_params.register_hook(backwards_debug_hook)
        if master_params.grad is None:
            master_params.grad = master_params.new(*master_params.size())
        return model_params, [master_params]
    else:
        master_params = [param.clone().float().detach() for param in model_params]
        for param in master_params:
            param.requires_grad = True
        return model_params, master_params


def model_grads_to_master_grads(model_params, master_params, flat_master=False, loss_scale=1.0, params_have_main_grad=False):
    """
    Copy model gradients to master gradients.

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.  If ``master_params`` was created with ``flat_master=True``, ``flat_master=True`` should also be supplied to :func:`model_grads_to_master_grads`.
    """
    if flat_master:
        # The flattening may incur one more deep copy than is necessary.
        master_params[0].grad.data.copy_(
            _flatten_dense_tensors([p.grad.data for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.device.type == "cpu":
                continue
            if model.grad is not None:
                if master.grad is None:
                    if params_have_main_grad:
                        # If gradient_as_bucket_view is False, this will be a copy
                        master.grad = model.grad.float()
                    else:
                        master.grad = Variable(master.data.new(*master.data.size()))
            else:
                master.grad = None
        model_grads = [p.grad for p in model_params if p.grad is not None]
        master_grads = [p.grad for p in master_params if p.grad is not None]
        if len(model_grads) == 0 or len(master_grads) == 0:
            return
        _overflow_buf = torch.cuda.IntTensor([0])
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             _overflow_buf,
                             [model_grads, master_grads],
                             1.0/loss_scale)


def master_params_to_model_params(model_params, master_params, flat_master=False):
    """
    Copy master parameters to model parameters.

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.  If ``master_params`` was created with ``flat_master=True``, ``flat_master=True`` should also be supplied to :func:`master_params_to_model_params`.
    """
    if flat_master:
        for model, master in zip(model_params,
                                 _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params):
            if model.device.type == "cpu":
                continue
            model.data.copy_(master.data)

def model_params_to_master_params(model_params, master_params, flat_master=False):
    """
    Copy model params to master params
    """
    if flat_master:
        raise ValueError("Not supported")
    else:
        for model, master in zip(model_params, master_params):
            if model.device.type == "cpu":
                continue
            master.data.copy_(model.data)


# Backward compatibility fixes


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])


def get_tp_merged_fp32_from_fp16_param_groups(optimizer, cpu_fp32_from_fp16_groups):
    def _merge_param_group_tp_group(group_idx, param_group):
        result_fp32_from_fp16_param_group = []
        param_name_group = {}
        for i, param in enumerate(param_group):
            # for each param, obtain param_name from param using two dicts above for tp_rank 0
            param_index = param_id_to_index_tp_group[rank_0][
                fp32_from_fp16_paramid_groups_tp_group[rank_0][group_idx][i]
            ]
            param_name = param_index_to_name_tp_group[rank_0][param_index]
            # obtain distribution axis for the param and check if its distributed
            # axis = master_distribution_axis_tp_rank_0[fp32_from_fp16_paramid_groups_tp_group[rank_0][group_idx][i]]
            axis = master_distribution_axis_tp_rank_0.get(
                fp32_from_fp16_paramid_groups_tp_group[rank_0][group_idx][i], None
            )
            if axis is not None:
                tensors = []
                for r in range(smp.tp_size()):
                    # if distributed, for each rank, obtain param id from index using above two dicts
                    param_index_r = param_name_to_index_tp_group[r][param_name]
                    param_id_r = param_index_to_id_tp_group[r][param_index_r]

                    # search param id in fp32_from_fp16_groups_param_ids and find the index.
                    group_param_idx = fp32_from_fp16_paramid_groups_tp_group[r][group_idx].index(
                        param_id_r
                    )
                    # use the param corresponding to the index from fp32_from_fp16_groups for concatenation along axis
                    tensors.append(
                        fp32_from_fp16_param_groups_tp_group[r][group_idx][group_param_idx]
                    )
                result_fp32_from_fp16_param_group.append(torch.cat(tensors, axis))
            else:
                # if not distributed set tp_rank 0 param as the param
                result_fp32_from_fp16_param_group.append(param)
            param_name_group[param_name] = i
        return result_fp32_from_fp16_param_group, param_name_group

    # get param_index_to_name all and param_name_to_index_all
    param_index_to_name_tp_group = smp_state.param_index_to_name_tp_group
    param_name_to_index_tp_group = smp_state.param_name_to_index_tp_group
    # get mapping of param_id_to_index_all and param_index_to_id_all
    param_id_to_index = optimizer._param_id_to_index()
    param_id_to_index_tp_group = smp.allgather(param_id_to_index, smp.TP_GROUP)
    param_index_to_id_tp_group = _get_param_index_to_id(param_id_to_index_tp_group)
    # allgather all param ids and all params for fp32_from_fp16_groups
    fp32_from_fp16_paramid_groups = optimizer.fp32_from_fp16_paramid_groups
    fp32_from_fp16_paramid_groups_tp_group = smp.allgather(
        fp32_from_fp16_paramid_groups, smp.TP_GROUP
    )
    fp32_from_fp16_param_groups_tp_group = smp.allgather(cpu_fp32_from_fp16_groups, smp.TP_GROUP)
    # broadcast distribution axis from tp_rank 0 to all tp_ranks
    master_distribution_axis_tp_rank_0 = None
    if smp.tp_rank() == 0:
        master_distribution_axis_tp_rank_0 = optimizer.master_distribution_axis
        smp.broadcast(master_distribution_axis_tp_rank_0, smp.TP_GROUP)
    else:
        master_distribution_axis_tp_rank_0 = smp.recv_from(0, smp.RankType.TP_RANK)

    result_fp32_from_fp16_param_groups = []
    param_name_groups = []
    rank_0 = 0
    # iterate through all the params for tp_group_fp32_from_fp16_groups[rank_0]
    for group_idx, param_group in enumerate(fp32_from_fp16_param_groups_tp_group[rank_0]):
        result_fp32_from_fp16_param_group, param_name_group = _merge_param_group_tp_group(
            group_idx, param_group
        )
        result_fp32_from_fp16_param_groups.append(result_fp32_from_fp16_param_group)
        param_name_groups.append(param_name_group)
    return result_fp32_from_fp16_param_groups, param_name_groups


def get_pp_merged_fp32_from_fp16_param_groups(
    optimizer, fp32_from_fp16_groups, param_name_groups=None
):
    pp_group_fp32_from_fp16_groups = smp.allgather(fp32_from_fp16_groups, smp.PP_GROUP)
    if param_name_groups is not None:
        index_to_param_name_groups = []
        # obtain index_to_param_name mapping across tp_group
        for param_name_group in param_name_groups:
            index_to_param_name = {}
            for param_name, index in param_name_group.items():
                index_to_param_name[index] = param_name
            index_to_param_name_groups.append(index_to_param_name)
        # allgather the index_to_param_name_groups across the pp_group
        pp_index_to_param_name_groups = smp.allgather(index_to_param_name_groups, smp.PP_GROUP)
    else:
        raise ValueError("Merging is not supported when param_name_groups is None")

    pp_merged_fp32_from_fp16_groups = []
    result_param_groups = []

    # iterate through all the groups for rank 0
    for group_idx in range(len(pp_group_fp32_from_fp16_groups[0])):
        merged = []
        start_idx = 0
        result_param_group = {}
        # for each group iterate through all ranks and merge the param groups across pp_ranks
        for rank, group in enumerate(pp_group_fp32_from_fp16_groups):
            cur_g = group[group_idx]
            start_idx += len(merged)
            for i, _ in enumerate(cur_g):
                param_name = pp_index_to_param_name_groups[rank][group_idx][i]
                if param_name in result_param_group:
                    raise ValueError(
                        "same param_name present in the param_groups of different pipeline parallel partitions"
                    )
                result_param_group[param_name] = i + start_idx
            merged.extend(cur_g)
        pp_merged_fp32_from_fp16_groups.append(merged)
        result_param_groups.append(result_param_group)
    return pp_merged_fp32_from_fp16_groups, result_param_groups


def _get_param_index_to_id(param_id_to_index_tp_group):
    param_index_to_id_tp_group = []
    for param_id_to_index_map in param_id_to_index_tp_group:
        param_index_to_id_map = {}
        for param_id, param_index in param_id_to_index_map.items():
            param_index_to_id_map[param_index] = param_id
        param_index_to_id_tp_group.append(param_index_to_id_map)
    return param_index_to_id_tp_group


def register_optimizer_hooks(model):
    def param_name_to_index(self):
        param_id_to_index = self._param_id_to_index()
        name_to_index = {}
        if self.redefined_params:
            param_gen = model.virtual_named_parameters()
        else:
            param_gen = model.named_parameters()
        for name, param in param_gen:
            fp16_param_id = id(param)
            if fp16_param_id in self.fp32paramid_from_fp16paramid:
                param_id = self.fp32paramid_from_fp16paramid[fp16_param_id]
            else:
                param_id = fp16_param_id
            if param_id in param_id_to_index:
                name_to_index[name] = param_id_to_index[param_id]
        return name_to_index

    def _param_index_to_param_local(self):
        param_id_to_index = self._param_id_to_index()
        param_index_to_param = {}

        if not model:
            return param_index_to_param

        if self.redefined_params:
            param_gen = model.virtual_named_parameters()
        else:
            param_gen = model.named_parameters()
        for name, param in param_gen:
            fp16_param_id = id(param)
            if fp16_param_id in self.fp32paramid_from_fp16paramid:
                param_id = self.fp32paramid_from_fp16paramid[fp16_param_id]
            else:
                param_id = fp16_param_id
            if param_id in param_id_to_index:
                param_index_to_param[param_id_to_index[param_id]] = param

        return param_index_to_param

    def hook_fn(model, optimizer):
        from functools import partial

        optimizer.param_name_to_index = partial(param_name_to_index, optimizer)
        optimizer._param_index_to_param_local = partial(_param_index_to_param_local, optimizer)

    model.register_post_partition_hook(hook_fn)
