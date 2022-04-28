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

"""Gradient clipping."""

import torch
from torch._six import inf
import smdistributed.modelparallel.torch as smp

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C


def clip_grad_norm_fp32(parameters, param_is_distributed, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    torch.cuda.set_device(smp.local_rank())
    grads = []
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = not hasattr(param, "shared") or not param.shared
        is_not_tp_duplicate = smp.tp_rank() == 0 or (param in param_is_distributed and param_is_distributed[param])
        if grad_not_none:
            grad = param.grad.detach()
            # Make sure the grads are in fp32
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grads.append(grad)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grads_for_norm.append(grad)

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.tensor(0.0, device=torch.device("cuda"))

    # Calculate norm.
    if norm_type == inf:
        if len(grads_for_norm) > 0:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=smp.get_mp_process_group())
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.cuda.IntTensor([0], device=torch.device("cuda", smp.local_rank()))
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if len(grads_for_norm) > 0:
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False # no per-parameter norm
                )
                # Since we will be summing across data parallel groups,
                # we need the pow(norm-type).
                total_norm = grad_norm ** norm_type
        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=smp.get_mp_process_group())
        total_norm = total_norm.item() ** (1.0 / norm_type)

    # Scale.
    if len(grads) > 0:
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0:
            dummy_overflow_buf = torch.cuda.IntTensor([0], device=torch.device("cuda", smp.local_rank()))
            multi_tensor_applier(amp_C.multi_tensor_scale,
                                 dummy_overflow_buf,
                                 [grads, grads],
                                 clip_coeff)

    return total_norm


def count_zeros_fp32(parameters):

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = 0.0
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = not hasattr(param, "shared") or not param.shared
        is_not_tp_duplicate = smp.tp_rank() == 0 or (param in param_is_distributed and param_is_distributed[param])
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(total_num_zeros,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=smp.get_mp_process_group())
    total_num_zeros = total_num_zeros.item()

    return total_num_zeros
