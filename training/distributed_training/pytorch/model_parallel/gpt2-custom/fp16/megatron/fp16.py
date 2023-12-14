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

"""Megatron optimizer."""
from abc import ABC, abstractmethod
from contextlib import contextmanager

import amp_C
import humanize
import smdistributed.modelparallel.torch as smp
import torch
import torch.nn as nn
from apex.multi_tensor_apply import multi_tensor_applier
from smdistributed.modelparallel.torch.state_mod import state as smp_state
from smdistributed.modelparallel.torch.utils import get_distribution_axis

from ..fp16util import (
    get_pp_merged_fp32_from_fp16_param_groups,
    get_tp_merged_fp32_from_fp16_param_groups,
    register_optimizer_hooks,
)
from .clip_grads import clip_grad_norm_fp32, count_zeros_fp32


def _multi_tensor_copy_this_to_that(this, that, overflow_buf=None):
    """Use multi-tensor-applier to copy values from one list to another.
    We don't have a blfoat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16."""
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


def _zero_grad_group_helper(group, set_to_none):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


@contextmanager
def measure_additional_mem_context():
    smp.barrier()
    mem_before = torch.cuda.memory_allocated(device=smp.local_rank())
    yield
    import gc

    gc.collect()
    gc.collect()
    gc.collect()
    mem_after = torch.cuda.memory_allocated(device=smp.local_rank())
    print(
        f"rank is {smp.local_rank()}, memory usage is {humanize.naturalsize(mem_after - mem_before)}"
    )
    smp.barrier()


def save_fp16_optimizer(args, model, optimizer, partial=True):
    state_dict = {}
    # state_dict['optimizer'] = optimizer.state_dict()
    if optimizer.grad_scaler:
        state_dict["grad_scaler"] = optimizer.grad_scaler.state_dict()
    # state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups

    cpu_fp32_from_fp16_groups = [
        [param.cpu() for param in group] for group in optimizer.fp32_from_float16_groups
    ]
    if optimizer.master_params_created:
        register_optimizer_hooks(model)
    if partial:
        state_dict["optimizer_state_dict"] = optimizer.local_state_dict()
        if args.shard_optimizer_state:
            if smp.rdp_rank() == 0:
                print("With shard_optimizer_state=True, gather full fp32_from_fp16_groups for the rdp_group on rdp rank 0")
                gathered_cpu_fp32_from_fp16_groups = [cpu_fp32_from_fp16_groups]
                for src in range(1, smp.rdp_size()):
                    gathered_cpu_fp32_from_fp16_groups.append(smp.recv_from(src, smp.RankType.RDP_RANK))
                state_dict["fp32_from_fp16"] = gathered_cpu_fp32_from_fp16_groups
            else:
                smp.send(cpu_fp32_from_fp16_groups, 0, smp.RankType.RDP_RANK)
                state_dict["fp32_from_fp16"] = cpu_fp32_from_fp16_groups
        else:
            state_dict["fp32_from_fp16"] = cpu_fp32_from_fp16_groups
        if smp.pp_size() > 1:
            print(
                "WARNING: Ensure that partition decision doesnt change between runs (you can ensure this by setting use_times=False in smp config)."
                "If you want to save and load with partition decision changing between runs, use full save and load instead."
            )
    else:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
        if smp.tp_size() > 1 and not args.shard_optimizer_state:
            tp_merged_fp32_from_fp16_groups, param_name_groups = get_tp_merged_fp32_from_fp16_param_groups(
                optimizer, cpu_fp32_from_fp16_groups
            )
            pp_merged_fp32_from_fp16_groups, param_name_groups = get_pp_merged_fp32_from_fp16_param_groups(
                optimizer, tp_merged_fp32_from_fp16_groups, param_name_groups
            )
        else:
            raise ValueError(
                "Loading full optimizer state is not supported, when TP is not enabled or shard_optimizer_state is enabled"
            )
        state_dict["fp32_from_fp16"] = pp_merged_fp32_from_fp16_groups
        state_dict["param_name_groups"] = param_name_groups
    return state_dict


def load_fp16_optimizer(args, model, optimizer, state_dict, partial=True):
    opt_state_dict = state_dict["optimizer"]

    if optimizer.master_params_created:
        register_optimizer_hooks(model)

    def hook_fn(model, optimizer):
        optimizer.load_state_dict(opt_state_dict["optimizer_state_dict"])
        if partial:
            if args.shard_optimizer_state:
                assert isinstance(opt_state_dict["fp32_from_fp16"], list), "Loading with shard_optimizer_state=True must use the checkpoint that was trained with shard_optimizer_state=True!"
                optimizer.fp32_from_fp16 = opt_state_dict["fp32_from_fp16"][smp.rdp_rank()]
            else:
                optimizer.fp32_from_fp16 = opt_state_dict["fp32_from_fp16"]

            for current_group, saved_group in zip(
                optimizer.fp32_from_float16_groups, optimizer.fp32_from_fp16
            ):
                for current, saved in zip(current_group, saved_group):
                    current.data.copy_(saved.data)

        else:
            optimizer.fp32_from_fp16 = opt_state_dict["fp32_from_fp16"]
            param_name_groups = opt_state_dict["param_name_groups"]
            param_id_to_index = optimizer._param_id_to_index()
            param_index_to_name_tp_group = smp_state.param_index_to_name_tp_group
            param_index_to_name = param_index_to_name_tp_group[smp.tp_rank()]
            for group_idx, (current_group, saved_group) in enumerate(
                zip(optimizer.fp32_from_float16_groups, optimizer.fp32_from_fp16)
            ):
                for current in current_group:
                    param_id = id(current)
                    param_index = param_id_to_index[param_id]
                    param_name = param_index_to_name[param_index]
                    arr_index = param_name_groups[group_idx][param_name]
                    saved = saved_group[arr_index]
                    if optimizer.master_distribution_axis[param_id] is not None:
                        axis = optimizer.master_distribution_axis[param_id]
                        slice_size = saved.size(axis) // smp.tp_size()
                        saved = torch.narrow(
                            saved.data, axis, slice_size * smp.tp_rank(), slice_size
                        ).contiguous()
                    else:
                        saved = saved.data
                    current.data.copy_(saved)

        optimizer.grad_scaler.load_state_dict(opt_state_dict["grad_scaler"])

    model.register_post_partition_hook(hook_fn)


class MegatronOptimizer(ABC):
    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad):
        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, "no optimizer is provided."
        # Set gradient clipping and logging params.
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.params_have_main_grad = params_have_main_grad

    def get_parameters(self):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                params.append(param)
        return params

    def clip_grad_norm(self, clip_grad):
        params = self.get_parameters()
        return clip_grad_norm_fp32(params, self.master_is_distributed, clip_grad)

    def count_zeros(self):
        params = self.get_parameters()
        return count_zeros_fp32(params)

    @abstractmethod
    def zero_grad(self, set_grads_to_None=True):
        pass

    @abstractmethod
    def get_loss_scale(self):
        """The output should be a cuda tensor of size 1."""

    def scale_loss(self, loss):
        """Simple scaling."""
        return self.get_loss_scale() * loss

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reload_model_params(self):
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated."""

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)


class FP16_Module(nn.Module):
    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.add_module("module", module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


class Float16OptimizerWithFloat16Params(MegatronOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a contihuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
    """

    def __init__(
        self,
        model,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        bf16,
        grad_scaler,
        use_smp=False,
        shard_optimizer_state=False,
        verbose=False,
    ):

        super(Float16OptimizerWithFloat16Params, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad
        )

        self.model = model
        self.verbose = verbose
        self.use_smp = use_smp
        self.shard_optimizer_state = shard_optimizer_state
        self.bf16 = bf16
        self.grad_scaler = grad_scaler

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert self.bf16, "fp16 expects a grad scaler."

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []
        self.master_is_distributed = {}
        self.fp32_from_fp16_paramid_groups = []
        self.master_distribution_axis = {}
        self.master_params_created = False
        self.warned_set_grads_to_none = False
        if not self.use_smp:
            self.init_master_params()

    def measure_additional_mem(f):
        def wrapper(*args, **kwargs):
            mem_before = torch.cuda.memory_allocated(device=smp.local_rank())
            f(*args, **kwargs)
            import gc

            gc.collect()
            gc.collect()
            gc.collect()
            mem_after = torch.cuda.memory_allocated(device=smp.local_rank())
            print(
                f"rank is {smp.local_rank()}, function name is {f.__name__}, memory usage is {humanize.naturalsize(mem_after - mem_before)}"
            )

        return wrapper

    def init_master_params(self):
        if self.use_smp:
            torch.cuda.set_device(smp.local_rank())
            register_optimizer_hooks(self.model)
        self.fp32paramid_from_fp16paramid = {}
        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.cuda.FloatTensor([1.0])

        # ======================
        # main parameter stuff
        # ======================

        # only need to create contiguous buffer for fp16 params which require grads
        contig_buffer_size = 0
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.requires_grad and param.type() in [
                    "torch.cuda.HalfTensor",
                    "torch.cuda.BFloat16Tensor",
                ]:
                    contig_buffer_size += param.numel()

        self.fp32_param_buffer = torch.empty(
            contig_buffer_size,
            device=torch.device("cuda", smp.local_rank()),
            dtype=torch.float32,
            requires_grad=True,
        )
        offset = 0

        # only need to create contiguous buffer for fp16 params which require grads
        contig_buffer_size = 0
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.requires_grad and param.type() in [
                    "torch.cuda.HalfTensor",
                    "torch.cuda.BFloat16Tensor",
                ]:
                    contig_buffer_size += param.numel()

        self.fp32_param_buffer = torch.empty(
            contig_buffer_size,
            device=torch.device("cuda", smp.local_rank()),
            dtype=torch.float32,
            requires_grad=True,
        )
        offset = 0

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            fp32_from_fp16_paramids_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group["params"]):
                if param.requires_grad:
                    # float16 params:
                    if param.type() in ["torch.cuda.HalfTensor", "torch.cuda.BFloat16Tensor"]:
                        float16_params_this_group.append(param)
                        # Create a copy
                        with torch.no_grad():
                            master_param_buffer = self.fp32_param_buffer.narrow(
                                0, offset, param.numel()
                            ).view_as(param)
                            master_param_buffer.copy_(param.float())
                            offset += param.numel()

                        main_param = nn.Parameter(
                            master_param_buffer, requires_grad=param.requires_grad
                        )
                        self.master_is_distributed[
                            main_param
                        ] = self.model.is_distributed_parameter(param)
                        self.master_distribution_axis[id(main_param)] = get_distribution_axis(param)
                        fp32_from_fp16_paramids_this_group.append(id(main_param))
                        if hasattr(param, "shared"):
                            main_param.shared = param.shared

                        # Replace the optimizer params with the new fp32 copy.
                        param_group["params"][i] = main_param
                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] = self.optimizer.state.pop(param)
                        self.fp32paramid_from_fp16paramid[id(param)] = id(main_param)

                    # fp32 params.
                    elif param.type() == "torch.cuda.FloatTensor":
                        fp32_params_this_group.append(param)
                        param_group["params"][i] = param

                    else:
                        raise TypeError(
                            "Wrapped parameters must be one of "
                            "torch.cuda.FloatTensor,  "
                            "torch.cuda.HalfTensor, or "
                            "torch.cuda.BFloat16Tensor. "
                            "Received {}".format(param.type())
                        )

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp16_paramid_groups.append(fp32_from_fp16_paramids_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        self.master_params_created = True

    def maybe_print(self, msg):
        if self.verbose:
            print(msg)

    def zero_grad(self, set_grads_to_None=True):
        """We only need to zero the model related parameters, i.e.,
                float16_groups & fp32_from_fp32_groups."""

        if self.shard_optimizer_state and set_grads_to_None and not self.warned_set_grads_to_none:
            print("WARNING: Will not set fp16 gradients to None since shard_optimizer_state is enabled.")
            self.warned_set_grads_to_none = True

        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_grads_to_None and not self.shard_optimizer_state)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_grads_to_None)
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if model_param.grad is not None:
                    # If gradient_as_bucket_view is True for DistributedModel, the grads will be in FP32
                    # thus below line wont create a copy of grads
                    # Otherwise below line will create a copy of grads
                    main_param.grad = model_param.grad.float()

    def _unscale_main_grads_and_check_for_nan(self):
        main_grads = []
        # fp32 params fromm float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        if hasattr(torch, "_amp_foreach_non_finite_check_and_unscale_"):
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, self.found_inf, self.grad_scaler.inv_scale
            )
        else:
            if self.grad_scaler.inv_scale != 1.0:
                grads = [main_grad for main_grad in main_grads if main_grad is not None]
                _overflow_buf = torch.cuda.IntTensor([0])
                multi_tensor_applier(
                    amp_C.multi_tensor_scale,
                    _overflow_buf,
                    [grads, grads],
                    self.grad_scaler.inv_scale,
                )
                self.found_inf[0] = _overflow_buf[0]

        # Update across all model parallel instances.
        """
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_model_parallel_group())
        """
        torch.distributed.all_reduce(
            self.found_inf, op=torch.distributed.ReduceOp.MAX, group=smp.get_mp_process_group()
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0
        return found_inf_flag

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
        )

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        )

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    @torch.no_grad()
    def step(self):

        # Copy gradients from model params to main params.
        self._copy_model_grads_to_main_grads()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                print("Found, inf, skipping step")
                return False, None, None

        # Clip the main gradients.
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if self.log_num_zeros_in_grad else None

        # Step the optimizer.
        self.optimizer.step()

        # Update params from main params.
        self._copy_main_params_to_model_params()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def state_dict(self):
        if not self.use_smp:
            state_dict = {}
            state_dict["optimizer"] = self.optimizer.state_dict()
            if self.grad_scaler:
                state_dict["grad_scaler"] = self.grad_scaler.state_dict()
            state_dict["fp32_from_fp16_params"] = self.fp32_from_float16_groups
            return state_dict
        else:
            return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if not self.use_smp:
            # Optimizer.
            optimizer_key = "optimizer"
            if optimizer_key not in state_dict:
                optimizer_key = "optimizer_state_dict"
                print("***WARNING*** loading optimizer from " "an old checkpoint ...")
            self.optimizer.load_state_dict(state_dict[optimizer_key])

            # Grad scaler.
            if "grad_scaler" not in state_dict:
                print("***WARNING*** found an old checkpoint, will not " "load grad scaler ...")
            else:
                if self.grad_scaler:
                    self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
                else:
                    print(
                        "***WARNING*** fould the grad scaler in the "
                        "checkpoint but it is None in the class. "
                        "Skipping loading grad scaler ..."
                    )

            # Copy data for the main params.
            fp32_from_float16_params_key = "fp32_from_fp16_params"
            if fp32_from_float16_params_key not in state_dict:
                fp32_from_float16_params_key = "fp32_from_fp16"
            for current_group, saved_group in zip(
                self.fp32_from_float16_groups, state_dict[fp32_from_float16_params_key]
            ):
                for current_param, saved_param in zip(current_group, saved_group):
                    current_param.data.copy_(saved_param.data)
        else:
            self.optimizer.load_state_dict(state_dict)
