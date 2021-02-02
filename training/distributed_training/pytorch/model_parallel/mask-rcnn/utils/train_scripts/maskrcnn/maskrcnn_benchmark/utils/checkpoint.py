# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import logging
import os
import copy

import torch

from maskrcnn_benchmark.utils.model_serialization import load_state_dict, is_layer_nhwc_eligible
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url
from maskrcnn_benchmark.fp16.fp16 import FP16_Optimizer

from collections import defaultdict
from itertools import chain

import smdistributed.modelparallel.torch as smp

class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def _get_optimizer_state_dict(self, save_partial):
        optimizer_state_dict = {}
        loss_scaler = self.optimizer.loss_scaler
        _model = loss_scaler.model
        loss_scaler.model = None
        _loss_scaler = copy.deepcopy(loss_scaler)
        loss_scaler.model = _model
        optimizer_state_dict['loss_scaler'] = _loss_scaler
        optimizer_state_dict['dynamic_loss_scale'] = self.optimizer.dynamic_loss_scale
        optimizer_state_dict['overflow'] = self.optimizer.overflow
        optimizer_state_dict['first_closure_call_this_step'] = self.optimizer.first_closure_call_this_step
        cpu_fp32_from_fp16_groups = [[param.cpu() for param in group] for group in self.optimizer.fp32_from_fp16_groups]
        if save_partial:
            optimizer_state_dict['optimizer_state_dict'] = self.optimizer.local_state_dict()
            optimizer_state_dict['fp32_from_fp16'] = cpu_fp32_from_fp16_groups
        else:
            optimizer_state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            all_fp32_from_fp16_groups = smp.allgather(cpu_fp32_from_fp16_groups, smp.MP_GROUP)
            merged_fp32_from_fp16_groups = []
            for i in range(len(all_fp32_from_fp16_groups[0])):
                merged = set()
                for group in all_fp32_from_fp16_groups:
                    cur_g = set(group[i])
                    common = merged.intersection(cur_g)
                    if len(common) > 0:
                        print(f"[SMP] params {[id(t) for t in common]} are shared across devices")
                    merged = merged.union(cur_g)
                merged_fp32_from_fp16_groups.append(list(merged))
            optimizer_state_dict['fp32_from_fp16'] =  merged_fp32_from_fp16_groups

        return optimizer_state_dict

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        if smp.dp_rank() == 0:
            save_partial = "save_partial" in kwargs and kwargs["save_partial"]
            nhwc = kwargs.get("nhwc", False)
            data = {}
            if save_partial:
                data["model"] = self.model.local_state_dict()
            else:
                data["model"] = self.model.state_dict()

            if self.optimizer is not None:
                data["optimizer"] = self._get_optimizer_state_dict(save_partial)

            if self.scheduler is not None:
                data["scheduler"] = self.scheduler.state_dict()
            data.update(kwargs)
            # transpose to NCHW before saving as checkpoint if NHWC is used
            if nhwc:
                transpose_checkpoint_model_state_nhwc_to_nchw(data["model"])
                transpose_optimizer_state_nhwc_to_nchw(self.model, self.optimizer, data["optimizer"]["optimizer_state_dict"], partial_dict=save_partial)
            save_file = os.path.join(self.save_dir, "{}.pth".format(name))
            self.logger.info("Saving checkpoint to {}".format(save_file))
            smp.save(data, save_file, save_partial)
            self.tag_last_checkpoint(save_file)
            # Convert back to NHWC if NHWC layout is used, needed for optimizer buffers
            if nhwc:
                if self.optimizer is not None:
                    transpose_optimizer_state_nchw_to_nhwc(self.model, self.optimizer, partial_dict=save_partial)

    def _load_fp16_optimizer(self, opt_state_dict):
        def param_name_to_index(self):
            param_id_to_index = self._param_id_to_index()
            name_to_index = {}
            for name, param in self.model.named_parameters():
                fp16_param_id = id(param)
                if fp16_param_id in self.fp32paramid_from_fp16paramid:
                    param_id = self.fp32paramid_from_fp16paramid[fp16_param_id]
                else:
                    param_id = fp16_param_id
                if param_id in param_id_to_index:
                    name_to_index[name] = param_id_to_index[param_id]
            return name_to_index

        def hook_fn(model, optimizer):
            from functools import partial 
            optimizer.param_name_to_index = partial(param_name_to_index, optimizer)
            optimizer.load_state_dict(opt_state_dict['optimizer_state_dict'])
            transpose_optimizer_state_nchw_to_nhwc(model, optimizer, partial_dict=True)
            
            optimizer.fp32_from_fp16 = opt_state_dict['fp32_from_fp16']
            optimizer.loss_scaler = opt_state_dict['loss_scaler']
            optimizer.loss_scaler.model = model
            optimizer.dynamic_loss_scale = opt_state_dict['dynamic_loss_scale']
            optimizer.overflow = opt_state_dict['overflow']
            optimizer.first_closure_call_this_step = opt_state_dict['first_closure_call_this_step']
            
            for current_group, saved_group in zip(
                    optimizer.fp32_from_fp16_groups, optimizer.fp32_from_fp16):
                for current, saved in zip(current_group, saved_group):
                    current.data.copy_(saved.data)

        self.model.register_post_partition_hook(hook_fn)


    def load(self, f=None, nhwc=False, load_partial=False, model_only=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        else:
            load_partial=False
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f, load_partial)
        self._load_model(checkpoint, nhwc)

        def init_params(mod, opt):
            opt.init_master_params()

        self.model.register_post_partition_hook(init_params)

        if model_only:
            return checkpoint

        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            if isinstance(self.optimizer, FP16_Optimizer):
                if not load_partial:
                    raise ValueError("[SMP]FP16_Optimizer does not support load full checkpoint!")
                self._load_fp16_optimizer(checkpoint.pop("optimizer"))
            else:
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
                if nhwc:
                    transpose_optimizer_state_nchw_to_nhwc(self.model, self.optimizer, partial_dict=False)
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f, load_partial):
        return smp.load(f, partial=load_partial)

    def _load_model(self, checkpoint, nhwc):
        load_state_dict(self.model, checkpoint.pop("model"), nhwc)


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f, load_partial):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f, load_partial)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


def transpose_checkpoint_model_state_nhwc_to_nchw(model_dict):
    for k in model_dict:
        param_tensor = model_dict[k]
        needs_transpose = is_layer_nhwc_eligible(k) and len(param_tensor.shape)==4
        if needs_transpose:
            model_dict[k] = model_dict[k].permute(0,3,1,2).contiguous()

def transpose_optimizer_state_nhwc_to_nchw(model, optimizer, optimizer_dict, partial_dict=True):
    param_id_to_index = optimizer._param_id_to_index()
    layer_id_to_name_map = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            if id(param) in optimizer.id_trans:
                idx = param_id_to_index[optimizer.id_trans[id(param)]]
            else:
                idx = param_id_to_index[id(param)]
            layer_id_to_name_map[idx] = name

    if not partial_dict:
        # Get all params ids when save full
        all_layer_id_to_name_map = smp.allgather(layer_id_to_name_map, smp.MP_GROUP)
        for item in all_layer_id_to_name_map:
            layer_id_to_name_map.update(item)

    for name, param in model.named_parameters():
        layer_id_to_name_map[id(param)] = name
    for k in optimizer_dict['state']:
        needs_transpose = is_layer_nhwc_eligible(layer_id_to_name_map[k])
        needs_transpose = needs_transpose and  \
                          len(optimizer_dict['state'][k]['exp_avg'].shape) == 4
        if needs_transpose:    
            optimizer_dict['state'][k]['exp_avg'] =  \
                        optimizer_dict['state'][k]['exp_avg'].permute(0,3,1,2).contiguous()

def transpose_optimizer_state_nchw_to_nhwc(model, optimizer, partial_dict=True):
    if partial_dict:
        optimizer_dict = optimizer.local_state_dict()
    else:
        optimizer_dict = optimizer.state_dict()

    param_id_to_index = optimizer._param_id_to_index()
    layer_id_to_name_map = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            if id(param) in optimizer.id_trans:
                idx = param_id_to_index[optimizer.id_trans[id(param)]]
            else:
                idx = param_id_to_index[id(param)]
            layer_id_to_name_map[idx] = name

    if not partial_dict:
        # Get all params ids when save full
        all_layer_id_to_name_map = smp.allgather(layer_id_to_name_map, smp.MP_GROUP)
        for item in all_layer_id_to_name_map:
            layer_id_to_name_map.update(item)

    for name, param in model.named_parameters():
        layer_id_to_name_map[id(param)] = name
    for k in optimizer_dict['state']:
        needs_transpose = is_layer_nhwc_eligible(layer_id_to_name_map[k])
        needs_transpose = needs_transpose and  \
                          len(optimizer_dict['state'][k]['exp_avg'].shape) == 4
        if needs_transpose:    
            optimizer_dict['state'][k]['exp_avg'] =  \
                        optimizer_dict['state'][k]['exp_avg'].permute(0,2,3,1).contiguous()
