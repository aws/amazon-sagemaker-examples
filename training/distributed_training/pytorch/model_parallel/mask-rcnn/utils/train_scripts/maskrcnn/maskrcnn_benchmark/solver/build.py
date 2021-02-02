# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
import apex

from .lr_scheduler import WarmupMultiStepLR
from .cosine_lr_scheduler import CosineAnnealingWarmUpRestarts

from .fused_sgd import FusedSGD
from smdistributed.modelparallel.torch.optimizers import FusedNovoGrad
#from .fused_novograd import FusedNovoGrad

def make_optimizer(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    bias_params = []
    bias_lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    bias_weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            bias_params.append(value)
        else:
            params.append(value)

    is_fp16 = (cfg.DTYPE == "float16")
    if is_fp16: # with FP16_Optimizer wrapper
        if cfg.SOLVER.OPTIMIZER == "NovoGrad":
            optimizer = FusedNovoGrad(
                [
                    {"params": params, "lr": lr, "weight_decay": weight_decay},
                    {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
                ],
                lr, betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2), eps=1e-7, grad_averaging=False, init_zero=False, reg_inside_moment=True, bias_correction=True)
        elif cfg.SOLVER.OPTIMIZER == "SGD":
            optimizer = FusedSGD(
                [
                    {"params": params, "lr": lr, "weight_decay": weight_decay},
                    {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
                ],
                lr, momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise NotImplementedError
    else: # without FP16_Optimizer wrapper
        optimizer = apex.optimizers.FusedSGD(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, momentum=cfg.SOLVER.MOMENTUM)

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_SCHEDULE == "COSINE":
        return CosineAnnealingWarmUpRestarts(
            optimizer, # Novograd
            T_0 = cfg.SOLVER.MAX_ITER, # total steps solver.max_iter
            eta_max = cfg.SOLVER.BASE_LR, # max lr or base lr init_lr
            alpha = 0.001,
            T_up = cfg.SOLVER.WARMUP_ITERS, # warmup steps  , warmupsteps
        )
    elif cfg.SOLVER.LR_SCHEDULE == "MULTISTEP":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )

