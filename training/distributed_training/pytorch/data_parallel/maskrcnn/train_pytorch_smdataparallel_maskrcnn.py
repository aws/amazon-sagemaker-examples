# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import logging
import functools
import random
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.tester import test

# Import SMDataParallel modules for PyTorch.
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
import smdistributed.dataparallel.torch.distributed as dist
dist.init_process_group()

# SMDataParallel: Initialize
if not dist.is_initialized():
    dist.init_process_group()


# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
    use_amp = True
except ImportError:
    print('Use APEX for multi-precision via apex.amp')
    use_amp = False
# try:
#     from apex.parallel import DistributedDataParallel as DDP
#     use_apex_ddp = True
# except ImportError:
#     print('Use APEX for better performance')
use_apex_ddp = False

def test_and_exchange_map(tester, model, distributed):
    results = tester(model=model, distributed=distributed)

    # main process only
    #if is_main_process():
    if dist.get_rank() ==0:
        # Note: one indirection due to possibility of multiple test datasets, we only care about the first
        #       tester returns (parsed results, raw results). In our case, don't care about the latter
        map_results, raw_results = results[0]
        bbox_map = map_results.results["bbox"]['AP']
        segm_map = map_results.results["segm"]['AP']
    else:
        bbox_map = 0.
        segm_map = 0.

    if distributed:
        map_tensor = torch.tensor([bbox_map, segm_map], dtype=torch.float32, device=torch.device("cuda"))
        torch.distributed.broadcast(map_tensor, 0)
        bbox_map = map_tensor[0].item()
        segm_map = map_tensor[1].item()

    return bbox_map, segm_map

def mlperf_test_early_exit(iteration, iters_per_epoch, tester, model, distributed, min_bbox_map, min_segm_map):
    if iteration > 0 and iteration % iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch

        logger = logging.getLogger('maskrcnn_benchmark.trainer')
        logger.info("Starting evaluation...")

        bbox_map, segm_map = test_and_exchange_map(tester, model, distributed)

        # necessary for correctness
        model.train()
        logger.info('bbox mAP: {}, segm mAP: {}'.format(bbox_map, segm_map))

        # terminating condition
        if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
            logger.info("Target mAP reached, exiting...")
            return True

    return False


def train(cfg, args):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if use_amp:
        # Initialize mixed-precision training
        use_mixed_precision = cfg.DTYPE == "float16"

        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if args.distributed:
        # if use_apex_ddp:
        #     model = DDP(model, delay_allreduce=True)
        # else:
        # SMDataParallel: Wrap the PyTorch model with SMDataParallel’s DDP
        model = DDP(model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)
        #model = DDP(model)
    print("model parameter size: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    # SMDataParallel: Save model on master node.
    save_to_disk = dist.get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader, iters_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=args.distributed,
        start_iter=arguments["iteration"],
        data_dir = args.data_dir
    )
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # set the callback function to evaluate and potentially
    # early exit each epoch
    if cfg.PER_EPOCH_EVAL:
        per_iter_callback_fn = functools.partial(
            mlperf_test_early_exit,
            iters_per_epoch=iters_per_epoch,
            tester=functools.partial(test, cfg=cfg),
            model=model,
            distributed=args.distributed,
            min_bbox_map=cfg.MIN_BBOX_MAP,
            min_segm_map=cfg.MIN_MASK_MAP)
    else:
        per_iter_callback_fn = None
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        use_amp,
        cfg,
        per_iter_end_callback_fn=per_iter_callback_fn,
    )

    return model


def test_model(cfg, model, args):
    if args.distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=args.distributed,
                                        data_dir = args.data_dir)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        #synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=dist.get_local_rank())
    parser.add_argument(
        "--seed",
        help="manually set random seed for torch",
        type=int,
        default=99
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--bucket-cap-mb",
        dest="bucket_cap_mb",
        help="specify bucket size for SMDataParallel",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        help="Absolute path of dataset ",
        type=str,
        default=None
    )
    parser.add_argument(
        "--dtype",
        dest="dtype"
    )


    args = parser.parse_args()
    keys = list(os.environ.keys())
    args.data_dir = os.environ['SM_CHANNEL_TRAIN'] if 'SM_CHANNEL_TRAIN' in keys else args.data_dir
    print("dataset dir: ", args.data_dir)


    # Set seed to reduce randomness
    random.seed(args.seed + dist.get_local_rank())
    np.random.seed(args.seed + dist.get_local_rank())
    torch.manual_seed(args.seed + dist.get_local_rank())
    torch.cuda.manual_seed(args.seed + dist.get_local_rank())

    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_gpus = dist.get_world_size()
    args.distributed = num_gpus > 1

    if args.distributed:
        # SMDataParallel: Pin each GPU to a single SMDataParallel process. 
        torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )
        #synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DTYPE=args.dtype
    cfg.freeze()
    print ("CONFIG")
    print (cfg)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, dist.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        if not cfg.PER_EPOCH_EVAL:
            test_model(cfg, model, args)


if __name__ == "__main__":
    main()
