# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval_np import infer_coco_eval
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.async_evaluator import init, get_evaluator, set_epoch_tag, get_tag

from collections import deque
import threading

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

def has_checkpoint(checkpt_dir):
    saved_file = os.path.join(checkpt_dir, "last_checkpoint")
    return os.path.exists(saved_file)

def get_checkpoint_file(checkpt_dir):
    saved_file = os.path.join(checkpt_dir, "last_checkpoint")
    try:
        with open(saved_file, "r") as f:
            last_saved = f.read()
            last_saved = last_saved.strip()
    except IOError:
        # if file doesn't exist
        last_saved = ""
    return last_saved

def get_latest_checkpoint(q, checkpt_dir):
    encountered_checkpoints = set()
    while True:
        if has_checkpoint(checkpt_dir):
            latest = get_checkpoint_file(checkpt_dir) 
            if (latest is not None) and (len(q) == 0 or q[-1] != latest) and (latest not in encountered_checkpoints):
                q.append(latest)
                encountered_checkpoints.add(latest)
            time.sleep(1)

def do_eval(cfg, model, distributed):
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    start_data_time = time.time()
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    end_data_time = time.time()
    total_data_time = end_data_time - start_data_time
    model.eval()

    start_test_time = time.time()
    results = []
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = infer_coco_eval(model, data_loader_val, cfg, pool_size=8)
        results.append(result)
    end_test_time = time.time()
    total_testing_time = end_test_time - start_test_time
    print("number of inference calls ",len(results))

    if is_main_process():
        # map_results, raw_results = results[0]
        bbox_map = results[0]["bbox"]
        segm_map = results[0]["segm"]
        print("BBOX_mAP: ", bbox_map, " MASK_mAP: ", segm_map)

    print("Data time: ", total_data_time)
    print("Inference time: ", total_testing_time)
    return results[0]



def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    
    model_build_start = time.time()
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    is_fp16 = (cfg.DTYPE == "float16")
    if is_fp16:
        # convert model to FP16
        model.half()

    input_q = deque()
    output_dir = cfg.OUTPUT_DIR

    checkpt_loader = threading.Thread(target=get_latest_checkpoint, args=(input_q, output_dir))
    checkpt_loader.start()
    last = None
#    output_dir = cfg.OUTPUT_DIR
#    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
#    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    done = False
    while not done:
        if len(input_q) == 0 or input_q[0] == last:
            pass
        if len(input_q) !=0 and input_q[0] != last:
            last = input_q[0]
            input_q.popleft()
            print("Running eval for", last)
            checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
            _ = checkpointer.load(cfg.MODEL.WEIGHT)
            eval_result = do_eval(cfg, model, distributed)
            done = ((eval_result['bbox'] >= .377) and (eval_result['segm'] >= .339))
            print(done)
            if is_main_process and done:
                done_file = os.path.join(cfg.OUTPUT_DIR, "done")
                with open(done_file, 'w') as stop_trigger:
                    stop_trigger.write('done')
        time.sleep(5)

if __name__ == "__main__":
    main()
