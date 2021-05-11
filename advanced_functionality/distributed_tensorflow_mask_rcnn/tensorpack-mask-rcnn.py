"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time


def copy_files(src, dest):
    src_files = os.listdir(src)
    for file in src_files:
        path = os.path.join(src, file)
        if os.path.isfile(path):
            shutil.copy(path, dest)


def train():
    model_dir = os.environ["SM_MODEL_DIR"]
    log_dir = None

    copy_logs_to_model_dir = False

    try:
        log_dir = os.environ["SM_CHANNEL_LOG"]
        copy_logs_to_model_dir = True
    except KeyError:
        log_dir = model_dir

    train_data_dir = os.environ["SM_CHANNEL_TRAIN"]

    hyperparamters = json.loads(os.environ["SM_HPS"])

    try:
        batch_norm = hyperparamters["batch_norm"]
    except KeyError:
        batch_norm = "FreezeBN"

    try:
        mode_fpn = hyperparamters["mode_fpn"]
    except KeyError:
        mode_fpn = "True"

    try:
        mode_mask = hyperparamters["mode_mask"]
    except KeyError:
        mode_mask = "True"

    try:
        eval_period = hyperparamters["eval_period"]
    except KeyError:
        eval_period = 1

    try:
        lr_schedule = hyperparamters["lr_schedule"]
    except KeyError:
        lr_schedule = "[240000, 320000, 360000]"

    try:
        horovod_cycle_time = hyperparamters["horovod_cycle_time"]
    except KeyError:
        horovod_cycle_time = 0.5

    try:
        horovod_fusion_threshold = hyperparamters["horovod_fusion_threshold"]
    except KeyError:
        horovod_fusion_threshold = 67108864

    try:
        data_train = hyperparamters["data_train"]
    except KeyError:
        data_train = "coco_train2017"

    try:
        data_val = hyperparamters["data_val"]
    except KeyError:
        data_val = "coco_val2017"

    try:
        images_per_epoch = hyperparamters["images_per_epoch"]
    except KeyError:
        images_per_epoch = 120000

    try:
        backbone_weights = hyperparamters["backbone_weights"]
    except KeyError:
        backbone_weights = "ImageNet-R50-AlignPadding.npz"

    try:
        resnet_arch = hyperparamters["resnet_arch"]
    except KeyError:
        resnet_arch = "resnet50"

    load_model = None
    try:
        load_model = hyperparamters["load_model"]
    except KeyError:
        pass

    resnet_num_blocks = "[3, 4, 6, 3]"
    if resnet_arch == "resnet101":
        resnet_num_blocks = "[3, 4, 23, 3]"

    gpus_per_host = int(os.environ["SM_NUM_GPUS"])
    all_hosts = json.loads(os.environ["SM_HOSTS"])
    numprocesses = len(all_hosts) * int(gpus_per_host)

    steps_per_epoch = int(images_per_epoch / numprocesses)

    _cmd = f"""/usr/bin/python3 /tensorpack/examples/FasterRCNN/train.py \
--logdir {log_dir} \
--config DATA.BASEDIR={train_data_dir} \
MODE_FPN={mode_fpn} \
MODE_MASK={mode_mask} \
BACKBONE.RESNET_NUM_BLOCKS='{resnet_num_blocks}' \
BACKBONE.WEIGHTS={train_data_dir}/pretrained-models/{backbone_weights} \
BACKBONE.NORM={batch_norm} \
DATA.TRAIN='["{data_train}"]' \
DATA.VAL='("{data_val}",)' \
TRAIN.STEPS_PER_EPOCH={steps_per_epoch} \
TRAIN.EVAL_PERIOD={eval_period} \
TRAIN.LR_SCHEDULE='{lr_schedule}' \
TRAINER=horovod"""

    for key, item in hyperparamters.items():
        if key.startswith("config:"):
            hp = f" {key[7:]}={item}"
            _cmd += hp

    if load_model:
        _cmd += f" --load {train_data_dir}/pretrained-models/{load_model}"

    exitcode = 0
    try:
        process = subprocess.Popen(
            _cmd,
            encoding="utf-8",
            cwd="/tensorpack",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        while True:
            if process.poll() != None:
                break

            output = process.stdout.readline()
            if output:
                print(output.strip())

        exitcode = process.poll()
        print(f"Exit code:{exitcode}")
        exitcode = 0
    except Exception as e:
        print("train exception occured", file=sys.stderr)
        exitcode = 1
        print(str(e), file=sys.stderr)
    finally:
        if copy_logs_to_model_dir:
            copy_files(log_dir, model_dir)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exitcode)


if __name__ == "__main__":
    train()
