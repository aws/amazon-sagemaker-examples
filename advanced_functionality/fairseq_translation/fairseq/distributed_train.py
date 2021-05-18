#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import json
import os
import socket
import subprocess

import torch
from fairseq import distributed_utils, options
from multiprocessing_train import ErrorHandler
from train_driver import main as single_process_main


def run(args, error_queue):
    try:
        args.distributed_rank = distributed_utils.distributed_init(args)
        print(
            "| initialized host {} as rank {}".format(socket.gethostname(), args.distributed_rank)
        )
        single_process_main(args)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((args.distributed_rank, traceback.format_exc()))


def main(args):

    port = 1112
    with open("/opt/ml/input/config/resourceconfig.json", "r") as f:
        resource_config = json.load(f)
    hosts = resource_config["hosts"]
    current_host = resource_config["current_host"]

    num_gpus_per_node = torch.cuda.device_count()
    world_size = len(hosts)

    args.distributed_backend = "gloo"

    args.distributed_init_method = "tcp://{host}:{port}".format(host=hosts[0], port=port)

    args.distributed_world_size = world_size * num_gpus_per_node

    mp = torch.multiprocessing.get_context("spawn")

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(num_gpus_per_node):

        args.distributed_rank = hosts.index(current_host) * num_gpus_per_node + i
        args.device_id = i

        procs.append(
            mp.Process(
                target=run,
                args=(
                    args,
                    error_queue,
                ),
                daemon=True,
            )
        )
        procs[i].start()
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()
