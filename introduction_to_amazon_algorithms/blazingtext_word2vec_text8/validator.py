# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
#    http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from __future__ import print_function


def get_int(param_value, param_name):
    try:
        result = int(param_value)
    except ValueError as e:
        raise Exception("Parameter {} expects integer input.".format(param_name))
    return result


def get_float(param_value, param_name):
    try:
        result = float(param_value)
    except ValueError as e:
        raise Exception("Parameter {} expects float input.".format(param_name))
    return result


def validate_hyperparameters(cfg):
    warnings = 0

    if "mode" in cfg:
        tmp = cfg["mode"]
        if tmp not in ["skipgram", "cbow", "batch_skipgram"]:
            raise Exception(
                "mode should be one of [\"skipgram\", \"cbow\", \"batch_skipgram\"]")

    if "min_count" in cfg:
        tmp = get_int(cfg["min_count"], "min_count")
        if tmp < 0:
            raise Exception(
                "Parameter 'min_count' should be >= 0.")

    if "sampling_threshold" in cfg:
        tmp = get_float(cfg["sampling_threshold"], "sampling_threshold")
        if tmp <= 0 or tmp >= 1:
            raise Exception(
                "Parameter 'sampling_threshold' should be between (0,1)")

    if "learning_rate" in cfg:  # Default: .05
        tmp = get_float(cfg["learning_rate"], "learning_rate")
        if tmp <= 0:
            raise Exception(
                "Parameter 'learning_rate' should be > 0.")

    ws = 5
    if "window_size" in cfg:  # Default: 5
        ws = get_int(cfg["window_size"], "window_size")
        if ws <= 0:
            raise Exception(
                "Parameter 'window_size' should be > 0.")

    if "vector_dim" in cfg:  # Default: 100
        tmp = get_int(cfg["vector_dim"], "vector_dim")
        if tmp <= 0:
            raise Exception(
                "Parameter 'vector_dim' should be > 0.")
        if tmp > 2048:
            raise Exception(
                "Parameter 'vector_dim' should be <= 2048.")
        if tmp >= 1500:
            warnings += 1
            print("You are using a big vector dimension. Training might take a long time or might fail due to memory "
                  "issues.")

    if "epochs" in cfg:  # Default: 5
        tmp = get_int(cfg["epochs"], "epochs")
        if tmp <= 0:
            raise Exception(
                "Parameter 'epochs' should be > 0.")

    if "negative_samples" in cfg:  # Default: 5
        tmp = get_int(cfg["negative_samples"], "negative_samples")
        if tmp <= 0:
            raise Exception(
                "Parameter 'negative_samples' should be > 0.")

    if "batch_size" in cfg:  # Default: 11
        tmp = get_int(cfg["batch_size"], "batch_size")
        if tmp <= 0:
            raise Exception(
                "Parameter 'batch_size' should be > 0.")
        if tmp > 32:
            raise Exception(
                "Parameter 'batch_size' should be <= 32.")
        reco = 2 * ws + 1
        if tmp is not reco:
            warnings += 1
            print(
                "It is recommended that you set batch_size as 2*window_size + 1 which is %s in this case." % str(reco))
    return warnings


def validate_params(resource_config, hyperparameters):
    count = resource_config["InstanceCount"]
    instance = resource_config["InstanceType"]
    volume_size = resource_config["VolumeSizeInGB"]

    mode = hyperparameters.get("mode", None)

    if not mode:
        raise Exception("Please provide a mode in hyperparameters. It should be one of "
                        "[\"skipgram\", \"cbow\", \"batch_skipgram\"]")

    warnings = validate_hyperparameters(hyperparameters)

    if instance.startswith("ml.p"):
        if count > 1:
            raise Exception("Please use a single GPU instance or change to multiple CPU instances.")

        if mode == "batch_skipgram":
            raise Exception("batch_skipgram is not supported on GPU. Please select a CPU instance or use cbow/skipgram")

    else:
        if count > 1:
            if mode != "batch_skipgram":
                raise Exception("Only batch_skipgram is available when training on multiple CPU instances")

    if volume_size <= 1:
        raise Exception("Volume size <= 1 GB might not be sufficient for training. Please use a larger volume size")

    if not warnings:
        print("The configuration looks fine!")
    else:
        print("The configuration looks fine except some warnings that may or may not result in failure of the job!")
