# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# To use new EIA inference API, customer should use attach_eia(model, eia_ordinal_number)
VERSIONS_USE_NEW_API = ["1.5.1"]


def predict_fn(input_data, model):
    logger.info(
        "Performing EIA inference with Torch JIT context with input of size {}".format(
            input_data.shape
        )
    )
    # With EI, client instance should be CPU for cost-efficiency. Subgraphs with unsupported arguments run locally. Server runs with CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    # Please make sure model is loaded to cpu and has been eval(), in this example, we have done this step in model_fn()
    with torch.no_grad():
        if torch.__version__ in VERSIONS_USE_NEW_API:
            # Please make sure torcheia has been imported
            import torcheia

            # We need to set the profiling executor for EIA
            torch._C._jit_set_profiling_executor(False)
            with torch.jit.optimized_execution(True):
                return model.forward(input_data)
        # Set the target device to the accelerator ordinal
        else:
            with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
                return model(input_data)


def model_fn(model_dir):
    try:
        loaded_model = torch.jit.load("model.pth", map_location=torch.device("cpu"))
        if torch.__version__ in VERSIONS_USE_NEW_API:
            import torcheia

            loaded_model = loaded_model.eval()
            loaded_model = torcheia.jit.attach_eia(loaded_model, 0)
        return loaded_model
    except Exception as e:
        logger.exception(f"Exception in model fn {e}")
        return None
