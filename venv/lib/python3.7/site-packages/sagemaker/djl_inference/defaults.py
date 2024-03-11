# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

STABLE_DIFFUSION_MODEL_TYPE = "stable-diffusion"

VALID_MODEL_CONFIG_FILES = ["config.json", "model_index.json"]

DEEPSPEED_RECOMMENDED_ARCHITECTURES = {
    "bloom",
    "opt",
    "gpt_neox",
    "gptj",
    "gpt_neo",
    "gpt2",
    "xlm-roberta",
    "roberta",
    "bert",
    STABLE_DIFFUSION_MODEL_TYPE,
}

FASTER_TRANSFORMER_RECOMMENDED_ARCHITECTURES = {
    "t5",
}

FASTER_TRANSFORMER_SUPPORTED_ARCHITECTURES = {
    "bert",
    "gpt2",
    "bloom",
    "opt",
    "gptj",
    "gpt_neox",
    "gpt_neo",
    "t5",
}

ALLOWED_INSTANCE_FAMILIES = {
    "ml.g4dn",
    "ml.g5",
    "ml.p3",
    "ml.p3dn",
    "ml.p4",
    "ml.p4d",
    "ml.p4de",
    "local_gpu",
}
