# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

GLUE_TASK_INFO = {
    "cola": {
        "metric": "matthews_correlation",
        "mode": "max",
        "seq_length": 64,
        "keys": ("sentence", None),
    },
    "mnli": {
        "metric": "accuracy",
        "mode": "max",
        "seq_length": 128,
        "keys": ("premise", "hypothesis"),
    },
    "mrpc": {"metric": "f1", "mode": "max", "seq_length": 128, "keys": ("sentence1", "sentence2")},
    "qnli": {
        "metric": "accuracy",
        "mode": "max",
        "seq_length": 128,
        "keys": ("question", "sentence"),
    },
    "qqp": {"metric": "f1", "mode": "max", "seq_length": 128, "keys": ("question1", "question2")},
    "rte": {
        "metric": "accuracy",
        "mode": "max",
        "seq_length": 128,
        "keys": ("sentence1", "sentence2"),
    },
    "sst2": {"metric": "accuracy", "mode": "max", "seq_length": 64, "keys": ("sentence", None)},
    "stsb": {
        "metric": "spearmanr",
        "mode": "max",
        "seq_length": 128,
        "keys": ("sentence1", "sentence2"),
    },
}
