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
TASK_TO_SEQ_LEN = {
    "stsb": 128,
    "mrpc": 128,
    "rte": 128,
    "sst2": 64,
    "qqp": 128,
    "qnli": 128,
    "cola": 64,
    "mnli": 128,
    "mnli-m": 128,
    "mnli-mm": 128,
}


TASKINFO = {
    "cola": {"metric": "matthews_correlation", "mode": "max"},
    "mnli": {"metric": "accuracy", "mode": "max"},
    "mrpc": {"metric": "f1", "mode": "max"},
    "qnli": {"metric": "accuracy", "mode": "max"},
    "qqp": {"metric": "f1", "mode": "max"},
    "rte": {"metric": "accuracy", "mode": "max"},
    "sst2": {"metric": "accuracy", "mode": "max"},
    "stsb": {"metric": "spearmanr", "mode": "max"},
    "wnli": {"metric": "accuracy", "mode": "max"},
}
