#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import functools
import logging
import time
from abc import abstractmethod

hpo_log = logging.getLogger("hpo_log")


def create_workflow(hpo_config):
    """Workflow Factory [instantiate MLWorkflow based on config]"""
    if hpo_config.compute_type == "single-CPU":
        from workflows.MLWorkflowSingleCPU import MLWorkflowSingleCPU

        return MLWorkflowSingleCPU(hpo_config)

    if hpo_config.compute_type == "multi-CPU":
        from workflows.MLWorkflowMultiCPU import MLWorkflowMultiCPU

        return MLWorkflowMultiCPU(hpo_config)

    if hpo_config.compute_type == "single-GPU":
        from workflows.MLWorkflowSingleGPU import MLWorkflowSingleGPU

        return MLWorkflowSingleGPU(hpo_config)

    if hpo_config.compute_type == "multi-GPU":
        from workflows.MLWorkflowMultiGPU import MLWorkflowMultiGPU

        return MLWorkflowMultiGPU(hpo_config)


class MLWorkflow:
    @abstractmethod
    def ingest_data(self):
        pass

    @abstractmethod
    def handle_missing_data(self, dataset):
        pass

    @abstractmethod
    def split_dataset(self, dataset, i_fold):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, trained_model, X_test):
        pass

    @abstractmethod
    def score(self, y_test, predictions):
        pass

    @abstractmethod
    def save_trained_model(self, score, trained_model):
        pass

    @abstractmethod
    def cleanup(self, i_fold):
        pass

    @abstractmethod
    def emit_final_score(self):
        pass


def timer_decorator(target_function):
    @functools.wraps(target_function)
    def timed_execution_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = target_function(*args, **kwargs)
        exec_time = time.perf_counter() - start_time
        hpo_log.info(f" --- {target_function.__name__}" f" completed in {exec_time:.5f} s")
        return result

    return timed_execution_wrapper
