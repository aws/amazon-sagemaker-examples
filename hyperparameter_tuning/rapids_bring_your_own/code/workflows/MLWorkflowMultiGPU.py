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


import logging
import os
import time
import warnings

import cupy
import dask
import dask_cudf
import joblib
import xgboost
from cuml.dask.common.utils import persist_across_workers
from cuml.dask.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from dask_ml.model_selection import train_test_split
from MLWorkflow import MLWorkflow, timer_decorator

hpo_log = logging.getLogger("hpo_log")
warnings.filterwarnings("ignore")


class MLWorkflowMultiGPU(MLWorkflow):
    """Multi-GPU Workflow"""

    def __init__(self, hpo_config):
        hpo_log.info("Multi-GPU Workflow")
        self.start_time = time.perf_counter()

        self.hpo_config = hpo_config
        self.dataset_cache = None

        self.cv_fold_scores = []
        self.best_score = -1

        self.cluster, self.client = self.cluster_initialize()

    @timer_decorator
    def cluster_initialize(self):
        """Initialize dask GPU cluster"""

        cluster = None
        client = None

        self.n_workers = cupy.cuda.runtime.getDeviceCount()

        cluster = LocalCUDACluster(n_workers=self.n_workers)
        client = Client(cluster)

        hpo_log.info(f"dask multi-GPU cluster with {self.n_workers} workers ")

        dask.config.set(
            {
                "temporary_directory": self.hpo_config.output_artifacts_directory,
                "logging": {"loggers": {"distributed.nanny": {"level": "CRITICAL"}}},  # noqa
            }
        )

        return cluster, client

    def ingest_data(self):
        """Ingest dataset, CSV and Parquet supported [ async/lazy ]"""

        if self.dataset_cache is not None:
            hpo_log.info("> skipping ingestion, using cache")
            return self.dataset_cache

        if "Parquet" in self.hpo_config.input_file_type:
            hpo_log.info("> parquet data ingestion")

            dataset = dask_cudf.read_parquet(
                self.hpo_config.target_files, columns=self.hpo_config.dataset_columns
            )

        elif "CSV" in self.hpo_config.input_file_type:
            hpo_log.info("> csv data ingestion")

            dataset = dask_cudf.read_csv(
                self.hpo_config.target_files, names=self.hpo_config.dataset_columns, header=0
            )

        hpo_log.info(f"\t dataset len: {len(dataset)}")
        self.dataset_cache = dataset
        return dataset

    def handle_missing_data(self, dataset):
        """Drop samples with missing data [ inplace ]"""
        dataset = dataset.dropna()
        return dataset

    @timer_decorator
    def split_dataset(self, dataset, random_state):
        """
        Split dataset into train and test data subsets,
        currently using CV-fold index for randomness.
        Plan to refactor with dask_ml KFold
        """
        hpo_log.info("> train-test split")
        label_column = self.hpo_config.label_column

        train, test = train_test_split(dataset, random_state=random_state)

        # build X [ features ], y [ labels ] for the train and test subsets
        y_train = train[label_column]
        X_train = train.drop(label_column, axis=1)
        y_test = test[label_column]
        X_test = test.drop(label_column, axis=1)

        # force execution
        X_train, y_train, X_test, y_test = persist_across_workers(
            self.client, [X_train, y_train, X_test, y_test], workers=self.client.has_what().keys()
        )

        # wait!
        wait([X_train, y_train, X_test, y_test])

        return (
            X_train.astype(self.hpo_config.dataset_dtype),
            X_test.astype(self.hpo_config.dataset_dtype),
            y_train.astype(self.hpo_config.dataset_dtype),
            y_test.astype(self.hpo_config.dataset_dtype),
        )

    @timer_decorator
    def fit(self, X_train, y_train):
        """Fit decision tree model"""
        if "XGBoost" in self.hpo_config.model_type:
            hpo_log.info("> fit xgboost model")
            dtrain = xgboost.dask.DaskDMatrix(self.client, X_train, y_train)
            num_boost_round = self.hpo_config.model_params["num_boost_round"]

            xgboost_output = xgboost.dask.train(
                self.client, self.hpo_config.model_params, dtrain, num_boost_round=num_boost_round
            )

            trained_model = xgboost_output["booster"]

        elif "RandomForest" in self.hpo_config.model_type:
            hpo_log.info("> fit randomforest model")
            trained_model = RandomForestClassifier(
                n_estimators=self.hpo_config.model_params["n_estimators"],
                max_depth=self.hpo_config.model_params["max_depth"],
                max_features=self.hpo_config.model_params["max_features"],
                n_bins=self.hpo_config.model_params["n_bins"],
            ).fit(X_train, y_train.astype("int32"))

        return trained_model

    @timer_decorator
    def predict(self, trained_model, X_test, threshold=0.5):
        """Inference with the trained model on the unseen test data"""

        hpo_log.info("> predict with trained model ")
        if "XGBoost" in self.hpo_config.model_type:
            dtest = xgboost.dask.DaskDMatrix(self.client, X_test)
            predictions = xgboost.dask.predict(self.client, trained_model, dtest).compute()

            predictions = (predictions > threshold) * 1.0

        elif "RandomForest" in self.hpo_config.model_type:
            predictions = trained_model.predict(X_test).compute()

        return predictions

    @timer_decorator
    def score(self, y_test, predictions):
        """Score predictions vs ground truth labels on test data"""
        hpo_log.info("> score predictions")
        y_test = y_test.compute()
        score = accuracy_score(
            y_test.astype(self.hpo_config.dataset_dtype),
            predictions.astype(self.hpo_config.dataset_dtype),
        )

        hpo_log.info(f"\t score = {score}")
        self.cv_fold_scores.append(score)
        return score

    def save_best_model(self, score, trained_model, filename="saved_model"):
        """Persist/save model that sets a new high score"""

        if score > self.best_score:
            self.best_score = score
            hpo_log.info("> saving high-scoring model")
            output_filename = os.path.join(self.hpo_config.model_store_directory, filename)

            if "XGBoost" in self.hpo_config.model_type:
                trained_model.save_model(f"{output_filename}_mgpu_xgb")
            elif "RandomForest" in self.hpo_config.model_type:
                joblib.dump(trained_model, f"{output_filename}_mgpu_rf")

    @timer_decorator
    async def cleanup(self, i_fold):
        """
        Close and restart the cluster when multiple cross validation folds
        are used to prevent memory creep.
        """
        if i_fold == self.hpo_config.cv_folds - 1:
            hpo_log.info("> done all folds; closing cluster")
            await self.client.close()
            await self.cluster.close()
        elif i_fold < self.hpo_config.cv_folds - 1:
            hpo_log.info("> end of fold; reinitializing cluster")
            await self.client.close()
            await self.cluster.close()
            self.cluster, self.client = self.cluster_initialize()

    def emit_final_score(self):
        """Emit score for parsing by the cloud HPO orchestrator"""
        exec_time = time.perf_counter() - self.start_time
        hpo_log.info(f"total_time = {exec_time:.5f} s ")

        if self.hpo_config.cv_folds > 1:
            hpo_log.info(f"fold scores : {self.cv_fold_scores}")

        # average over CV folds
        final_score = sum(self.cv_fold_scores) / len(self.cv_fold_scores)

        hpo_log.info(f"final-score: {final_score};")
