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

import joblib
import pandas
import xgboost
from MLWorkflow import MLWorkflow, timer_decorator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

hpo_log = logging.getLogger("hpo_log")


class MLWorkflowSingleCPU(MLWorkflow):
    """Single-CPU Workflow"""

    def __init__(self, hpo_config):
        hpo_log.info("Single-CPU Workflow")
        self.start_time = time.perf_counter()

        self.hpo_config = hpo_config
        self.dataset_cache = None

        self.cv_fold_scores = []
        self.best_score = -1

    @timer_decorator
    def ingest_data(self):
        """Ingest dataset, CSV and Parquet supported"""

        if self.dataset_cache is not None:
            hpo_log.info("> skipping ingestion, using cache")
            return self.dataset_cache

        if "Parquet" in self.hpo_config.input_file_type:
            hpo_log.info("> parquet data ingestion")

            assert isinstance(self.hpo_config.target_files, str)
            filepath = self.hpo_config.target_files

            dataset = pandas.read_parquet(
                filepath, columns=self.hpo_config.dataset_columns, engine="pyarrow"  # noqa
            )

        elif "CSV" in self.hpo_config.input_file_type:
            hpo_log.info("> csv data ingestion")

            if isinstance(self.hpo_config.target_files, list):
                filepath = self.hpo_config.target_files[0]
            elif isinstance(self.hpo_config.target_files, str):
                filepath = self.hpo_config.target_files

            dataset = pandas.read_csv(
                filepath,
                names=self.hpo_config.dataset_columns,
                dtype=self.hpo_config.dataset_dtype,
                header=0,
            )

        hpo_log.info(f"\t dataset shape: {dataset.shape}")
        self.dataset_cache = dataset
        return dataset

    @timer_decorator
    def handle_missing_data(self, dataset):
        """Drop samples with missing data [ inplace ]"""
        dataset = dataset.dropna()
        return dataset

    @timer_decorator
    def split_dataset(self, dataset, random_state):
        """
        Split dataset into train and test data subsets,
        currently using CV-fold index for randomness.
        Plan to refactor with sklearn KFold
        """

        hpo_log.info("> train-test split")
        label_column = self.hpo_config.label_column

        X_train, X_test, y_train, y_test = train_test_split(
            dataset.loc[:, dataset.columns != label_column],
            dataset[label_column],
            random_state=random_state,
        )

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
            dtrain = xgboost.DMatrix(data=X_train, label=y_train)
            num_boost_round = self.hpo_config.model_params["num_boost_round"]
            trained_model = xgboost.train(
                dtrain=dtrain, params=self.hpo_config.model_params, num_boost_round=num_boost_round
            )

        elif "RandomForest" in self.hpo_config.model_type:
            hpo_log.info("> fit randomforest model")
            trained_model = RandomForestClassifier(
                n_estimators=self.hpo_config.model_params["n_estimators"],
                max_depth=self.hpo_config.model_params["max_depth"],
                max_features=self.hpo_config.model_params["max_features"],
                bootstrap=self.hpo_config.model_params["bootstrap"],
                n_jobs=-1,
            ).fit(X_train, y_train)

        return trained_model

    @timer_decorator
    def predict(self, trained_model, X_test, threshold=0.5):
        """Inference with the trained model on the unseen test data"""

        hpo_log.info("> predict with trained model ")
        if "XGBoost" in self.hpo_config.model_type:
            dtest = xgboost.DMatrix(X_test)
            predictions = trained_model.predict(dtest)
            predictions = (predictions > threshold) * 1.0
        elif "RandomForest" in self.hpo_config.model_type:
            predictions = trained_model.predict(X_test)

        return predictions

    @timer_decorator
    def score(self, y_test, predictions):
        """Score predictions vs ground truth labels on test data"""
        dataset_dtype = self.hpo_config.dataset_dtype
        score = accuracy_score(y_test.astype(dataset_dtype), predictions.astype(dataset_dtype))

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
                trained_model.save_model(f"{output_filename}_scpu_xgb")
            elif "RandomForest" in self.hpo_config.model_type:
                joblib.dump(trained_model, f"{output_filename}_scpu_rf")

    def cleanup(self, i_fold):
        hpo_log.info("> end of fold \n")

    def emit_final_score(self):
        """Emit score for parsing by the cloud HPO orchestrator"""
        exec_time = time.perf_counter() - self.start_time
        hpo_log.info(f"total_time = {exec_time:.5f} s ")

        if self.hpo_config.cv_folds > 1:
            hpo_log.info(f"fold scores : {self.cv_fold_scores} \n")

        # average over CV folds
        final_score = sum(self.cv_fold_scores) / len(self.cv_fold_scores)

        hpo_log.info(f"final-score: {final_score}; \n")
