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
import sys
import traceback

from HPOConfig import HPOConfig
from MLWorkflow import create_workflow


def train():
    hpo_config = HPOConfig(input_args=sys.argv[1:])
    ml_workflow = create_workflow(hpo_config)

    # cross-validation to improve robustness via multiple train/test reshuffles
    for i_fold in range(hpo_config.cv_folds):
        # ingest
        dataset = ml_workflow.ingest_data()

        # handle missing samples [ drop ]
        dataset = ml_workflow.handle_missing_data(dataset)

        # split into train and test set
        X_train, X_test, y_train, y_test = ml_workflow.split_dataset(dataset, random_state=i_fold)

        # train model
        trained_model = ml_workflow.fit(X_train, y_train)

        # use trained model to predict target labels of test data
        predictions = ml_workflow.predict(trained_model, X_test)

        # score test set predictions against ground truth
        score = ml_workflow.score(y_test, predictions)

        # save trained model [ if it sets a new-high score ]
        ml_workflow.save_best_model(score, trained_model)

        # restart cluster to avoid memory creep [ for multi-CPU/GPU ]
        ml_workflow.cleanup(i_fold)

    # emit final score to cloud HPO [i.e., SageMaker]
    ml_workflow.emit_final_score()


def configure_logging():
    hpo_log = logging.getLogger("hpo_log")
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(
        logging.Formatter("%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    )
    hpo_log.addHandler(log_handler)
    hpo_log.setLevel(logging.DEBUG)
    hpo_log.propagate = False


if __name__ == "__main__":
    configure_logging()
    try:
        train()
        sys.exit(0)  # success exit code
    except Exception:
        traceback.print_exc()
        sys.exit(-1)  # failure exit code
