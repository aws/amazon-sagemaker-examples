# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import argparse
import ast
import logging
import os
import torch
import json
from bertopic import BERTopic

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _train(args):
    
    logger.debug("BERTtopic training starting")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    model = BERTopic(language=args.language)

    logger.info("BERTtopic Model loaded for language {}".format(args.language))
    
    print("Loading Training data")
    docs = []
    print(f"data_dir: {args.data_dir}")
    with open(args.data_dir+"/training_file.txt") as file:
        for line in file:
            docs.append(line.rstrip())
    
    print("Started Training")
    topics, probs = model.fit_transform(docs)
    print("Finished Training")
    return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "my_model")
    model.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # custom parameter for BERTopic
    parser.add_argument(
        "--language", type=str, default="english", help="main language for the input documents. If you want a multilingual model that supports 50+ languages, select \"multilingual\"."
    )

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    _train(parser.parse_args())
