#!/usr/bin/env python

import os
import json
import sys
import pathlib
import argparse
import datetime
import time
import boto3

os.system("du -a /opt/ml")
print(os.environ)


def train(args):
    print(args)
    comprehend = boto3.client("comprehend", region_name=os.environ["AWS_REGION"])

    s3_train_data = args.train_input_file
    s3_train_output = args.train_output_path

    role_arn = args.iam_role_arn

    id_ = str(datetime.datetime.now().strftime("%s"))
    create_custom_classify_response = comprehend.create_document_classifier(
        DocumentClassifierName="custom-classifier" + id_,
        DataAccessRoleArn=role_arn,
        InputDataConfig={"DataFormat": "COMPREHEND_CSV", "S3Uri": s3_train_data},
        OutputDataConfig={"S3Uri": s3_train_output},
        LanguageCode="en",
    )

    jobArn = create_custom_classify_response["DocumentClassifierArn"]

    max_time = time.time() + 3 * 60 * 60  # 3 hours
    while time.time() < max_time:
        describe_custom_classifier = comprehend.describe_document_classifier(
            DocumentClassifierArn=jobArn
        )
        status = describe_custom_classifier["DocumentClassifierProperties"]["Status"]
        print("Custom classifier: {}".format(status))

        if status == "IN_ERROR":
            sys.exit(1)

        if status == "TRAINED":
            evaluation_metrics = describe_custom_classifier[
                "DocumentClassifierProperties"
            ]["ClassifierMetadata"]["EvaluationMetrics"]

            acc = evaluation_metrics.get("Accuracy")
            arn = describe_custom_classifier["DocumentClassifierProperties"][
                "DocumentClassifierArn"
            ]

            evaluation_output_dir = "/opt/ml/processing/evaluation"
            pathlib.Path(evaluation_output_dir).mkdir(parents=True, exist_ok=True)

            print("Writing out evaluation report with Accuracy: %f", acc)
            evaluation_path = f"{evaluation_output_dir}/evaluation.json"
            with open(evaluation_path, "w") as f:
                f.write(json.dumps(evaluation_metrics))

            arn_output_dir = "/opt/ml/processing/arn"
            pathlib.Path(arn_output_dir).mkdir(parents=True, exist_ok=True)

            print(f"Writing out classifier arn {arn}")
            arn_path = f"{arn_output_dir}/arn.txt"
            with open(arn_path, "w") as f:
                f.write(arn)

            break

        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-input-file", type=str, help="Path of input training file"
    )
    parser.add_argument("--train-output-path", type=str, help="s3 output folder")
    parser.add_argument(
        "--iam-role-arn",
        type=str,
        help="ARN of role allowing SageMaker to trigger Comprehend",
    )
    args = parser.parse_args()
    print(args)

    train(args)
