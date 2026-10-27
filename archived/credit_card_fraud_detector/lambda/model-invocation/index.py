##############################################################################
#  Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.   #
#                                                                            #
#  Licensed under the Amazon Software License (the "License"). You may not   #
#  use this file except in compliance with the License. A copy of the        #
#  License is located at                                                     #
#                                                                            #
#      http://aws.amazon.com/asl/                                            #
#                                                                            #
#  or in the "license" file accompanying this file. This file is distributed #
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,        #
#  express or implied. See the License for the specific language governing   #
#  permissions and limitations under the License.                            #
##############################################################################
import json
import os
import logging

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

STREAM_NAME = os.environ['StreamName']
SOLUTION_PREFIX = os.environ['SolutionPrefix']


def lambda_handler(event, context):
    logger.info(event)
    metadata = event.get('metadata', None)
    assert metadata, "Request did not include metadata!"
    data_payload = event.get('data', None)
    assert data_payload, "Payload did not include a data field!"
    model_choice = event.get('model', None)
    valid_models = {'anomaly_detector', 'fraud_classifier'}
    if model_choice:
        assert model_choice in valid_models, "The requested model, {}, was not a valid model name {}".format(model_choice, valid_models)
    models = {model_choice} if model_choice else valid_models

    output = {}
    if 'anomaly_detector' in models:
        output["anomaly_detector"] = get_anomaly_prediction(data_payload)

    if 'fraud_classifier' in models:
        output["fraud_classifier"] = get_fraud_prediction(data_payload)

    store_data_prediction(output, metadata)
    return output


def get_anomaly_prediction(data):
    sagemaker_endpoint_name = "{}-rcf".format(SOLUTION_PREFIX)
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name, ContentType='text/csv', Body=data)
    # Extract anomaly score from the endpoint response
    anomaly_score = json.loads(response['Body'].read().decode())["scores"][0]["score"]
    logger.info("anomaly score: {}".format(anomaly_score))

    return {"score": anomaly_score}


def get_fraud_prediction(data, threshold=0.5):
    sagemaker_endpoint_name = "{}-xgb".format(SOLUTION_PREFIX)
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name, ContentType='text/csv',Body=data)
    pred_proba = json.loads(response['Body'].read().decode())
    prediction = 0 if pred_proba < threshold else 1

    logger.info("classification pred_proba: {}, prediction: {}".format(pred_proba, prediction))

    return {"pred_proba": pred_proba, "prediction": prediction}


def store_data_prediction(output_dict, metadata):
    firehose_delivery_stream = STREAM_NAME
    firehose = boto3.client('firehose', region_name=os.environ['AWS_REGION'])

    # Extract anomaly score and classifier prediction, if they exist
    fraud_pred = output_dict["fraud_classifier"]["prediction"] if 'fraud_classifier' in output_dict else ""
    anomaly_score = output_dict["anomaly_detector"]["score"] if 'anomaly_detector' in output_dict else ""

    record = ','.join(metadata + [str(fraud_pred), str(anomaly_score)]) + '\n'

    success = firehose.put_record(
        DeliveryStreamName=firehose_delivery_stream, Record={'Data': record})
    if success:
        logger.info("Record logged: {}".format(record))
    else:
        logger.warning("Record delivery failed for record: {}".format(record))
