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
"""This module contains code related to the DataCaptureConfig class.

Codes are used for configuring capture, collection, and storage, for prediction requests and
responses for models hosted on SageMaker Endpoints.
"""
from __future__ import print_function, absolute_import

from sagemaker import s3
from sagemaker.config import ENDPOINT_CONFIG_DATA_CAPTURE_KMS_KEY_ID_PATH
from sagemaker.session import Session
from sagemaker.utils import resolve_value_from_config

_MODEL_MONITOR_S3_PATH = "model-monitor"
_DATA_CAPTURE_S3_PATH = "data-capture"


class DataCaptureConfig(object):
    """Configuration object passed in when deploying models to Amazon SageMaker Endpoints.

    This object specifies configuration related to endpoint data capture for use with
    Amazon SageMaker Model Monitoring.
    """

    API_MAPPING = {"REQUEST": "Input", "RESPONSE": "Output"}

    def __init__(
        self,
        enable_capture,
        sampling_percentage=20,
        destination_s3_uri=None,
        kms_key_id=None,
        capture_options=None,
        csv_content_types=None,
        json_content_types=None,
        sagemaker_session=None,
    ):
        """Initialize a DataCaptureConfig object for capturing data from Amazon SageMaker Endpoints.

        Args:
            enable_capture (bool): Required. Whether data capture should be enabled or not.
            sampling_percentage (int): Optional. Default=20. The percentage of data to sample.
                Must be between 0 and 100.
            destination_s3_uri (str): Optional. Defaults to "s3://<default-session-bucket>/
                model-monitor/data-capture".
            kms_key_id (str): Optional. Default=None. The kms key to use when writing to S3.
            capture_options ([str]): Optional. Must be a list containing any combination of the
                following values: "REQUEST", "RESPONSE". Default=["REQUEST", "RESPONSE"]. Denotes
                which data to capture between request and response.
            csv_content_types ([str]): Optional. Default=["text/csv"].
            json_content_types([str]): Optional. Default=["application/json"].
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
        """
        self.enable_capture = enable_capture
        self.sampling_percentage = sampling_percentage
        self.destination_s3_uri = destination_s3_uri
        sagemaker_session = sagemaker_session or Session()
        if self.destination_s3_uri is None:
            self.destination_s3_uri = s3.s3_path_join(
                "s3://",
                sagemaker_session.default_bucket(),
                sagemaker_session.default_bucket_prefix,
                _MODEL_MONITOR_S3_PATH,
                _DATA_CAPTURE_S3_PATH,
            )

        self.kms_key_id = resolve_value_from_config(
            kms_key_id,
            ENDPOINT_CONFIG_DATA_CAPTURE_KMS_KEY_ID_PATH,
            sagemaker_session=sagemaker_session,
        )
        self.capture_options = capture_options or ["REQUEST", "RESPONSE"]
        self.csv_content_types = csv_content_types or ["text/csv"]
        self.json_content_types = json_content_types or ["application/json"]

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {
            "EnableCapture": self.enable_capture,
            "InitialSamplingPercentage": self.sampling_percentage,
            "DestinationS3Uri": self.destination_s3_uri,
            "CaptureOptions": [
                #  Convert to API values or pass value directly through if unable to convert.
                {"CaptureMode": self.API_MAPPING.get(capture_option.upper(), capture_option)}
                for capture_option in self.capture_options
            ],
        }

        if self.kms_key_id is not None:
            request_dict["KmsKeyId"] = self.kms_key_id

        if self.csv_content_types is not None or self.json_content_types is not None:
            request_dict["CaptureContentTypeHeader"] = {}

        if self.csv_content_types is not None:
            request_dict["CaptureContentTypeHeader"]["CsvContentTypes"] = self.csv_content_types

        if self.json_content_types is not None:
            request_dict["CaptureContentTypeHeader"]["JsonContentTypes"] = self.json_content_types

        return request_dict
