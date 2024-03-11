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
"""This module contains code related to the ModelMonitoringFile class.

Codes are used for managing the constraints and statistics JSON files generated and consumed by
Amazon SageMaker Model Monitoring Schedules.
"""
from __future__ import print_function, absolute_import

import json
import os
import uuid

from botocore.exceptions import ClientError

from sagemaker import s3
from sagemaker.session import Session

NO_SUCH_KEY_CODE = "NoSuchKey"


class ModelMonitoringFile(object):
    """Represents a file with a body and an S3 uri."""

    def __init__(self, body_dict, file_s3_uri, kms_key, sagemaker_session):
        """Initializes a file with a body and an S3 uri.

        Args:
            body_dict (str): The body of the JSON file.
            file_s3_uri (str): The uri of the JSON file.
            kms_key (str): The kms key to be used to decrypt the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        """
        self.body_dict = body_dict
        self.file_s3_uri = file_s3_uri
        self.kms_key = kms_key
        self.session = sagemaker_session

    def save(self, new_save_location_s3_uri=None):
        """Save the current instance's body to s3 using the instance's s3 path.

        The S3 path can be overridden by providing one. This also overrides the
        default save location for this object.

        Args:
            new_save_location_s3_uri (str): Optional. The S3 path to save the file to. If not
                provided, the file is saved in place in S3. If provided, the file's S3 path is
                permanently updated.

        Returns:
            str: The s3 location to which the file was saved.

        """
        if new_save_location_s3_uri is not None:
            self.file_s3_uri = new_save_location_s3_uri

        return s3.S3Uploader.upload_string_as_file_body(
            body=json.dumps(self.body_dict),
            desired_s3_uri=self.file_s3_uri,
            kms_key=self.kms_key,
            sagemaker_session=self.session,
        )


class Statistics(ModelMonitoringFile):
    """Represents the statistics JSON file used in Amazon SageMaker Model Monitoring."""

    def __init__(self, body_dict, statistics_file_s3_uri, kms_key=None, sagemaker_session=None):
        """Initializes the Statistics object used in Amazon SageMaker Model Monitoring.

        Args:
            body_dict (str): The body of the statistics JSON file.
            statistics_file_s3_uri (str): The uri of the statistics JSON file.
            kms_key (str): The kms key to be used to decrypt the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        """
        super(Statistics, self).__init__(
            body_dict=body_dict,
            file_s3_uri=statistics_file_s3_uri,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def from_s3_uri(cls, statistics_file_s3_uri, kms_key=None, sagemaker_session=None):
        """Generates a Statistics object from an s3 uri.

        Args:
            statistics_file_s3_uri (str): The uri of the statistics JSON file.
            kms_key (str): The kms key to be used to decrypt the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.Statistics: The instance of Statistics generated from
                the s3 uri.

        """
        try:
            body_dict = json.loads(
                s3.S3Downloader.read_file(
                    s3_uri=statistics_file_s3_uri, sagemaker_session=sagemaker_session
                )
            )
        except ClientError as error:
            print(
                "\nCould not retrieve statistics file at location '{}'. "
                "To manually retrieve Statistics object from a given uri, "
                "use 'my_model_monitor.statistics(my_s3_uri)' or "
                "'Statistics.from_s3_uri(my_s3_uri)'".format(statistics_file_s3_uri)
            )
            raise error

        return cls(
            body_dict=body_dict, statistics_file_s3_uri=statistics_file_s3_uri, kms_key=kms_key
        )

    @classmethod
    def from_string(
        cls, statistics_file_string, kms_key=None, file_name=None, sagemaker_session=None
    ):
        """Generates a Statistics object from an s3 uri.

        Args:
            statistics_file_string (str): The uri of the statistics JSON file.
            kms_key (str): The kms key to be used to encrypt the file in S3.
            file_name (str): The file name to use when uploading to S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.Statistics: The instance of Statistics generated from
                the s3 uri.

        """
        sagemaker_session = sagemaker_session or Session()
        file_name = file_name or "statistics.json"
        desired_s3_uri = s3.s3_path_join(
            "s3://",
            sagemaker_session.default_bucket(),
            sagemaker_session.default_bucket_prefix,
            "monitoring",
            str(uuid.uuid4()),
            file_name,
        )
        s3_uri = s3.S3Uploader.upload_string_as_file_body(
            body=statistics_file_string,
            desired_s3_uri=desired_s3_uri,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

        return Statistics.from_s3_uri(
            statistics_file_s3_uri=s3_uri, kms_key=kms_key, sagemaker_session=sagemaker_session
        )

    @classmethod
    def from_file_path(cls, statistics_file_path, kms_key=None, sagemaker_session=None):
        """Initializes a Statistics object from a file path.

        Args:
            statistics_file_path (str): The path to the statistics file.
            kms_key (str): The kms_key to use when encrypting the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.Statistics: The instance of Statistics generated from
                the local file path.

        """
        file_name = os.path.basename(statistics_file_path)

        with open(statistics_file_path, "r") as f:
            file_body = f.read()

        return Statistics.from_string(
            statistics_file_string=file_body,
            file_name=file_name,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )


class Constraints(ModelMonitoringFile):
    """Represents the constraints JSON file used in Amazon SageMaker Model Monitoring."""

    def __init__(self, body_dict, constraints_file_s3_uri, kms_key=None, sagemaker_session=None):
        """Initializes the Constraints object used in Amazon SageMaker Model Monitoring.

        Args:
            body_dict (str): The body of the constraints JSON file.
            constraints_file_s3_uri (str): The uri of the constraints JSON file.
            kms_key (str): The kms key to be used to decrypt the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        """
        super(Constraints, self).__init__(
            body_dict=body_dict,
            file_s3_uri=constraints_file_s3_uri,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def from_s3_uri(cls, constraints_file_s3_uri, kms_key=None, sagemaker_session=None):
        """Generates a Constraints object from an s3 uri.

        Args:
            constraints_file_s3_uri (str): The uri of the constraints JSON file.
            kms_key (str): The kms key to be used to decrypt the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.Constraints: The instance of Constraints generated from
                the s3 uri.

        """
        try:
            body_dict = json.loads(
                s3.S3Downloader.read_file(
                    s3_uri=constraints_file_s3_uri, sagemaker_session=sagemaker_session
                )
            )
        except ClientError as error:
            print(
                "\nCould not retrieve constraints file at location '{}'. "
                "To manually retrieve Constraints object from a given uri, "
                "use 'my_model_monitor.constraints(my_s3_uri)' or "
                "'Constraints.from_s3_uri(my_s3_uri)'".format(constraints_file_s3_uri)
            )
            raise error

        return cls(
            body_dict=body_dict,
            constraints_file_s3_uri=constraints_file_s3_uri,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def from_string(
        cls, constraints_file_string, kms_key=None, file_name=None, sagemaker_session=None
    ):
        """Generates a Constraints object from an s3 uri.

        Args:
            constraints_file_string (str): The uri of the constraints JSON file.
            kms_key (str): The kms key to be used to encrypt the file in S3.
            file_name (str): The file name to use when uploading to S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.Constraints: The instance of Constraints generated from
                the s3 uri.

        """
        sagemaker_session = sagemaker_session or Session()
        file_name = file_name or "constraints.json"
        desired_s3_uri = s3.s3_path_join(
            "s3://",
            sagemaker_session.default_bucket(),
            sagemaker_session.default_bucket_prefix,
            "monitoring",
            str(uuid.uuid4()),
            file_name,
        )
        s3_uri = s3.S3Uploader.upload_string_as_file_body(
            body=constraints_file_string,
            desired_s3_uri=desired_s3_uri,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

        return Constraints.from_s3_uri(
            constraints_file_s3_uri=s3_uri, kms_key=kms_key, sagemaker_session=sagemaker_session
        )

    @classmethod
    def from_file_path(cls, constraints_file_path, kms_key=None, sagemaker_session=None):
        """Initializes a Constraints object from a file path.

        Args:
            constraints_file_path (str): The path to the constraints file.
            kms_key (str): The kms_key to use when encrypting the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.Constraints: The instance of Constraints generated from
                the local file path.

        """
        file_name = os.path.basename(constraints_file_path)

        with open(constraints_file_path, "r") as f:
            file_body = f.read()

        return Constraints.from_string(
            constraints_file_string=file_body,
            file_name=file_name,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

    def set_monitoring(self, enable_monitoring, feature_name=None):
        """Sets the monitoring flags on this Constraints object.

        If feature-name is provided, modify the feature-level override.
        Else, modify the top-level monitoring flag.

        Args:
            enable_monitoring (bool): Whether to enable monitoring or not.
            feature_name (str): Sets the feature-level monitoring flag if provided. Otherwise,
                sets the file-level override.

        """
        monitoring_api_map = {True: "Enabled", False: "Disabled"}
        flag = monitoring_api_map[enable_monitoring]
        if feature_name is None:
            self.body_dict["monitoring_config"]["evaluate_constraints"] = flag
        else:
            for feature in self.body_dict["features"]:
                if feature["name"] == feature_name:
                    string_constraints = feature["string_constraints"]
                    if string_constraints.get("monitoring_config_overrides") is None:
                        string_constraints["monitoring_config_overrides"] = {}
                    string_constraints["monitoring_config_overrides"]["evaluate_constraints"] = flag


class ConstraintViolations(ModelMonitoringFile):
    """Represents the constraint violations JSON file used in Amazon SageMaker Model Monitoring."""

    def __init__(
        self, body_dict, constraint_violations_file_s3_uri, kms_key=None, sagemaker_session=None
    ):
        """Initializes the ConstraintViolations object used in Amazon SageMaker Model Monitoring.

        Args:
            body_dict (str): The body of the constraint violations JSON file.
            constraint_violations_file_s3_uri (str): The uri of the constraint violations JSON file.
            kms_key (str): The kms key to be used to decrypt the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        """
        super(ConstraintViolations, self).__init__(
            body_dict=body_dict,
            file_s3_uri=constraint_violations_file_s3_uri,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def from_s3_uri(cls, constraint_violations_file_s3_uri, kms_key=None, sagemaker_session=None):
        """Generates a ConstraintViolations object from an s3 uri.

        Args:
            constraint_violations_file_s3_uri (str): The uri of the constraint violations JSON file.
            kms_key (str): The kms key to be used to decrypt the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.ConstraintViolations: The instance of ConstraintViolations
                generated from the s3 uri.

        """
        try:
            body_dict = json.loads(
                s3.S3Downloader.read_file(
                    s3_uri=constraint_violations_file_s3_uri, sagemaker_session=sagemaker_session
                )
            )
        except ClientError as error:
            print(
                "\nCould not retrieve constraints file at location '{}'. "
                "To manually retrieve ConstraintViolations object from a given uri, "
                "use 'my_model_monitor.constraints(my_s3_uri)' or "
                "'ConstraintViolations.from_s3_uri(my_s3_uri)'".format(
                    constraint_violations_file_s3_uri
                )
            )
            raise error

        return cls(
            body_dict=body_dict,
            constraint_violations_file_s3_uri=constraint_violations_file_s3_uri,
            kms_key=kms_key,
        )

    @classmethod
    def from_string(
        cls, constraint_violations_file_string, kms_key=None, file_name=None, sagemaker_session=None
    ):
        """Generates a ConstraintViolations object from an s3 uri.

        Args:
            constraint_violations_file_string (str): The uri of the constraint violations JSON file.
            kms_key (str): The kms key to be used to encrypt the file in S3.
            file_name (str): The file name to use when uploading to S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.ConstraintViolations: The instance of ConstraintViolations
                generated from the s3 uri.

        """
        sagemaker_session = sagemaker_session or Session()
        file_name = file_name or "constraint_violations.json"
        desired_s3_uri = s3.s3_path_join(
            "s3://",
            sagemaker_session.default_bucket(),
            sagemaker_session.default_bucket_prefix,
            "monitoring",
            str(uuid.uuid4()),
            file_name,
        )
        s3_uri = s3.S3Uploader.upload_string_as_file_body(
            body=constraint_violations_file_string,
            desired_s3_uri=desired_s3_uri,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

        return ConstraintViolations.from_s3_uri(
            constraint_violations_file_s3_uri=s3_uri,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def from_file_path(cls, constraint_violations_file_path, kms_key=None, sagemaker_session=None):
        """Initializes a ConstraintViolations object from a file path.

        Args:
            constraint_violations_file_path (str): The path to the constraint violations file.
            kms_key (str): The kms_key to use when encrypting the file in S3.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.

        Returns:
            sagemaker.model_monitor.ConstraintViolations: The instance of ConstraintViolations
                generated from the local file path.

        """
        file_name = os.path.basename(constraint_violations_file_path)

        with open(constraint_violations_file_path, "r") as f:
            file_body = f.read()

        return ConstraintViolations.from_string(
            constraint_violations_file_string=file_body,
            file_name=file_name,
            kms_key=kms_key,
            sagemaker_session=sagemaker_session,
        )
