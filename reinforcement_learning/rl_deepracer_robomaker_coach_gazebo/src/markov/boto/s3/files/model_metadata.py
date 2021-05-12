"""This module implements s3 client for model metadata"""

import json
import logging
import os

from markov.architecture.constants import Input, NeuralNetwork
from markov.boto.s3.constants import ActionSpaceTypes, ModelMetadataKeys, TrainingAlgorithm
from markov.boto.s3.s3_client import S3Client
from markov.constants import SIMAPP_VERSION_1, SIMAPP_VERSION_2
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class ModelMetadata:
    """model metadata file upload, download, and parse"""

    def __init__(
        self,
        bucket,
        s3_key,
        region_name="us-east-1",
        local_path="./custom_files/agent/model_metadata.json",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
    ):
        """Model metadata upload, download, and parse

        Args:
            bucket (str): S3 bucket string
            s3_key: (str): S3 key string.
            region_name (str): S3 region name
            local_path (str): file local path
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry

        """
        # check s3 key and s3 bucket exist
        if not bucket or not s3_key:
            log_and_exit(
                "model_metadata S3 key or bucket not available for S3. \
                         bucket: {}, key {}".format(
                    bucket, s3_key
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._bucket = bucket
        # Strip the s3://<bucket> from uri, if s3_key past in as uri
        self._s3_key = s3_key.replace("s3://{}/".format(self._bucket), "")
        self._local_path = local_path
        self._local_dir = os.path.dirname(self._local_path)
        self._model_metadata = None
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)

    @property
    def local_dir(self):
        """return local dir of model metadata"""
        return self._local_dir

    @property
    def local_path(self):
        """return local path of model metadata"""
        return self._local_path

    @property
    def action_space(self):
        """return action space values passed as part of model metadata"""
        # download model_metadata.json and after successfully download, parse and set the class variable
        if not self._model_metadata:
            self._download_and_set_model_metadata()
        return self._model_metadata[ModelMetadataKeys.ACTION_SPACE.value]

    @property
    def action_space_type(self):
        """return action space type value passed as part of model metadata"""
        # download model_metadata.json and after successfully download, parse and set the class variable
        if not self._model_metadata:
            self._download_and_set_model_metadata()
        return self._model_metadata[ModelMetadataKeys.ACTION_SPACE_TYPE.value]

    @property
    def training_algorithm(self):
        """return training algorithm value passed as part of model metadata"""
        # download model_metadata.json and after successfully download, parse and set the class variable
        if not self._model_metadata:
            self._download_and_set_model_metadata()
        return self._model_metadata[ModelMetadataKeys.TRAINING_ALGORITHM.value]

    def get_action_dict(self, action):
        """return the action dict containing the steering_angle and speed value

        Args:
            action (int or list): model metadata action_space index for discreet action spaces
                                  or [steering, speed] float values for continuous action spaces

        Returns:
            dict (str, float): dictionary containing {steering_angle: value, speed: value}
        """
        if self.action_space_type == ActionSpaceTypes.DISCRETE.value:
            return self._model_metadata[ModelMetadataKeys.ACTION_SPACE.value][action]
        elif self.action_space_type == ActionSpaceTypes.CONTINUOUS.value:
            json_action = dict()
            json_action[ModelMetadataKeys.STEERING_ANGLE.value] = action[0]
            json_action[ModelMetadataKeys.SPEED.value] = action[1]
            return json_action
        else:
            log_and_exit(
                "Unknown action_space_type found while getting action dict. \
                action_space_type: {}".format(
                    self.action_space_type
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def _download_and_set_model_metadata(self):
        """download and parse the model metadata file"""
        # download model_metadata.json
        self._download()
        # after successfully download or use default model metadata, then parse
        self._model_metadata = self.parse_model_metadata(self._local_path)

    def get_model_metadata_info(self):
        """retrive the model metadata info

        Returns:
            dict: dictionary containing the information that is parsed from model_metatadata.json

        """
        # download model_metadata.json and after successfully download, parse and set the class variable
        if not self._model_metadata:
            self._download_and_set_model_metadata()

        return self._model_metadata

    def persist(self, s3_kms_extra_args):
        """upload local model_metadata.json into S3 bucket

        Args:
            s3_kms_extra_args (dict): s3 key management service extra argument

        """

        # persist model metadata
        # if retry failed, s3_client upload_file will log and exit 500
        self._s3_client.upload_file(
            bucket=self._bucket,
            s3_key=self._s3_key,
            local_path=self._local_path,
            s3_kms_extra_args=s3_kms_extra_args,
        )

    def _download(self):
        """download model_metadata.json with retry from s3 bucket"""

        # check and make local directory
        if self._local_dir and not os.path.exists(self._local_dir):
            os.makedirs(self._local_dir)
        # download model metadata
        # if retry failed, each worker.py and download_params_and_roslaunch_agent.py
        # will handle 400 and 500 separately
        self._s3_client.download_file(
            bucket=self._bucket, s3_key=self._s3_key, local_path=self._local_path
        )
        LOG.info(
            "[s3] Successfully downloaded model metadata \
                 from s3 key {} to local {}.".format(
                self._s3_key, self._local_path
            )
        )

    @staticmethod
    def parse_model_metadata(local_model_metadata_path):
        """parse model metadata give the local path

        Args:
            local_model_metadata_path (str): local model metadata string

        Returns:
            dict (str, obj): dictionary of all required information parsed out of model_metadata.json file

        """
        model_metadata = dict()
        try:
            with open(local_model_metadata_path, "r") as json_file:
                data = json.load(json_file)
                # simapp_version 2.0+ should contain version as key in
                # model_metadata.json
                if ModelMetadataKeys.ACTION_SPACE.value not in data:
                    raise ValueError("no action space defined")
                action_values = data[ModelMetadataKeys.ACTION_SPACE.value]
                if ModelMetadataKeys.VERSION.value in data:
                    simapp_version = float(data[ModelMetadataKeys.VERSION.value])
                    if simapp_version >= SIMAPP_VERSION_2:
                        sensor = data[ModelMetadataKeys.SENSOR.value]
                    else:
                        sensor = [Input.OBSERVATION.value]
                else:
                    if ModelMetadataKeys.SENSOR.value in data:
                        sensor = data[ModelMetadataKeys.SENSOR.value]
                        simapp_version = SIMAPP_VERSION_2
                    else:
                        sensor = [Input.OBSERVATION.value]
                        simapp_version = SIMAPP_VERSION_1
                if ModelMetadataKeys.NEURAL_NETWORK.value in data:
                    network = data[ModelMetadataKeys.NEURAL_NETWORK.value]
                else:
                    network = NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value
                training_algorithm = TrainingAlgorithm.CLIPPED_PPO.value
                if ModelMetadataKeys.TRAINING_ALGORITHM.value in data:
                    data_training_algorithm = (
                        data[ModelMetadataKeys.TRAINING_ALGORITHM.value].lower().strip()
                    )
                    # Update the training algorithm value if its valid else log and exit
                    if TrainingAlgorithm.has_training_algorithm(data_training_algorithm):
                        training_algorithm = data_training_algorithm
                    else:
                        log_and_exit(
                            "Unknown training_algorithm found while parsing model_metadata. \
                            training_algorithm: {}".format(
                                data_training_algorithm
                            ),
                            SIMAPP_SIMULATION_WORKER_EXCEPTION,
                            SIMAPP_EVENT_ERROR_CODE_500,
                        )
                action_space_type = ActionSpaceTypes.DISCRETE.value
                if ModelMetadataKeys.ACTION_SPACE_TYPE.value in data:
                    data_action_space_type = (
                        data[ModelMetadataKeys.ACTION_SPACE_TYPE.value].lower().strip()
                    )
                    # Update the training algorithm value if its valid else log and exit
                    if ActionSpaceTypes.has_action_space(data_action_space_type):
                        action_space_type = data_action_space_type
                    else:
                        log_and_exit(
                            "Unknown action_space_type found while parsing model_metadata. \
                            action_space_type: {}".format(
                                data_action_space_type
                            ),
                            SIMAPP_SIMULATION_WORKER_EXCEPTION,
                            SIMAPP_EVENT_ERROR_CODE_500,
                        )

            LOG.info(
                "Sensor list %s, network %s, simapp_version %s, training_algorithm %s, action_space_type %s",
                sensor,
                network,
                simapp_version,
                training_algorithm,
                action_space_type,
            )
            model_metadata[ModelMetadataKeys.SENSOR.value] = sensor
            model_metadata[ModelMetadataKeys.NEURAL_NETWORK.value] = network
            model_metadata[ModelMetadataKeys.VERSION.value] = simapp_version
            model_metadata[ModelMetadataKeys.TRAINING_ALGORITHM.value] = training_algorithm
            model_metadata[ModelMetadataKeys.ACTION_SPACE.value] = action_values
            model_metadata[ModelMetadataKeys.ACTION_SPACE_TYPE.value] = action_space_type
            return model_metadata
        except ValueError as ex:
            raise ValueError("model_metadata ValueError: {}".format(ex))
        except Exception as ex:
            raise Exception("Model metadata does not exist: {}".format(ex))
