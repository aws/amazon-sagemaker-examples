"""This module implements s3 client for yaml file"""

import logging
import os

import botocore
import yaml
from markov.boto.s3.constants import (
    EVAL_MANDATORY_YAML_KEY,
    F1_RACE_TYPE,
    F1_SHELL_USERS_LIST,
    MODEL_METADATA_S3_POSTFIX,
    TOUR_MANDATORY_YAML_KEY,
    TRAINING_MANDATORY_YAML_KEY,
    VIRUTAL_EVENT_MANDATORY_YAML_KEY,
    AgentType,
    YamlKey,
)
from markov.boto.s3.s3_client import S3Client
from markov.boto.s3.utils import is_power_of_two
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.reset.constants import RaceType
from markov.rollout_constants import BodyShellType, CarColorType
from markov.utils import force_list

LOG = Logger(__name__, logging.INFO).get_logger()


class YamlFile:
    """yaml file upload, download, and parse"""

    def __init__(
        self,
        agent_type,
        bucket,
        s3_key,
        region_name="us-east-1",
        local_path="params.yaml",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
    ):
        """yaml upload, download, and parse

        Args:
            agent_type (str): rollout for training, evaluation for eval
            bucket (str): S3 bucket string
            s3_key: (str): S3 key string.
            region_name (str): S3 region name
            local_path (str): file local path
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry

        """
        if not bucket or not s3_key:
            log_and_exit(
                "yaml file S3 key or bucket not available for S3. \
                         bucket: {}, key: {}".format(
                    bucket, s3_key
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._bucket = bucket
        # Strip the s3://<bucket> from uri, if s3_key past in as uri
        self._s3_key = s3_key.replace("s3://{}/".format(self._bucket), "")
        self._local_path = local_path
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)
        self._agent_type = agent_type
        if self._agent_type == AgentType.ROLLOUT.value:
            self._model_s3_bucket_yaml_key = YamlKey.SAGEMAKER_SHARED_S3_BUCKET_YAML_KEY.value
            self._model_s3_prefix_yaml_key = YamlKey.SAGEMAKER_SHARED_S3_PREFIX_YAML_KEY.value
            self._mandatory_yaml_key = TRAINING_MANDATORY_YAML_KEY
        elif self._agent_type == AgentType.EVALUATION.value:
            self._model_s3_bucket_yaml_key = YamlKey.MODEL_S3_BUCKET_YAML_KEY.value
            self._model_s3_prefix_yaml_key = YamlKey.MODEL_S3_PREFIX_YAML_KEY.value
            self._mandatory_yaml_key = EVAL_MANDATORY_YAML_KEY
        elif self._agent_type == AgentType.VIRTUAL_EVENT.value:
            self._mandatory_yaml_key = VIRUTAL_EVENT_MANDATORY_YAML_KEY
        else:
            log_and_exit(
                "Unknown agent type in launch file: {}".format(self._agent_type),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._yaml_values = None
        self._is_multicar = False
        self._is_f1 = False
        self._model_s3_buckets = list()
        self._model_metadata_s3_keys = list()
        self._body_shell_types = list()
        self._kinesis_webrtc_signaling_channel_name = None

    @property
    def local_path(self):
        """return local path of yaml file"""
        return self._local_path

    @property
    def is_f1(self):
        """return True for f1 race, else False"""
        # is f1 is updated within get_yaml_values. Therefore,
        # call get_yaml_values first to make sure value is updated.
        self.get_yaml_values()
        return self._is_f1

    @property
    def is_multicar(self):
        """return True for multicar, else False"""
        # is multicar is updated within get_yaml_values. Therefore,
        # call get_yaml_values first to make sure value is update.
        self.get_yaml_values()
        return self._is_multicar

    @property
    def model_s3_buckets(self):
        """return list of model s3 bucket"""
        self.get_yaml_values()
        return self._model_s3_buckets

    @property
    def model_metadata_s3_keys(self):
        """return list of model metadata s3 keys"""
        self.get_yaml_values()
        return self._model_metadata_s3_keys

    @property
    def body_shell_types(self):
        """return list of body shell types"""
        self.get_yaml_values()
        return self._body_shell_types

    @property
    def kinesis_webrtc_signaling_channel_name(self):
        """return the KVS WebRTC Signaling Channel Name"""
        self.get_yaml_values()
        return self._kinesis_webrtc_signaling_channel_name

    def get_yaml_values(self):
        """download yaml file from s3, load yaml file into yaml value,
           update yaml values, and return the yaml values dictionary

        Returns:
            dict: dictionary of yaml values
        """

        # parse yaml dict into yaml value
        if not self._yaml_values:
            # download yaml file
            self._download()
            # parse yaml file into yaml values
            self._load_yaml_values()
            # No need to any of these check for virtual event as they will be passed in dynamically
            # TODO: THIS CHECK IS VERY UGLY
            # Consider refactor Yaml class separately to reduce these kinds of if checks
            if not self._agent_type == AgentType.VIRTUAL_EVENT.value:
                # update body shell type
                # TODO: delete upload_body_shell if body shell type is past in by cloud service
                # team. Right now, body shell type is not past in tournament.
                self._update_body_shell()
                # postprocess yaml values
                self._postprocess_yaml_values()
                # update body color
                # TODO: delete after car_color is a mandatory yaml key
                self._update_car_color()
            # validate the yaml values based on different agent type
            if self._agent_type == AgentType.VIRTUAL_EVENT.value:
                self._validate_virtual_event_yaml_values()
                # set body shell types
                self._body_shell_types = self._yaml_values[YamlKey.BODY_SHELL_TYPE_YAML_KEY.value]
                self._kinesis_webrtc_signaling_channel_name = self._yaml_values.get(
                    YamlKey.KINESIS_WEBRTC_SIGNALING_CHANNEL_NAME.value
                )
            else:
                self._validate_yaml_values()
        return self._yaml_values

    def _download(self):
        """download yaml file from se with retry logic"""

        # check and make local directory
        local_dir = os.path.dirname(self._local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # download with retry
        # fault will be caught as 500 as original logic in
        # download_params_and_roslaunch_agent.py
        self._s3_client.download_file(
            bucket=self._bucket, s3_key=self._s3_key, local_path=self._local_path
        )
        LOG.info(
            "[s3] Successfully downloaded yaml file from s3 key {} to local {}.".format(
                self._s3_key, self._local_path
            )
        )

    def _load_yaml_values(self):
        """load local yaml file into dict"""

        # load yaml file into yaml values
        with open(self._local_path, "r") as yaml_file:
            try:
                self._yaml_values = yaml.safe_load(yaml_file)
            except yaml.YAMLError as exc:
                log_and_exit(
                    "yaml read error: {}".format(exc),
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_500,
                )

    def _postprocess_yaml_values(self):
        """postprocess yaml values

        First, it will force all FORCE_LIST_PARAMS to a list.
        Second, MODEL_METADATA_FILE_S3_YAML_KEY does not exist in eval,
        so we have to manually write it.
        """
        if not self._agent_type == AgentType.VIRTUAL_EVENT.value:
            # Forcing all mandatory yaml param to list
            for params in self._mandatory_yaml_key:
                if params in self._yaml_values:
                    self._yaml_values[params] = force_list(self._yaml_values[params])
        # TODO: delete this logic if cloud service team can always pass model metadata
        # populate the model_metadata_s3_key values to handle both
        # training and evaluation for all race_formats
        if YamlKey.MODEL_METADATA_FILE_S3_YAML_KEY.value not in self._yaml_values:
            # MODEL_METADATA_FILE_S3_KEY not passed as part of yaml file ==> This happens during
            # evaluation Assume model_metadata.json is present in the s3_prefix/model/ folder
            self._yaml_values[YamlKey.MODEL_METADATA_FILE_S3_YAML_KEY.value] = list()
            for s3_prefix in self._yaml_values[self._model_s3_prefix_yaml_key]:
                self._yaml_values[YamlKey.MODEL_METADATA_FILE_S3_YAML_KEY.value].append(
                    os.path.join(s3_prefix, MODEL_METADATA_S3_POSTFIX)
                )

        # set model s3 buckets
        self._model_s3_buckets = self._yaml_values[self._model_s3_bucket_yaml_key]
        # set model metadata
        self._model_metadata_s3_keys = self._yaml_values[
            YamlKey.MODEL_METADATA_FILE_S3_YAML_KEY.value
        ]
        # set body shell types
        self._body_shell_types = self._yaml_values[YamlKey.BODY_SHELL_TYPE_YAML_KEY.value]
        # set multicar value if there is more than one value in self._model_s3_bucket_yaml_key.
        self._is_multicar = len(self._yaml_values[self._model_s3_bucket_yaml_key]) > 1
        # set f1 as true if RACE_TYPE is F1
        self._is_f1 = (
            self._yaml_values.get(YamlKey.RACE_TYPE_YAML_KEY.value, RaceType.TIME_TRIAL.value)
            == F1_RACE_TYPE
        )

    def _validate_yaml_values(self):
        """Validate that the parameter provided in the yaml file for configuration is correct.
        Some of the params requires list of two values. This is mostly checked as part of
        this function."""

        # Verify yaml keys required for launching models have same number of values
        LOG.info(self._yaml_values)
        if not all(
            map(
                lambda param: len(self._yaml_values[param])
                == len(self._yaml_values[self._mandatory_yaml_key[0]]),
                self._mandatory_yaml_key,
            )
        ):
            raise Exception(
                "Incorrect number of values for these yaml parameters {}".format(
                    self._mandatory_yaml_key
                )
            )

        # Verify if all yaml keys have at least 2 values for multi car racing
        if self._is_multicar and len(self._yaml_values[self._model_s3_prefix_yaml_key]) < 2:
            raise Exception(
                "Incorrect number of values for multicar racing yaml parameters {}".format(
                    self._mandatory_yaml_key
                )
            )

        # Verify if all yaml keys have 1 value for single car racing
        if not self._is_multicar and len(self._yaml_values[self._model_s3_prefix_yaml_key]) != 1:
            raise Exception(
                "Incorrect number of values for single car racing yaml parameters {}".format(
                    self._mandatory_yaml_key
                )
            )

    def _validate_virtual_event_yaml_values(self):
        """Validate that the parameter provided in the yaml file for
        virtual event configuration is correct.
        """
        # Verify the mendatory yaml values for virtual event are present
        for params in self._mandatory_yaml_key:
            if params not in self._yaml_values:
                raise Exception("Mandatory YAML key is not present {}".format(params))

    def _update_car_color(self):
        """update car color to default black if not exist to make sure validation can pass"""
        if YamlKey.CAR_COLOR_YAML_KEY.value not in self._yaml_values:
            self._yaml_values[YamlKey.CAR_COLOR_YAML_KEY.value] = [CarColorType.BLACK.value] * len(
                self._model_s3_buckets
            )

    def _update_body_shell(self):
        """update body shell type in yaml dict and then overwrite local yaml file
        with updated body shell type. This is for backward compatibility when
        body shell is not pasted in for F1 with specific racer alias or
        display name"""

        # List of body shell types
        body_shell_types = force_list(
            self._yaml_values.get(YamlKey.BODY_SHELL_TYPE_YAML_KEY.value, [None])
        )
        racer_names = force_list(self._yaml_values.get(YamlKey.RACER_NAME_YAML_KEY.value, [None]))
        display_names = force_list(
            self._yaml_values.get(YamlKey.DISPLAY_NAME_YAML_KEY.value, [None])
        )
        # If body_shell_types contains only None, figure out shell based on names
        # otherwise use body_shell_type defined in body_shell_types

        if None in body_shell_types:
            # use default shells only if both RACER_NAME and DISPLAY_NAME are empty
            if None in racer_names and None in display_names:
                body_shell_types = [BodyShellType.DEFAULT.value] * len(
                    force_list(self._yaml_values[self._model_s3_bucket_yaml_key])
                )
            else:
                # If RACER_NAME is empty, use DISPLAY_NAME to get racer_alias,
                # and check racer_alias in F1_SHELL_USERS_LIST whether to use F1 shell or not,
                # otherwise use RACER_NAME as racer_alias to figure out
                # whether to use the f1 body shell.
                if None in racer_names:
                    body_shell_types = [
                        BodyShellType.F1_2021.value
                        if racer_alias in F1_SHELL_USERS_LIST
                        else BodyShellType.DEFAULT.value
                        for racer_alias in display_names
                    ]
                else:
                    body_shell_types = [
                        BodyShellType.F1_2021.value
                        if racer_alias in F1_SHELL_USERS_LIST
                        else BodyShellType.DEFAULT.value
                        for racer_alias in racer_names
                    ]
            self._yaml_values[YamlKey.BODY_SHELL_TYPE_YAML_KEY.value] = body_shell_types
            # override local yaml file with updated BODY_SHELL_TYPE
            self._overwrite_local_yaml_file()

    def _overwrite_local_yaml_file(self):
        """override local yaml file with updated self._yaml_values"""

        with open(self._local_path, "w") as yaml_file:
            yaml.dump(self._yaml_values, yaml_file)
