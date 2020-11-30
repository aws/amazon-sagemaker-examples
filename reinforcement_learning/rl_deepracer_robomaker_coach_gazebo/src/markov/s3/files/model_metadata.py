'''This module implements s3 client for model metadata'''

import os
import logging
import json

from markov.log_handler.logger import Logger
from markov.log_handler.constants import (SIMAPP_EVENT_ERROR_CODE_500,
                                          SIMAPP_SIMULATION_WORKER_EXCEPTION)
from markov.log_handler.exception_handler import log_and_exit
from markov.architecture.constants import Input, NeuralNetwork
from markov.constants import SIMAPP_VERSION_2, SIMAPP_VERSION_1
from markov.s3.s3_client import S3Client

LOG = Logger(__name__, logging.INFO).get_logger()

class ModelMetadata():
    '''model metadata file upload, download, and parse
    '''
    def __init__(self, bucket, s3_key, region_name="us-east-1",
                 s3_endpoint_url=None,
                 local_path="./custom_files/agent/model_metadata.json",
                 max_retry_attempts=5, backoff_time_sec=1.0):
        '''Model metadata upload, download, and parse

        Args:
            bucket (str): S3 bucket string
            s3_key: (str): S3 key string.
            region_name (str): S3 region name
            local_path (str): file local path
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry

        '''
        # check s3 key and s3 bucket exist
        if not bucket or not s3_key:
            log_and_exit("model_metadata S3 key or bucket not available for S3. \
                         bucket: {}, key {}"
                         .format(bucket, s3_key),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
        self._bucket = bucket
        # Strip the s3://<bucket> from uri, if s3_key past in as uri
        self._s3_key = s3_key.replace('s3://{}/'.format(self._bucket), '')
        self._local_path = local_path
        self._local_dir = os.path.dirname(self._local_path)
        self._model_metadata = None
        self._s3_client = S3Client(region_name,
                                   s3_endpoint_url,
                                   max_retry_attempts,
                                   backoff_time_sec)

    @property
    def local_dir(self):
        '''return local dir of model metadata'''
        return self._local_dir

    @property
    def local_path(self):
        '''return local path of model metadata'''
        return self._local_path

    def get_model_metadata_info(self):
        '''retrive the model metadata info

        Returns:
            tuple (str, str, str): string of sensor, network, simapp_version

        '''
        # download model_metadata.json
        if not self._model_metadata:
            self._download()
            # after successfully download or use default model metadata, then parse
            self._model_metadata = self.parse_model_metadata(self._local_path)

        return self._model_metadata

    def persist(self, s3_kms_extra_args):
        '''upload local model_metadata.json into S3 bucket

        Args:
            s3_kms_extra_args (dict): s3 key management service extra argument

        '''

        # persist model metadata
        # if retry failed, s3_client upload_file will log and exit 500
        self._s3_client.upload_file(bucket=self._bucket,
                                    s3_key=self._s3_key,
                                    local_path=self._local_path,
                                    s3_kms_extra_args=s3_kms_extra_args)

    def _download(self):
        '''download model_metadata.json with retry from s3 bucket'''

        # check and make local directory
        if self._local_dir and not os.path.exists(self._local_dir):
            os.makedirs(self._local_dir)

        # download model metadata
        # if retry failed, each worker.py and download_params_and_roslaunch_agent.py
        # will handle 400 adn 500 separately
        self._s3_client.download_file(bucket=self._bucket,
                                      s3_key=self._s3_key,
                                      local_path=self._local_path)
        LOG.info("[s3] Successfully downloaded model metadata \
                 from s3 key {} to local {}.".format(self._s3_key, self._local_path))

    @staticmethod
    def parse_model_metadata(local_model_metadata_path):
        """parse model metadata give the local path

        Args:
            local_model_metadata_path (str): local model metadata string

        Returns:
            tuple (list, str, str): list of sensor, network, simapp_version

        """

        try:
            with open(local_model_metadata_path, "r") as json_file:
                data = json.load(json_file)
                # simapp_version 2.0+ should contain version as key in
                # model_metadata.json
                if 'action_space' not in data:
                    raise ValueError("no action space defined")
                if 'version' in data:
                    simapp_version = float(data['version'])
                    if simapp_version >= SIMAPP_VERSION_2:
                        sensor = data['sensor']
                    else:
                        sensor = [Input.OBSERVATION.value]
                else:
                    if 'sensor' in data:
                        sensor = data['sensor']
                        simapp_version = SIMAPP_VERSION_2
                    else:
                        sensor = [Input.OBSERVATION.value]
                        simapp_version = SIMAPP_VERSION_1
                if 'neural_network' in data:
                    network = data['neural_network']
                else:
                    network = NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value
            LOG.info("Sensor list %s, network %s, simapp_version %s", sensor, network, simapp_version)
            return sensor, network, simapp_version
        except ValueError as ex:
            raise ValueError('model_metadata ValueError: {}'.format(ex))
        except Exception as ex:
            raise Exception('Model metadata does not exist: {}'.format(ex))
