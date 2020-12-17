'''This module implements checkpoint file'''

import os
import io
import logging
import json
import time
import boto3
import botocore

from rl_coach.checkpoint import CheckpointStateFile
from rl_coach.data_stores.data_store import SyncFiles
from markov.log_handler.logger import Logger
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.constants import (SIMAPP_EVENT_ERROR_CODE_500,
                                          SIMAPP_SIMULATION_WORKER_EXCEPTION,
                                          SIMAPP_S3_DATA_STORE_EXCEPTION,
                                          SIMAPP_EVENT_ERROR_CODE_400)
from markov.s3.constants import (CHECKPOINT_POSTFIX_DIR,
                                 COACH_CHECKPOINT_POSTFIX,
                                 DEEPRACER_CHECKPOINT_KEY_POSTFIX,
                                 FINISHED_FILE_KEY_POSTFIX,
                                 LOCKFILE_KEY_POSTFIX,
                                 BEST_CHECKPOINT,
                                 LAST_CHECKPOINT)
from markov.s3.files.checkpoint_files.deepracer_checkpoint_json import DeepracerCheckpointJson
from markov.s3.files.checkpoint_files.rl_coach_checkpoint import RLCoachCheckpoint
from markov.s3.files.checkpoint_files.rl_coach_sync_file import RlCoachSyncFile
from markov.s3.files.checkpoint_files.tensorflow_model import TensorflowModel

LOG = Logger(__name__, logging.INFO).get_logger()


class Checkpoint():
    '''This class is a placeholder for RLCoachCheckpoint, DeepracerCheckpointJson,
    RlCoachSyncFile, TensorflowModel to handle all checkpoint related logic
    '''
    def __init__(self, bucket, s3_prefix, region_name="us-east-1",
                 s3_endpoint_url=None,
                 agent_name='agent', checkpoint_dir="./checkpoint",
                 max_retry_attempts=5, backoff_time_sec=1.0):
        '''This class is a placeholder for RLCoachCheckpoint, DeepracerCheckpointJson,
        RlCoachSyncFile, TensorflowModel to handle all checkpoint related logic

        Args:
            bucket (str): S3 bucket string
            s3_prefix (str): S3 prefix string
            region_name (str): S3 region name
            agent_name (str): agent name
            checkpoint_dir (str): root checkpoint directory
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry
        '''
        if not bucket or not s3_prefix:
            log_and_exit("checkpoint S3 prefix or bucket not available for S3. \
                         bucket: {}, prefix {}"
                         .format(bucket, s3_prefix),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
        self._agent_name = agent_name
        self._s3_dir = os.path.normpath(os.path.join(s3_prefix,
                                                     CHECKPOINT_POSTFIX_DIR))

        # rl coach checkpoint
        self._rl_coach_checkpoint = RLCoachCheckpoint(bucket=bucket,
                                                      s3_prefix=s3_prefix,
                                                      region_name=region_name,
                                                      s3_endpoint_url=s3_endpoint_url,
                                                      local_dir=os.path.join(checkpoint_dir,
                                                                             agent_name),
                                                      max_retry_attempts=max_retry_attempts,
                                                      backoff_time_sec=backoff_time_sec)

        # deepracer checkpoint json
        # do not retry on deepracer checkpoint because initially
        # it can do not exist.
        self._deepracer_checkpoint_json = \
            DeepracerCheckpointJson(bucket=bucket,
                                    s3_prefix=s3_prefix,
                                    region_name=region_name,
                                    s3_endpoint_url=s3_endpoint_url,
                                    local_dir=os.path.join(checkpoint_dir, agent_name),
                                    max_retry_attempts=0,
                                    backoff_time_sec=backoff_time_sec)

        # rl coach .finished
        self._syncfile_finished = RlCoachSyncFile(syncfile_type=SyncFiles.FINISHED.value,
                                                  bucket=bucket,
                                                  s3_prefix=s3_prefix,
                                                  region_name=region_name,
                                                  s3_endpoint_url=s3_endpoint_url,
                                                  local_dir=os.path.join(checkpoint_dir,
                                                                         agent_name),
                                                  max_retry_attempts=max_retry_attempts,
                                                  backoff_time_sec=backoff_time_sec)

        # rl coach .lock: global lock for all agent located at checkpoint directory
        self._syncfile_lock = RlCoachSyncFile(syncfile_type=SyncFiles.LOCKFILE.value,
                                              bucket=bucket,
                                              s3_prefix=s3_prefix,
                                              region_name=region_name,
                                              s3_endpoint_url=s3_endpoint_url,
                                              local_dir=checkpoint_dir,
                                              max_retry_attempts=max_retry_attempts,
                                              backoff_time_sec=backoff_time_sec)

        # rl coach .ready
        self._syncfile_ready = RlCoachSyncFile(syncfile_type=SyncFiles.TRAINER_READY.value,
                                               bucket=bucket,
                                               s3_prefix=s3_prefix,
                                               region_name=region_name,
                                               s3_endpoint_url=s3_endpoint_url,
                                               local_dir=os.path.join(checkpoint_dir,
                                                                      agent_name),
                                               max_retry_attempts=max_retry_attempts,
                                               backoff_time_sec=backoff_time_sec)

        # tensorflow .ckpt files
        self._tensorflow_model = TensorflowModel(bucket=bucket,
                                                 s3_prefix=s3_prefix,
                                                 region_name=region_name,
                                                 s3_endpoint_url=s3_endpoint_url,
                                                 local_dir=os.path.join(checkpoint_dir,
                                                                        agent_name),
                                                 max_retry_attempts=max_retry_attempts,
                                                 backoff_time_sec=backoff_time_sec)

    @property
    def agent_name(self):
        '''return agent name in str
        '''
        return self._agent_name

    @property
    def s3_dir(self):
        '''return s3 directory in str
        '''
        return self._s3_dir

    @property
    def rl_coach_checkpoint(self):
        '''return RLCoachCheckpoint class instance
        '''
        return self._rl_coach_checkpoint

    @property
    def deepracer_checkpoint_json(self):
        '''return DeepracerCheckpointJson class instance
        '''
        return self._deepracer_checkpoint_json

    @property
    def syncfile_finished(self):
        '''return RlCoachSyncFile .finished file class instance
        '''
        return self._syncfile_finished

    @property
    def syncfile_lock(self):
        '''return RlCoachSyncFile .lock file class instance
        '''
        return self._syncfile_lock

    @property
    def syncfile_ready(self):
        '''return RlCoachSyncFile .ready file class instance
        '''
        return self._syncfile_ready

    @property
    def tensorflow_model(self):
        '''return TensorflowModel class instance
        '''
        return self._tensorflow_model
