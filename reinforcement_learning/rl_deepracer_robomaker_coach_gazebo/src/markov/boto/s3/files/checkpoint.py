"""This module implements checkpoint file"""

import io
import json
import logging
import os
import time

import boto3
import botocore
from markov.boto.s3.constants import (
    BEST_CHECKPOINT,
    CHECKPOINT_POSTFIX_DIR,
    COACH_CHECKPOINT_POSTFIX,
    DEEPRACER_CHECKPOINT_KEY_POSTFIX,
    FINISHED_FILE_KEY_POSTFIX,
    LAST_CHECKPOINT,
    LOCKFILE_KEY_POSTFIX,
)
from markov.boto.s3.files.checkpoint_files.deepracer_checkpoint_json import DeepracerCheckpointJson
from markov.boto.s3.files.checkpoint_files.rl_coach_checkpoint import RLCoachCheckpoint
from markov.boto.s3.files.checkpoint_files.rl_coach_sync_file import RlCoachSyncFile
from markov.boto.s3.files.checkpoint_files.tensorflow_model import TensorflowModel
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_S3_DATA_STORE_EXCEPTION,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from rl_coach.checkpoint import CheckpointStateFile
from rl_coach.data_stores.data_store import SyncFiles

LOG = Logger(__name__, logging.INFO).get_logger()


class Checkpoint:
    """This class is a placeholder for RLCoachCheckpoint, DeepracerCheckpointJson,
    RlCoachSyncFile, TensorflowModel to handle all checkpoint related logic
    """

    def __init__(
        self,
        bucket,
        s3_prefix,
        region_name="us-east-1",
        agent_name="agent",
        checkpoint_dir="./checkpoint",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
        output_head_format="main_level/{}/main/online/network_1/ppo_head_0/policy",
        log_and_cont: bool = False,
    ):
        """This class is a placeholder for RLCoachCheckpoint, DeepracerCheckpointJson,
        RlCoachSyncFile, TensorflowModel to handle all checkpoint related logic

        Args:
            bucket (str): S3 bucket string.
            s3_prefix (str): S3 prefix string.
            region_name (str): S3 region name.
                               Defaults to 'us-east-1'.
            agent_name (str): Agent name.
                              Defaults to 'agent'.
            checkpoint_dir (str, optional): Local file directory.
                                            Defaults to './checkpoint'.
            max_retry_attempts (int, optional): Maximum number of retry attempts for S3 download/upload.
                                                Defaults to 5.
            backoff_time_sec (float, optional): Backoff second between each retry.
                                                Defaults to 1.0.
            output_head_format (str): output head format for the specific algorithm and action space
                                      which will be used to store the frozen graph
            log_and_cont (bool, optional): Log the error and continue with the flow.
                                           Defaults to False.
        """
        if not bucket or not s3_prefix:
            log_and_exit(
                "checkpoint S3 prefix or bucket not available for S3. \
                         bucket: {}, prefix {}".format(
                    bucket, s3_prefix
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._agent_name = agent_name
        self._s3_dir = os.path.normpath(os.path.join(s3_prefix, CHECKPOINT_POSTFIX_DIR))

        # rl coach checkpoint
        self._rl_coach_checkpoint = RLCoachCheckpoint(
            bucket=bucket,
            s3_prefix=s3_prefix,
            region_name=region_name,
            local_dir=os.path.join(checkpoint_dir, agent_name),
            max_retry_attempts=max_retry_attempts,
            backoff_time_sec=backoff_time_sec,
            log_and_cont=log_and_cont,
        )

        # deepracer checkpoint json
        # do not retry on deepracer checkpoint because initially
        # it can do not exist.
        self._deepracer_checkpoint_json = DeepracerCheckpointJson(
            bucket=bucket,
            s3_prefix=s3_prefix,
            region_name=region_name,
            local_dir=os.path.join(checkpoint_dir, agent_name),
            max_retry_attempts=0,
            backoff_time_sec=backoff_time_sec,
            log_and_cont=log_and_cont,
        )

        # rl coach .finished
        self._syncfile_finished = RlCoachSyncFile(
            syncfile_type=SyncFiles.FINISHED.value,
            bucket=bucket,
            s3_prefix=s3_prefix,
            region_name=region_name,
            local_dir=os.path.join(checkpoint_dir, agent_name),
            max_retry_attempts=max_retry_attempts,
            backoff_time_sec=backoff_time_sec,
        )

        # rl coach .lock: global lock for all agent located at checkpoint directory
        self._syncfile_lock = RlCoachSyncFile(
            syncfile_type=SyncFiles.LOCKFILE.value,
            bucket=bucket,
            s3_prefix=s3_prefix,
            region_name=region_name,
            local_dir=checkpoint_dir,
            max_retry_attempts=max_retry_attempts,
            backoff_time_sec=backoff_time_sec,
        )

        # rl coach .ready
        self._syncfile_ready = RlCoachSyncFile(
            syncfile_type=SyncFiles.TRAINER_READY.value,
            bucket=bucket,
            s3_prefix=s3_prefix,
            region_name=region_name,
            local_dir=os.path.join(checkpoint_dir, agent_name),
            max_retry_attempts=max_retry_attempts,
            backoff_time_sec=backoff_time_sec,
        )

        # tensorflow .ckpt files
        self._tensorflow_model = TensorflowModel(
            bucket=bucket,
            s3_prefix=s3_prefix,
            region_name=region_name,
            local_dir=os.path.join(checkpoint_dir, agent_name),
            max_retry_attempts=max_retry_attempts,
            backoff_time_sec=backoff_time_sec,
            output_head_format=output_head_format,
        )

    @property
    def agent_name(self):
        """return agent name in str"""
        return self._agent_name

    @property
    def s3_dir(self):
        """return s3 directory in str"""
        return self._s3_dir

    @property
    def rl_coach_checkpoint(self):
        """return RLCoachCheckpoint class instance"""
        return self._rl_coach_checkpoint

    @property
    def deepracer_checkpoint_json(self):
        """return DeepracerCheckpointJson class instance"""
        return self._deepracer_checkpoint_json

    @property
    def syncfile_finished(self):
        """return RlCoachSyncFile .finished file class instance"""
        return self._syncfile_finished

    @property
    def syncfile_lock(self):
        """return RlCoachSyncFile .lock file class instance"""
        return self._syncfile_lock

    @property
    def syncfile_ready(self):
        """return RlCoachSyncFile .ready file class instance"""
        return self._syncfile_ready

    @property
    def tensorflow_model(self):
        """return TensorflowModel class instance"""
        return self._tensorflow_model
