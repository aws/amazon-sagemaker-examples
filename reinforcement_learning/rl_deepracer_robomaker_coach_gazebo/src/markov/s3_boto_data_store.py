import io
import logging
import os
import queue
import time
from typing import Dict

import boto3
import botocore
import tensorflow as tf
from markov.boto.s3.files.checkpoint import Checkpoint
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_EVENT_SYSTEM_ERROR,
    SIMAPP_EVENT_USER_ERROR,
    SIMAPP_S3_DATA_STORE_EXCEPTION,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericNonFatalException
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.multi_agent_coach.multi_agent_graph_manager import MultiAgentGraphManager
from markov.utils import get_s3_kms_extra_args
from rl_coach.checkpoint import CheckpointStateFile, _filter_checkpoint_files
from rl_coach.data_stores.data_store import DataStore, DataStoreParameters, SyncFiles

LOG = Logger(__name__, logging.INFO).get_logger()

SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND = 1
SLEEP_SECONDS = 10  # sleep 10 seconds


class S3BotoDataStoreParameters(DataStoreParameters):
    def __init__(self, checkpoint_dict: Dict[str, Checkpoint]):
        super().__init__("s3", "", "")
        self.checkpoint_dict = checkpoint_dict


class S3BotoDataStore(DataStore):
    #! TODO remove ignore_lock after refactoring this class
    def __init__(
        self,
        params: S3BotoDataStoreParameters,
        graph_manager: MultiAgentGraphManager,
        ignore_lock: bool = False,
        log_and_cont: bool = False,
    ):
        """Initialize a DataStore that works with aws s3 storage using boto interface.

        Args:
            params (S3BotoDataStoreParameters): The parameters for s3 boto data store.
            graph_manager (MultiAgentGraphManager): The Graph Manager for the current tf session.
            ignore_lock (bool, optional): Ignore the lock. Defaults to False.
            log_and_cont (bool, optional): Log the error and continue with the flow.
                                           Defaults to False.
        """
        self.params = params
        if not graph_manager:
            log_and_exit(
                "None type for graph manager",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self.graph_manager = graph_manager
        self.ignore_lock = ignore_lock
        self.syncfile_lock = (list(self.params.checkpoint_dict.values())[0]).syncfile_lock
        self._log_and_cont = log_and_cont

    def deploy(self) -> bool:
        return True

    def get_info(self, agent_name):
        return "s3://{}".format(self.params.checkpoint_dict[agent_name].s3_dir)

    def undeploy(self) -> bool:
        return True

    def upload_finished_file(self):
        for _, checkpoint in self.params.checkpoint_dict.items():
            checkpoint.syncfile_finished.persist(s3_kms_extra_args=get_s3_kms_extra_args())

    def save_to_store(self):
        try:
            # remove lock file if it exists
            self.syncfile_lock.delete()
            # acquire lock
            self.syncfile_lock.persist(s3_kms_extra_args=get_s3_kms_extra_args())
            for _, checkpoint in self.params.checkpoint_dict.items():
                # upload tensorflow models, tensorflow frozen graph, and rl coach checkpoint
                self._save_tf_model_to_store(checkpoint)
            # release lock by delete it
            self.syncfile_lock.delete()
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to upload checkpoint",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in uploading checkpoint: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def signal_ready(self):
        """upload rl coach .ready file"""
        try:
            # remove lock file if it exists
            self.syncfile_lock.delete()
            # acquire lock
            self.syncfile_lock.persist(s3_kms_extra_args=get_s3_kms_extra_args())
            for _, checkpoint in self.params.checkpoint_dict.items():
                # upload .ready
                checkpoint.syncfile_ready.persist(s3_kms_extra_args=get_s3_kms_extra_args())
            # release lock by delete it
            self.syncfile_lock.delete()
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to upload .ready",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in uploading .ready file: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def flush_finished(self):
        """upload rl coach .finished file"""
        try:
            # remove lock file if it exists
            self.syncfile_lock.delete()
            # acquire lock
            self.syncfile_lock.persist(s3_kms_extra_args=get_s3_kms_extra_args())
            for _, checkpoint in self.params.checkpoint_dict.items():
                # upload .finished
                checkpoint.syncfile_finished.persist(s3_kms_extra_args=get_s3_kms_extra_args())

            # release lock by delete it
            self.syncfile_lock.delete()
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to upload .finished",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in uploading .finished file: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def _save_tf_model_to_store(self, checkpoint):
        # rl coach .coach_checkpoint state file
        state_file = checkpoint.rl_coach_checkpoint.coach_checkpoint_state_file

        # upload tensorflow models
        checkpoint.tensorflow_model.persist(
            coach_checkpoint_state_file=state_file, s3_kms_extra_args=get_s3_kms_extra_args()
        )

        # persist rl coach checkpoint
        checkpoint.rl_coach_checkpoint.persist(s3_kms_extra_args=get_s3_kms_extra_args())

        # Upload the frozen graph which is used for deployment
        if self.graph_manager:
            checkpoint.tensorflow_model.persist_tensorflow_frozen_graph(
                agent_name=checkpoint.agent_name,
                graph_manager=self.graph_manager,
                coach_checkpoint_state_file=state_file,
                best_checkpoint_number=checkpoint.deepracer_checkpoint_json.get_deepracer_best_checkpoint_number(),
                last_checkpoint_number=checkpoint.deepracer_checkpoint_json.get_deepracer_last_checkpoint_number(),
                s3_kms_extra_args=get_s3_kms_extra_args(),
            )

        # Clean up old checkpoints
        checkpoint.tensorflow_model.delete(
            coach_checkpoint_state_file=state_file,
            best_checkpoint=checkpoint.deepracer_checkpoint_json.get_deepracer_best_checkpoint(),
        )

    def get_coach_checkpoint_number(self, agent_key):
        try:
            # If there is a lock file return -1 since it means the trainer has the lock
            response = self.syncfile_lock.list()
            chkpoint_num = -1
            if "Contents" not in response:
                # download rl coach .coach_checkpoint file
                self.params.checkpoint_dict[agent_key].rl_coach_checkpoint.get()
                # read .coach_checkpoint file after download
                checkpoint_state = self.params.checkpoint_dict[
                    agent_key
                ].rl_coach_checkpoint.coach_checkpoint_state_file.read()
                if checkpoint_state is not None:
                    chkpoint_num = checkpoint_state.num
            return chkpoint_num
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to download checkpoint",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in downloading checkpoint: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def load_from_store(self, expected_checkpoint_number=-1):
        """download tf model, rl coach .coach_checkpoint, .finished, .ready file from s3

        Args:
            expected_checkpoint_number (int): for training, rollout worker will expect the latest
            file for eval, validation, expected_checkpoint_number will always be -1
            to make sure last/best tf model can be downloaded
        """
        try:
            for _, checkpoint in self.params.checkpoint_dict.items():
                while True:
                    # load tf models and rl coach .coach_checkpoint from s3 store
                    if not self._load_tf_model_from_store(
                        checkpoint=checkpoint, expected_checkpoint_number=expected_checkpoint_number
                    ):
                        continue
                    # load .finished from s3 store
                    self._load_syncfile_from_store(sync_file=checkpoint.syncfile_finished)
                    # load .ready from s3 store
                    self._load_syncfile_from_store(sync_file=checkpoint.syncfile_ready)
                    break
        except botocore.exceptions.ClientError as ex:
            if self._log_and_cont:
                error_msg = "[s3] ClientError: Unable to download checkpoint. {}".format(ex)
                raise GenericNonFatalException(
                    error_msg=error_msg,
                    error_code=SIMAPP_EVENT_ERROR_CODE_400,
                    error_name=SIMAPP_EVENT_USER_ERROR,
                )
            log_and_exit(
                "Unable to download checkpoint",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            if self._log_and_cont:
                error_msg = "[s3] SystemError: Unable to download checkpoint. {}".format(ex)
                raise GenericNonFatalException(
                    error_msg=error_msg,
                    error_code=SIMAPP_EVENT_ERROR_CODE_500,
                    error_name=SIMAPP_EVENT_SYSTEM_ERROR,
                )
            log_and_exit(
                "Exception in downloading checkpoint: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def load_trainer_ready_from_store(self):
        try:
            for _, checkpoint in self.params.checkpoint_dict.items():
                # load .ready from s3 store
                self._load_syncfile_from_store(sync_file=checkpoint.syncfile_ready)
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to download .ready",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in downloading .ready: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def _load_syncfile_from_store(self, sync_file):
        """download a specific sync file from s3 if exist

        Args:
            sync_file (RlCoachSyncFile): RlCoachSyncFile class instance
        """
        # list rl coach sync file
        response = sync_file.list()
        if "Contents" in response:
            try:
                # download rl coach sync file
                sync_file.download()
            except Exception:
                pass

    def _load_tf_model_from_store(self, checkpoint, expected_checkpoint_number):
        """load tf models and rl coach .coach_checkpoint from s3 store

        Args:
            checkpoint (Checkpoint): Checkpoint class instance
            expected_checkpoint_number (int): for training, rollout worker will expect the latest
            file for eval, validation, expected_checkpoint_number will always be -1
            to make sure last/best tf model can be downloaded

        Returns:
            bool: True if load tf model from store succeed. Otherwise, False
        """
        # list rl coach .lock
        response = self.syncfile_lock.list()
        if "Contents" not in response or self.ignore_lock:
            try:
                # download rl coach checkpoint
                checkpoint.rl_coach_checkpoint.get()
            except botocore.exceptions.ClientError:
                if self.ignore_lock:
                    log_and_exit(
                        "Checkpoint not found",
                        SIMAPP_S3_DATA_STORE_EXCEPTION,
                        SIMAPP_EVENT_ERROR_CODE_400,
                    )
                time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                return False
            except Exception:
                if self.ignore_lock:
                    log_and_exit(
                        "Checkpoint not found",
                        SIMAPP_S3_DATA_STORE_EXCEPTION,
                        SIMAPP_EVENT_ERROR_CODE_500,
                    )
                time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                return False
        else:
            time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
            return False

        checkpoint_state = checkpoint.rl_coach_checkpoint.coach_checkpoint_state_file.read()
        if checkpoint_state is not None:
            # if we get a checkpoint that is older that the expected checkpoint, we wait for
            #  the new checkpoint to arrive.
            if checkpoint_state.num < expected_checkpoint_number:
                time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                return False
            # download tensorflow models
            checkpoint.tensorflow_model.get(
                checkpoint.rl_coach_checkpoint.coach_checkpoint_state_file
            )
        return True

    def wait_for_checkpoints(self, num_retry=10):
        """
        block until there is a checkpoint in all of the checkpoint_dirs.

        Args:
            num_retry (int, optional): The number of retries to download the checkpoints.
                                       The total wait time is num_retry * SLEEP_SECONDS.
                                       Defaults to 10.
        """

        for _ in range(num_retry):
            self.load_from_store()
            all_agent_checkpoint_copied = all(
                [
                    checkpoint.rl_coach_checkpoint.coach_checkpoint_state_file.read() is not None
                    for _, checkpoint in self.params.checkpoint_dict.items()
                ]
            )
            if all_agent_checkpoint_copied:
                return
            time.sleep(SLEEP_SECONDS)

        # one last time
        all_agent_checkpoint_copied = all(
            [
                checkpoint.rl_coach_checkpoint.coach_checkpoint_state_file.read() is not None
                for _, checkpoint in self.params.checkpoint_dict.items()
            ]
        )
        if all_agent_checkpoint_copied:
            return
        if self._log_and_cont:
            error_msg = "[s3] Checkpoint never found, waited {} seconds.".format(timeout)
            raise GenericNonFatalException(
                error_msg=error_msg,
                error_code=SIMAPP_EVENT_ERROR_CODE_500,
                error_name=SIMAPP_EVENT_SYSTEM_ERROR,
            )
        log_and_exit(
            "Checkpoint never found, waited {} seconds.".format(timeout),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )

    def wait_for_trainer_ready(self, num_retry=30):
        """
        Try to download the .ready file which signals the trainer is ready.
        Block until the file is found. Exit if it's never found.

        Args:
            num_retry (int, optional): The number of retries to download the ready file.
                                       The total wait time is num_retry * SLEEP_SECONDS.
                                       Defaults to 15.
        """
        for _ in range(num_retry):
            self.load_trainer_ready_from_store()
            all_agent_ready_copied = all(
                [
                    "Contents" in checkpoint.syncfile_ready.list()
                    for _, checkpoint in self.params.checkpoint_dict.items()
                ]
            )
            if all_agent_ready_copied:
                return
            time.sleep(SLEEP_SECONDS)

        # one last time
        all_agent_ready_copied = all(
            [
                "Contents" in checkpoint.syncfile_ready.list()
                for _, checkpoint in self.params.checkpoint_dict.items()
            ]
        )
        if all_agent_ready_copied:
            return

        log_and_exit(
            "Ready never found, waited {} seconds.".format(num_retry * SLEEP_SECONDS),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )

    def modify_checkpoint_variables(self):
        for agent_name, checkpoint in self.params.checkpoint_dict.items():
            checkpoint.tensorflow_model.rename(
                coach_checkpoint_state_file=checkpoint.rl_coach_checkpoint.coach_checkpoint_state_file,
                agent_name=agent_name,
            )
