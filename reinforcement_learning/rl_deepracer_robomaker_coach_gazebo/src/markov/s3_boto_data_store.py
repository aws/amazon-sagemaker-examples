import io
import logging
import os
import time
import queue
from typing import Dict
import botocore
import boto3

from rl_coach.checkpoint import CheckpointStateFile, _filter_checkpoint_files
from rl_coach.data_stores.data_store import DataStore, DataStoreParameters, SyncFiles
from markov.multi_agent_coach.multi_agent_graph_manager import MultiAgentGraphManager
from markov.utils import log_and_exit, Logger, get_best_checkpoint, get_boto_config, \
                         copy_best_frozen_model_to_sm_output_dir, \
                         SIMAPP_EVENT_ERROR_CODE_500, SIMAPP_EVENT_ERROR_CODE_400, \
                         SIMAPP_S3_DATA_STORE_EXCEPTION
import tensorflow as tf

LOG = Logger(__name__, logging.INFO).get_logger()

# The number of models to keep in S3
#! TODO discuss with product team if this number should be configurable
NUM_MODELS_TO_KEEP = 50

SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND = 1
SM_MODEL_OUTPUT_DIR = os.environ.get("ALGO_MODEL_DIR", "/opt/ml/model")
# Temporary folder where the model_{}.pb for best_checkpoint_iteration, last_checkpoint_iteration
# and other iterations > last_checkpoint_iteration are stored
SM_MODEL_PB_TEMP_FOLDER = './frozen_models'


class S3BotoDataStoreParameters(DataStoreParameters):
    def __init__(self, aws_region: str = "us-west-2", bucket_names: Dict[str, str] = {"agent": None},
                 s3_folders: Dict[str, str] = {"agent": None},
                 base_checkpoint_dir: str = None):
        super().__init__("s3", "", "")
        self.aws_region = aws_region
        self.buckets = bucket_names
        self.s3_folders = s3_folders
        self.base_checkpoint_dir = base_checkpoint_dir


class S3BotoDataStore(DataStore):
    #! TODO remove ignore_lock after refactoring this class
    def __init__(self, params: S3BotoDataStoreParameters, graph_manager: MultiAgentGraphManager,
                 ignore_lock: bool = False):
        self.params = params
        self.key_prefixes = dict()
        self.ip_data_keys = dict()
        self.ip_done_keys = dict()
        self.preset_data_keys = dict()
        self.delete_queues = dict()
        for agent_key, s3_folder in self.params.s3_folders.items():
            self.key_prefixes[agent_key] = os.path.join(s3_folder, "model")
            self.ip_data_keys[agent_key] = os.path.join(s3_folder, "ip/ip.json")
            self.ip_done_keys[agent_key] = os.path.join(s3_folder, "ip/done")
            self.preset_data_keys[agent_key] = os.path.join(s3_folder, "presets/preset.py")
            self.delete_queues[agent_key] = queue.Queue()
        if not graph_manager:
            log_and_exit("None type for graph manager",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

        self.graph_manager = graph_manager
        self.ignore_lock = ignore_lock

    def _get_s3_key(self, key, agent_key):
        return os.path.join(self.key_prefixes[agent_key], key)

    def _get_client(self):
        session = boto3.session.Session()
        return session.client('s3', region_name=self.params.aws_region,
                              config=get_boto_config())

    def deploy(self) -> bool:
        return True

    def get_info(self, agent_key):
        return "s3://{}/{}".format(self.params.buckets[agent_key], self.params.s3_folder[agent_key])

    def undeploy(self) -> bool:
        return True

    def upload_finished_file(self):
        try:
            s3_client = self._get_client()
            for agent_key, bucket in self.params.buckets.items():
                s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                         Bucket=bucket,
                                         Key=self._get_s3_key(SyncFiles.FINISHED.value, agent_key))
        except botocore.exceptions.ClientError:
            log_and_exit("Unable to upload finish file",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception:
            log_and_exit("Unable to upload finish file",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def save_to_store(self):
        try:
            s3_client = self._get_client()
            base_checkpoint_dir = self.params.base_checkpoint_dir
            for agent_key, bucket in self.params.buckets.items():
                # remove lock file if it exists
                s3_client.delete_object(Bucket=bucket, Key=self._get_s3_key(SyncFiles.LOCKFILE.value, agent_key))

                # acquire lock
                s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                         Bucket=bucket,
                                         Key=self._get_s3_key(SyncFiles.LOCKFILE.value, agent_key))

                checkpoint_dir = base_checkpoint_dir if len(self.graph_manager.agents_params) == 1 else \
                    os.path.join(base_checkpoint_dir, agent_key)

                state_file = CheckpointStateFile(os.path.abspath(checkpoint_dir))
                ckpt_state = None
                check_point_key_list = []
                if state_file.exists():
                    ckpt_state = state_file.read()
                    checkpoint_file = None
                    num_files_uploaded = 0
                    for root, _, files in os.walk(checkpoint_dir):
                        for filename in files:
                            if filename == CheckpointStateFile.checkpoint_state_filename:
                                checkpoint_file = (root, filename)
                                continue
                            if filename.startswith(ckpt_state.name):
                                abs_name = os.path.abspath(os.path.join(root, filename))
                                rel_name = os.path.relpath(abs_name, checkpoint_dir)
                                s3_client.upload_file(Filename=abs_name,
                                                      Bucket=bucket,
                                                      Key=self._get_s3_key(rel_name, agent_key))
                                check_point_key_list.append(self._get_s3_key(rel_name, agent_key))
                                num_files_uploaded += 1
                    LOG.info("Uploaded %s files for checkpoint %s", num_files_uploaded, ckpt_state.num)
                    if check_point_key_list:
                        self.delete_queues[agent_key].put(check_point_key_list)

                    abs_name = os.path.abspath(os.path.join(checkpoint_file[0], checkpoint_file[1]))
                    rel_name = os.path.relpath(abs_name, checkpoint_dir)
                    s3_client.upload_file(Filename=abs_name,
                                          Bucket=bucket,
                                          Key=self._get_s3_key(rel_name, agent_key))

                # upload Finished if present
                if os.path.exists(os.path.join(checkpoint_dir, SyncFiles.FINISHED.value)):
                    s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                             Bucket=bucket,
                                             Key=self._get_s3_key(SyncFiles.FINISHED.value, agent_key))

                # upload Ready if present
                if os.path.exists(os.path.join(checkpoint_dir, SyncFiles.TRAINER_READY.value)):
                    s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                             Bucket=bucket,
                                             Key=self._get_s3_key(SyncFiles.TRAINER_READY.value, agent_key))

                # release lock
                s3_client.delete_object(Bucket=bucket,
                                        Key=self._get_s3_key(SyncFiles.LOCKFILE.value, agent_key))

                # Upload the frozen graph which is used for deployment
                if self.graph_manager:
                    # checkpoint state is always present for the checkpoint dir passed.
                    # We make same assumption while we get the best checkpoint in s3_metrics
                    checkpoint_num = ckpt_state.num
                    self.write_frozen_graph(self.graph_manager, agent_key, checkpoint_num)
                    frozen_name = "model_{}.pb".format(checkpoint_num)
                    frozen_graph_fpath = os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_key,
                                                      frozen_name)
                    frozen_graph_s3_name = frozen_name if len(self.graph_manager.agents_params) == 1 \
                        else os.path.join(agent_key, frozen_name)
                    # upload the model_<ID>.pb to S3.
                    s3_client.upload_file(Filename=frozen_graph_fpath,
                                          Bucket=bucket,
                                          Key=self._get_s3_key(frozen_graph_s3_name, agent_key))
                    LOG.info("saved intermediate frozen graph: %s", self._get_s3_key(frozen_graph_s3_name, agent_key))

                    # Copy the best checkpoint to the SM_MODEL_OUTPUT_DIR
                    copy_best_frozen_model_to_sm_output_dir(bucket,
                                                            self.params.s3_folders[agent_key],
                                                            self.params.aws_region,
                                                            os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_key),
                                                            os.path.join(SM_MODEL_OUTPUT_DIR, agent_key))

                # Clean up old checkpoints
                if ckpt_state and self.delete_queues[agent_key].qsize() > NUM_MODELS_TO_KEEP:
                    best_checkpoint = get_best_checkpoint(bucket,
                                                          self.params.s3_folders[agent_key],
                                                          self.params.aws_region)
                    while self.delete_queues[agent_key].qsize() > NUM_MODELS_TO_KEEP:
                        key_list = self.delete_queues[agent_key].get()
                        if best_checkpoint and all(list(map(lambda file_name: best_checkpoint in file_name,
                                                            [os.path.split(file)[-1] for file in key_list]))):
                            self.delete_queues[agent_key].put(key_list)
                        else:
                            delete_iteration_ids = set()
                            for key in key_list:
                                s3_client.delete_object(Bucket=bucket, Key=key)
                                # Get the name of the file in the checkpoint directory that has to be deleted
                                # and extract the iteration id out of the name
                                file_in_checkpoint_dir = os.path.split(key)[-1]
                                if len(file_in_checkpoint_dir.split("_Step")) > 0:
                                    delete_iteration_ids.add(file_in_checkpoint_dir.split("_Step")[0])
                            LOG.info("Deleting the frozen models in s3 for the iterations: %s",
                                     delete_iteration_ids)
                            # Delete the model_{}.pb files from the s3 bucket for the previous iterations
                            for iteration_id in list(delete_iteration_ids):
                                frozen_name = "model_{}.pb".format(iteration_id)
                                frozen_graph_s3_name = frozen_name if len(self.graph_manager.agents_params) == 1 \
                                    else os.path.join(agent_key, frozen_name)
                                s3_client.delete_object(Bucket=bucket,
                                                        Key=self._get_s3_key(frozen_graph_s3_name, agent_key))
        except botocore.exceptions.ClientError:
            log_and_exit("Unable to upload checkpoint",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception:
            log_and_exit("Unable to upload checkpoint",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def write_frozen_graph(self, graph_manager, agent_name, iteration_id):
        """Write the frozen graph to the temporary folder with a name model_{}.pb for the iteration_id passed
        Args:
            graph_manager (MultiAgentGraphManager): MultiAgentGraphManager object
            agent_name (str): Name of the agent
            iteration_id (int): Iteration id for which we are saving the model_{}.pb
        """
        if not os.path.exists(os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_name)):
            os.makedirs(os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_name))
        if not os.path.exists(os.path.join(SM_MODEL_OUTPUT_DIR, agent_name)):
            os.makedirs(os.path.join(SM_MODEL_OUTPUT_DIR, agent_name))
        output_head = ['main_level/{}/main/online/network_1/ppo_head_0/policy'.format(agent_name)]
        frozen = tf.graph_util.convert_variables_to_constants(graph_manager.sess[agent_name],
                                                              graph_manager.sess[agent_name].graph_def, output_head)
        tf.train.write_graph(frozen, os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_name),
                             'model_{}.pb'.format(iteration_id), as_text=False)

    def get_chkpoint_num(self, agent_key):
        try:
            s3_client = self._get_client()
            # If there is a lock file return -1 since it means the trainer has the lock
            response = s3_client.list_objects_v2(Bucket=self.params.buckets[agent_key],
                                                 Prefix=self._get_s3_key(SyncFiles.LOCKFILE.value, agent_key))
            chkpoint_num = -1
            if "Contents" not in response:
                base_checkpoint_dir = self.params.base_checkpoint_dir
                checkpoint_dir = base_checkpoint_dir if len(self.graph_manager.agents_params) == 1 else os.path.join(base_checkpoint_dir, agent_key)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                state_file = CheckpointStateFile(os.path.abspath(checkpoint_dir))
                s3_client.download_file(Bucket=self.params.buckets[agent_key],
                                        Key=self._get_s3_key(state_file.filename, agent_key),
                                        Filename=state_file.path)
                checkpoint_state = state_file.read()
                if checkpoint_state is not None:
                    chkpoint_num = checkpoint_state.num
            return chkpoint_num
        except botocore.exceptions.ClientError:
            log_and_exit("Unable to download checkpoint",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception:
            log_and_exit("Unable to download checkpoint",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def load_from_store(self, expected_checkpoint_number=-1):
        try:
            s3_client = self._get_client()
            base_checkpoint_dir = self.params.base_checkpoint_dir
            for agent_key, bucket in self.params.buckets.items():
                checkpoint_dir = base_checkpoint_dir if len(self.graph_manager.agents_params) == 1 else os.path.join(base_checkpoint_dir, agent_key)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                while True:
                    s3_client = self._get_client()
                    state_file = CheckpointStateFile(os.path.abspath(checkpoint_dir))

                    # wait until lock is removed
                    response = s3_client.list_objects_v2(Bucket=bucket,
                                                         Prefix=self._get_s3_key(SyncFiles.LOCKFILE.value, agent_key))
                    if "Contents" not in response or self.ignore_lock:
                        try:
                            checkpoint_file_path = os.path.abspath(os.path.join(checkpoint_dir,
                                                                                state_file.path))
                            # fetch checkpoint state file from S3
                            s3_client.download_file(Bucket=bucket,
                                                    Key=self._get_s3_key(state_file.filename, agent_key),
                                                    Filename=checkpoint_file_path)
                        except botocore.exceptions.ClientError:
                            if self.ignore_lock:
                                log_and_exit("Checkpoint not found",
                                             SIMAPP_S3_DATA_STORE_EXCEPTION,
                                             SIMAPP_EVENT_ERROR_CODE_400)
                            time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                            continue
                        except Exception:
                            if self.ignore_lock:
                                log_and_exit("Checkpoint not found",
                                             SIMAPP_S3_DATA_STORE_EXCEPTION,
                                             SIMAPP_EVENT_ERROR_CODE_500)
                            time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                            continue
                    else:
                        time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                        continue

                    # check if there's a Finished file
                    response = s3_client.list_objects_v2(Bucket=bucket,
                                                         Prefix=self._get_s3_key(SyncFiles.FINISHED.value, agent_key))
                    if "Contents" in response:
                        try:
                            finished_file_path = os.path.abspath(os.path.join(checkpoint_dir,
                                                                              SyncFiles.FINISHED.value))
                            s3_client.download_file(Bucket=bucket,
                                                    Key=self._get_s3_key(SyncFiles.FINISHED.value, agent_key),
                                                    Filename=finished_file_path)
                        except Exception:
                            pass

                    # check if there's a Ready file
                    response = s3_client.list_objects_v2(Bucket=bucket,
                                                         Prefix=self._get_s3_key(SyncFiles.TRAINER_READY.value, agent_key))
                    if "Contents" in response:
                        try:
                            ready_file_path = os.path.abspath(os.path.join(checkpoint_dir,
                                                                           SyncFiles.TRAINER_READY.value))
                            s3_client.download_file(Bucket=bucket,
                                                    Key=self._get_s3_key(SyncFiles.TRAINER_READY.value, agent_key),
                                                    Filename=ready_file_path)
                        except Exception:
                            pass

                    checkpoint_state = state_file.read()
                    if checkpoint_state is not None:

                        # if we get a checkpoint that is older that the expected checkpoint, we wait for
                        #  the new checkpoint to arrive.

                        if checkpoint_state.num < expected_checkpoint_number:
                            time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                            continue

                        response = s3_client.list_objects_v2(Bucket=bucket,
                                                             Prefix=self._get_s3_key("", agent_key))
                        if "Contents" in response:
                            # Check to see if the desired checkpoint is in the bucket
                            has_chkpnt = any(list(map(lambda obj: os.path.split(obj['Key'])[1].\
                                                                startswith(checkpoint_state.name),
                                                      response['Contents'])))
                            for obj in response["Contents"]:
                                full_key_prefix = os.path.normpath(self.key_prefixes[agent_key]) + "/"
                                filename = os.path.abspath(os.path.join(checkpoint_dir,
                                                                        obj["Key"].\
                                                                        replace(full_key_prefix, "")))
                                dirname, basename = os.path.split(filename)
                                # Download all the checkpoints but not the frozen models since they
                                # are not necessary
                                _, file_extension = os.path.splitext(obj["Key"])
                                if file_extension != '.pb' \
                                and (basename.startswith(checkpoint_state.name) or not has_chkpnt):
                                    if not os.path.exists(dirname):
                                        os.makedirs(dirname)
                                    s3_client.download_file(Bucket=bucket,
                                                            Key=obj["Key"],
                                                            Filename=filename)
                            # Change the coach checkpoint file to point to the latest available checkpoint,
                            # also log that we are changing the checkpoint.
                            if not has_chkpnt:
                                all_ckpnts = _filter_checkpoint_files(os.listdir(checkpoint_dir))
                                if all_ckpnts:
                                    LOG.info("%s not in s3 bucket, downloading all checkpoints \
                                                and using %s", checkpoint_state.name, all_ckpnts[-1])
                                    state_file.write(all_ckpnts[-1])
                                else:
                                    log_and_exit("No checkpoint files",
                                                 SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                 SIMAPP_EVENT_ERROR_CODE_400)
                    break
            return True

        except botocore.exceptions.ClientError:
            log_and_exit("Unable to download checkpoint",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception:
            log_and_exit("Unable to download checkpoint",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
