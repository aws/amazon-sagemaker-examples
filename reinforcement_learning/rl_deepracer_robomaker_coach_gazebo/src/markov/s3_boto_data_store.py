import io
import logging
import os
import time
import boto3
import botocore

from rl_coach.checkpoint import CheckpointStateFile, _filter_checkpoint_files
from rl_coach.data_stores.data_store import DataStore, DataStoreParameters, SyncFiles

from markov import utils

import tensorflow as tf

logger = utils.Logger(__name__, logging.INFO).get_logger()

# The number of models to keep in S3
#! TODO discuss with product team if this number should be configurable
NUM_MODELS_TO_KEEP = 4

SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND = 1
SM_MODEL_OUTPUT_DIR = os.environ.get("ALGO_MODEL_DIR", "/opt/ml/model")

class S3BotoDataStoreParameters(DataStoreParameters):
    def __init__(self, aws_region: str = "us-west-2", bucket_name: str = None, s3_folder: str = None,
                 checkpoint_dir: str = None):
        super().__init__("s3", "", "")
        self.aws_region = aws_region
        self.bucket = bucket_name
        self.s3_folder = s3_folder
        self.checkpoint_dir = checkpoint_dir


class S3BotoDataStore(DataStore):
    def __init__(self, params: S3BotoDataStoreParameters):
        self.params = params
        self.key_prefix = os.path.normpath(self.params.s3_folder + "/model")
        self.ip_data_key = os.path.normpath(self.params.s3_folder + "/ip/ip.json")
        self.ip_done_key = os.path.normpath(self.params.s3_folder + "/ip/done")
        self.preset_data_key = os.path.normpath(self.params.s3_folder + "/presets/preset.py")
        self.graph_manager = None

    def _get_s3_key(self, key):
        return os.path.normpath(self.key_prefix + "/" + key)

    def _get_client(self):
        session = boto3.session.Session()
        return session.client('s3', region_name=self.params.aws_region)

    def deploy(self) -> bool:
        return True

    def get_info(self):
        return "s3://{}/{}".format(self.params.bucket, self.params.s3_folder)

    def undeploy(self) -> bool:
        return True

    def upload_finished_file(self):
        try:
            s3_client = self._get_client()
            s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                     Bucket=self.params.bucket,
                                     Key=self._get_s3_key(SyncFiles.FINISHED.value))
        except botocore.exceptions.ClientError as e:
            utils.json_format_logger("Unable to upload finished file to {}, {}"
                                     .format(self.params.bucket, e.response['Error']['Code']),
                                     **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                   utils.SIMAPP_EVENT_ERROR_CODE_400))
            utils.simapp_exit_gracefully()
        except Exception as e:
            utils.json_format_logger("Unable to upload finished file to {}, {}"
                                     .format(self.params.bucket, e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                     utils.SIMAPP_EVENT_ERROR_CODE_500))
            utils.simapp_exit_gracefully()

    def save_to_store(self):
        try:
            s3_client = self._get_client()
            checkpoint_dir = self.params.checkpoint_dir

            # remove lock file if it exists
            s3_client.delete_object(Bucket=self.params.bucket, Key=self._get_s3_key(SyncFiles.LOCKFILE.value))

            # acquire lock
            s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                     Bucket=self.params.bucket,
                                     Key=self._get_s3_key(SyncFiles.LOCKFILE.value))

            state_file = CheckpointStateFile(os.path.abspath(checkpoint_dir))
            ckpt_state = None
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
                                                  Bucket=self.params.bucket,
                                                  Key=self._get_s3_key(rel_name))
                            num_files_uploaded += 1
                logger.info("Uploaded {} files for checkpoint {}".format(num_files_uploaded, ckpt_state.num))

                abs_name = os.path.abspath(os.path.join(checkpoint_file[0], checkpoint_file[1]))
                rel_name = os.path.relpath(abs_name, checkpoint_dir)
                s3_client.upload_file(Filename=abs_name,
                                      Bucket=self.params.bucket,
                                      Key=self._get_s3_key(rel_name))

            # upload Finished if present
            if os.path.exists(os.path.join(checkpoint_dir, SyncFiles.FINISHED.value)):
                s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                         Bucket=self.params.bucket,
                                         Key=self._get_s3_key(SyncFiles.FINISHED.value))

            # upload Ready if present
            if os.path.exists(os.path.join(checkpoint_dir, SyncFiles.TRAINER_READY.value)):
                s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                         Bucket=self.params.bucket,
                                         Key=self._get_s3_key(SyncFiles.TRAINER_READY.value))

            # release lock
            s3_client.delete_object(Bucket=self.params.bucket, Key=self._get_s3_key(SyncFiles.LOCKFILE.value))

            # Upload the frozen graph which is used for deployment
            if self.graph_manager:
                self.write_frozen_graph(self.graph_manager)
                # upload the model_<ID>.pb to S3. NOTE: there's no cleanup as we don't know the best checkpoint
                for agent_params in self.graph_manager.agents_params:
                    iteration_id = self.graph_manager.level_managers[0].agents[agent_params.name].training_iteration
                    frozen_graph_fpath = os.path.join(SM_MODEL_OUTPUT_DIR, agent_params.name, "model.pb")
                    frozen_name = "model_{}.pb".format(iteration_id)
                    frozen_graph_s3_name = frozen_name if len(self.graph_manager.agents_params) == 1 \
                        else os.path.join(agent_params.name, frozen_name)
                    s3_client.upload_file(Filename=frozen_graph_fpath,
                                          Bucket=self.params.bucket,
                                          Key=self._get_s3_key(frozen_graph_s3_name))
                    logger.info("saved intermediate frozen graph: {}".format(self._get_s3_key(frozen_graph_s3_name)))

            # Clean up old checkpoints
            if ckpt_state:
                checkpoint_number_to_delete = ckpt_state.num - NUM_MODELS_TO_KEEP

                # List all the old checkpoint files to be deleted
                response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                     Prefix=self._get_s3_key(""))
                if "Contents" in response:
                    for obj in response["Contents"]:
                        _, basename = os.path.split(obj["Key"])
                        if basename.startswith("{}_".format(checkpoint_number_to_delete)):
                            s3_client.delete_object(Bucket=self.params.bucket,
                                                    Key=obj["Key"])

        except botocore.exceptions.ClientError as e:
            utils.json_format_logger("Unable to upload checkpoint to {}, {}"
                                     .format(self.params.bucket, e.response['Error']['Code']),
                                     **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                   utils.SIMAPP_EVENT_ERROR_CODE_400))
            utils.simapp_exit_gracefully()
        except Exception as e:
            utils.json_format_logger("Unable to upload checkpoint to {}, {}"
                                     .format(self.params.bucket, e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                     utils.SIMAPP_EVENT_ERROR_CODE_500))
            utils.simapp_exit_gracefully()

    def write_frozen_graph(self, graph_manager):
        if not os.path.exists(SM_MODEL_OUTPUT_DIR):
            os.makedirs(SM_MODEL_OUTPUT_DIR)
        for agent_params in graph_manager.agents_params:
            agent_name = agent_params.name
            output_head = ['main_level/{}/main/online/network_1/ppo_head_0/policy'.format(agent_name)]
            frozen = tf.graph_util.convert_variables_to_constants(graph_manager.sess[agent_name], graph_manager.sess[agent_name].graph_def, output_head)
            tf.train.write_graph(frozen, os.path.join(SM_MODEL_OUTPUT_DIR, agent_name), 'model.pb', as_text=False)

    def load_from_store(self, expected_checkpoint_number=-1):
        try:
            if not os.path.exists(self.params.checkpoint_dir):
                os.makedirs(self.params.checkpoint_dir)

            while True:
                s3_client = self._get_client()
                state_file = CheckpointStateFile(os.path.abspath(self.params.checkpoint_dir))

                # wait until lock is removed
                response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                     Prefix=self._get_s3_key(SyncFiles.LOCKFILE.value))
                if "Contents" not in response:
                    try:
                        # fetch checkpoint state file from S3
                        s3_client.download_file(Bucket=self.params.bucket,
                                                Key=self._get_s3_key(state_file.filename),
                                                Filename=state_file.path)
                    except Exception as e:
                        time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                        continue
                else:
                    time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                    continue

                # check if there's a Finished file
                response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                     Prefix=self._get_s3_key(SyncFiles.FINISHED.value))
                if "Contents" in response:
                    try:
                        finished_file_path = os.path.abspath(os.path.join(self.params.checkpoint_dir,
                                                                          SyncFiles.FINISHED.value))
                        s3_client.download_file(Bucket=self.params.bucket,
                                                Key=self._get_s3_key(SyncFiles.FINISHED.value),
                                                Filename=finished_file_path)
                    except Exception as e:
                        pass

                # check if there's a Ready file
                response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                     Prefix=self._get_s3_key(SyncFiles.TRAINER_READY.value))
                if "Contents" in response:
                    try:
                        ready_file_path = os.path.abspath(os.path.join(self.params.checkpoint_dir,
                                                                       SyncFiles.TRAINER_READY.value))
                        s3_client.download_file(Bucket=self.params.bucket,
                                                Key=self._get_s3_key(SyncFiles.TRAINER_READY.value),
                                                Filename=ready_file_path)
                    except Exception as e:
                        pass

                checkpoint_state = state_file.read()
                if checkpoint_state is not None:

                    # if we get a checkpoint that is older that the expected checkpoint, we wait for
                    #  the new checkpoint to arrive.
                    if checkpoint_state.num < expected_checkpoint_number:
                        time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                        continue

                    response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                         Prefix=self._get_s3_key(""))
                    if "Contents" in response:
                        # Check to see if the desired checkpoint is in the bucket
                        has_chkpnt = any(list(map(lambda obj: os.path.split(obj['Key'])[1].\
                                                              startswith(checkpoint_state.name),
                                                  response['Contents'])))
                        for obj in response["Contents"]:
                            full_key_prefix = os.path.normpath(self.key_prefix) + "/"
                            filename = os.path.abspath(os.path.join(self.params.checkpoint_dir,
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
                                s3_client.download_file(Bucket=self.params.bucket,
                                                        Key=obj["Key"],
                                                        Filename=filename)
                        # Change the coach checkpoint file to point to the latest available checkpoint,
                        # also log that we are changing the checkpoint.
                        if not has_chkpnt:
                            all_ckpnts = _filter_checkpoint_files(os.listdir(self.params.checkpoint_dir))
                            if all_ckpnts:
                                logger.info("%s not in s3 bucket, downloading all checkpoints \
                                            and using %s", checkpoint_state.name, all_ckpnts[-1])
                                state_file.write(all_ckpnts[-1])
                            else:
                                utils.json_format_logger("No checkpoint files found in {}".format(self.params.bucket),
                                                         **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                                       utils.SIMAPP_EVENT_ERROR_CODE_400))
                                utils.simapp_exit_gracefully()
                return True

        except botocore.exceptions.ClientError as e:
            utils.json_format_logger("Unable to download checkpoint from {}, {}"
                                     .format(self.params.bucket, e.response['Error']['Code']),
                                     **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                   utils.SIMAPP_EVENT_ERROR_CODE_400))
            utils.simapp_exit_gracefully()
        except Exception as e:
            utils.json_format_logger("Unable to download checkpoint from {}, {}"
                                     .format(self.params.bucket, e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                     utils.SIMAPP_EVENT_ERROR_CODE_500))
            utils.simapp_exit_gracefully()



    def get_latest_checkpoint(self):
        try:
            filename = os.path.abspath(os.path.join(self.params.checkpoint_dir, "latest_ckpt"))
            if not os.path.exists(self.params.checkpoint_dir):
                os.makedirs(self.params.checkpoint_dir)

            while True:
                s3_client = self._get_client()
                state_file = CheckpointStateFile(os.path.abspath(self.params.checkpoint_dir))

                # wait until lock is removed
                response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                     Prefix=self._get_s3_key(SyncFiles.LOCKFILE.value))
                if "Contents" not in response:
                    try:
                        # fetch checkpoint state file from S3
                        s3_client.download_file(Bucket=self.params.bucket,
                                                Key=self._get_s3_key(state_file.filename),
                                                Filename=filename)
                    except Exception as e:
                        time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                        continue
                else:
                    time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                    continue

                return self._get_current_checkpoint_number(checkpoint_metadata_filepath=filename)

        except Exception as e:
            utils.json_format_logger("Exception [{}] occured while getting latest checkpoint from S3.".format(e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_503))


    def _get_current_checkpoint_number(self, checkpoint_metadata_filepath=None):
        try:
            if not os.path.exists(checkpoint_metadata_filepath):
                return None
            with open(checkpoint_metadata_filepath, 'r') as fp:
                data = fp.read()
                return int(data.split('_')[0])
        except Exception as e:
            utils.json_format_logger("Exception[{}] occured while reading checkpoint metadata".format(e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_500))
            raise e
