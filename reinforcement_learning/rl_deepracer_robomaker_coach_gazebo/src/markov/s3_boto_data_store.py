import io
import os
import sys
import time
import json
import logging
import traceback

import boto3
from google.protobuf import text_format
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState

from rl_coach.data_stores.data_store import DataStore, DataStoreParameters, SyncFiles
from markov import utils

logger = utils.Logger(__name__, logging.INFO).get_logger()

CHECKPOINT_METADATA_FILENAME = "checkpoint"
SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND = 1
IP_KEY = "IP"


class S3BotoDataStoreParameters(DataStoreParameters):
    def __init__(self, aws_region: str = "us-west-2", bucket_name: str = None, s3_folder: str = None,
                 checkpoint_dir: str = None):
        super().__init__("s3", "", "")
        self.aws_region = aws_region
        self.bucket = bucket_name
        self.s3_folder = s3_folder
        self.checkpoint_dir = checkpoint_dir
        self.lock_file = ".lock"


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
        s3_client = self._get_client()
        s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                 Bucket=self.params.bucket,
                                 Key=self._get_s3_key(SyncFiles.FINISHED.value))

    def save_to_store(self):
        try:
            s3_client = self._get_client()

            # Delete any existing lock file
            s3_client.delete_object(Bucket=self.params.bucket, Key=self._get_s3_key(self.params.lock_file))

            # We take a lock by writing a lock file to the same location in S3
            s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                     Bucket=self.params.bucket,
                                     Key=self._get_s3_key(self.params.lock_file))

            # Start writing the model checkpoints to S3
            checkpoint = self._get_current_checkpoint()
            if checkpoint:
                checkpoint_number = self._get_checkpoint_number(checkpoint)

            checkpoint_file = None
            for root, dirs, files in os.walk(self.params.checkpoint_dir):
                num_files_uploaded = 0
                for filename in files:
                    # Skip the checkpoint file that has the latest checkpoint number
                    if filename == CHECKPOINT_METADATA_FILENAME:
                        checkpoint_file = (root, filename)
                        continue

                    if not filename.startswith(str(checkpoint_number)):
                        continue

                    # Upload all the other files from the checkpoint directory
                    abs_name = os.path.abspath(os.path.join(root, filename))
                    rel_name = os.path.relpath(abs_name, self.params.checkpoint_dir)
                    s3_client.upload_file(Filename=abs_name,
                                          Bucket=self.params.bucket,
                                          Key=self._get_s3_key(rel_name))
                    num_files_uploaded += 1
            logger.info("Uploaded {} files for checkpoint {}".format(num_files_uploaded, checkpoint_number))

            # After all the checkpoint files have been uploaded, we upload the version file.
            abs_name = os.path.abspath(os.path.join(checkpoint_file[0], checkpoint_file[1]))
            rel_name = os.path.relpath(abs_name, self.params.checkpoint_dir)
            s3_client.upload_file(Filename=abs_name,
                                  Bucket=self.params.bucket,
                                  Key=self._get_s3_key(rel_name))

            # Release the lock by deleting the lock file from S3
            s3_client.delete_object(Bucket=self.params.bucket, Key=self._get_s3_key(self.params.lock_file))

            # Upload the frozen graph which is used for deployment
            if self.graph_manager:
                utils.write_frozen_graph(self.graph_manager)
                # upload the model_<ID>.pb to S3. NOTE: there's no cleanup as we don't know the best checkpoint
                iteration_id = self.graph_manager.level_managers[0].agents['agent'].training_iteration
                frozen_graph_fpath = utils.SM_MODEL_OUTPUT_DIR + "/model.pb"
                frozen_graph_s3_name = "model_%s.pb" % iteration_id
                s3_client.upload_file(Filename=frozen_graph_fpath,
                                  Bucket=self.params.bucket,
                                  Key=self._get_s3_key(frozen_graph_s3_name))
                logger.info("saved intermediate frozen graph: {}".format(self._get_s3_key(frozen_graph_s3_name)))

            # Clean up old checkpoints
            checkpoint = self._get_current_checkpoint()
            if checkpoint:
                checkpoint_number = self._get_checkpoint_number(checkpoint)
                checkpoint_number_to_delete = checkpoint_number - 4

                # List all the old checkpoint files to be deleted
                response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                     Prefix=self._get_s3_key(str(checkpoint_number_to_delete) + "_"))
                if "Contents" in response:
                    num_files = 0
                    for obj in response["Contents"]:
                        s3_client.delete_object(Bucket=self.params.bucket,
                                                Key=obj["Key"])
                        num_files += 1
        except Exception as e:
            utils.json_format_logger("Exception [{}] occured while uploading files on S3 for checkpoint".format(e),
                      **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_500))
            raise e

    def load_from_store(self, expected_checkpoint_number=-1):
        try:
            filename = os.path.abspath(os.path.join(self.params.checkpoint_dir, CHECKPOINT_METADATA_FILENAME))
            if not os.path.exists(self.params.checkpoint_dir):
                os.makedirs(self.params.checkpoint_dir)

            while True:
                s3_client = self._get_client()
                # Check if there's a finished file
                response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                     Prefix=self._get_s3_key(SyncFiles.FINISHED.value))
                if "Contents" in response:
                    finished_file_path = os.path.abspath(os.path.join(self.params.checkpoint_dir,
                                                                      SyncFiles.FINISHED.value))
                    s3_client.download_file(Bucket=self.params.bucket,
                                            Key=self._get_s3_key(SyncFiles.FINISHED.value),
                                            Filename=finished_file_path)
                    return False

                # Check if there's a lock file
                response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                     Prefix=self._get_s3_key(self.params.lock_file))

                if "Contents" not in response:
                    try:
                        # If no lock is found, try getting the checkpoint
                        s3_client.download_file(Bucket=self.params.bucket,
                                                Key=self._get_s3_key(CHECKPOINT_METADATA_FILENAME),
                                                Filename=filename)
                    except Exception as e:
                        time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                        continue
                else:
                    time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                    continue

                checkpoint = self._get_current_checkpoint()
                if checkpoint:
                    checkpoint_number = self._get_checkpoint_number(checkpoint)

                    # if we get a checkpoint that is older that the expected checkpoint, we wait for
                    #  the new checkpoint to arrive.
                    if checkpoint_number < expected_checkpoint_number:
                        logger.info("Expecting checkpoint >= %s. Waiting." % expected_checkpoint_number)
                        time.sleep(SLEEP_TIME_WHILE_WAITING_FOR_DATA_FROM_TRAINER_IN_SECOND)
                        continue

                    # Found a checkpoint to be downloaded
                    response = s3_client.list_objects_v2(Bucket=self.params.bucket,
                                                         Prefix=self._get_s3_key(checkpoint.model_checkpoint_path))
                    if "Contents" in response:
                        num_files = 0
                        for obj in response["Contents"]:
                            # Get the local filename of the checkpoint file
                            full_key_prefix = os.path.normpath(self.key_prefix) + "/"
                            filename = os.path.abspath(os.path.join(self.params.checkpoint_dir,
                                                                    obj["Key"].replace(full_key_prefix, "")))
                            s3_client.download_file(Bucket=self.params.bucket,
                                                    Key=obj["Key"],
                                                    Filename=filename)
                            num_files += 1
                        return True

        except Exception as e:
            utils.json_format_logger("Exception [{}] occured while loading checkpoint from S3. Job failed!".format(e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_503))
            traceback.print_exc()
            sys.exit(1)

    def store_ip(self, ip_address):
        s3_client = self._get_client()
        ip_data = {IP_KEY: ip_address}
        ip_data_json_blob = json.dumps(ip_data)
        ip_data_file_object = io.BytesIO(ip_data_json_blob.encode())
        ip_done_file_object = io.BytesIO(b'done')
        s3_client.upload_fileobj(ip_data_file_object, self.params.bucket, self.ip_data_key)
        s3_client.upload_fileobj(ip_done_file_object, self.params.bucket, self.ip_done_key)

    def download_preset_if_present(self, local_path):
        s3_client = self._get_client()
        response = s3_client.list_objects(Bucket=self.params.bucket, Prefix=self.preset_data_key)

        # If we don't find a preset, return false
        if "Contents" not in response:
            return False

        success = s3_client.download_file(Bucket=self.params.bucket,
                                          Key=self.preset_data_key,
                                          Filename=local_path)
        return success

    def get_current_checkpoint_number(self):
        return self._get_checkpoint_number(self._get_current_checkpoint())

    def _get_current_checkpoint(self):
        try:
            checkpoint_metadata_filepath = os.path.abspath(
                os.path.join(self.params.checkpoint_dir, CHECKPOINT_METADATA_FILENAME))
            checkpoint = CheckpointState()
            if os.path.exists(checkpoint_metadata_filepath) == False:
                return None

            contents = open(checkpoint_metadata_filepath, 'r').read()
            text_format.Merge(contents, checkpoint)
            return checkpoint
        except Exception as e:
            utils.json_format_logger("Exception[{}] occured while reading checkpoint metadata".format(e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_500))
            raise e

    def _get_checkpoint_number(self, checkpoint):
        checkpoint_relative_path = checkpoint.model_checkpoint_path
        return int(checkpoint_relative_path.split('_Step')[0])
