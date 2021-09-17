"""This module implements tf model ckpt and pb file specifically"""

import logging
import os
import queue
import re
import shutil

import tensorflow as tf
from markov import utils
from markov.boto.s3.constants import (
    CHECKPOINT_LOCAL_DIR_FORMAT,
    CHECKPOINT_POSTFIX_DIR,
    FROZEN_HEAD_OUTPUT_GRAPH_FORMAT_MAPPING,
    NUM_MODELS_TO_KEEP,
    SM_MODEL_PB_TEMP_FOLDER,
    TEMP_RENAME_FOLDER,
    ActionSpaceTypes,
    TrainingAlgorithm,
)
from markov.boto.s3.s3_client import S3Client
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_S3_DATA_STORE_EXCEPTION,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from rl_coach.checkpoint import CheckpointStateFile, SingleCheckpoint, _filter_checkpoint_files

LOG = Logger(__name__, logging.INFO).get_logger()

SM_MODEL_OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

if not os.path.exists(TEMP_RENAME_FOLDER):
    os.makedirs(TEMP_RENAME_FOLDER)


class TensorflowModel:
    """This class is for tensorflow model upload and download"""

    def __init__(
        self,
        bucket,
        s3_prefix,
        region_name="us-east-1",
        local_dir="./checkpoint/agent",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
        output_head_format=FROZEN_HEAD_OUTPUT_GRAPH_FORMAT_MAPPING[
            TrainingAlgorithm.CLIPPED_PPO.value
        ],
    ):
        """This class is for tensorflow model upload and download

        Args:
            bucket (str): S3 bucket string
            s3_prefix (str): S3 prefix string
            region_name (str): S3 region name
            local_dir (str): local file directory
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry
            output_head_format (str): output head format for the specific algorithm and action space
                                      which will be used to store the frozen graph
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
        self._bucket = bucket
        self._local_dir = os.path.normpath(CHECKPOINT_LOCAL_DIR_FORMAT.format(local_dir))
        self._s3_key_dir = os.path.normpath(os.path.join(s3_prefix, CHECKPOINT_POSTFIX_DIR))
        self._delete_queue = queue.Queue()
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)
        self.output_head_format = output_head_format

    def _download(self, s3_key, local_path):
        """download files from s3 bucket

        Args:
            s3_key (str): S3 key string
            local_path (str): local path string
        """
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        self._s3_client.download_file(bucket=self._bucket, s3_key=s3_key, local_path=local_path)

        _, file_name = os.path.split(local_path)
        LOG.info(
            "[s3] Successfully downloaded {} from \
                 s3 key {} to local {}.".format(
                file_name, s3_key, local_path
            )
        )

    def get(self, coach_checkpoint_state_file):
        """get tensorflow model specified in the rl coach checkpoint state file
        If the rl coach checkpoint state file specified checkpoint is missing. It will
        download last checkpoints and over write the last in local rl coach checkpoint state file

        Args:
            coach_checkpoint_state_file (CheckpointStateFile): CheckpointStateFile instance
        """
        has_checkpoint = False
        last_checkpoint_number = -1
        last_checkpoint_name = None
        # list everything in tensorflow model s3 bucket dir
        # to find the checkpoint specified in .coach_checkpoint
        # or use the last
        checkpoint_name = str(coach_checkpoint_state_file.read())
        for page in self._s3_client.paginate(bucket=self._bucket, prefix=self._s3_key_dir):
            if "Contents" in page:
                # Check to see if the desired tensorflow model is in the bucket
                # for example if obj is (dir)/487_Step-2477372.ckpt.data-00000-of-00001
                # curr_checkpoint_number: 487
                # curr_checkpoint_name: 487_Step-2477372.ckpt.data-00000-of-00001
                for obj in page["Contents"]:
                    curr_checkpoint_name = os.path.split(obj["Key"])[1]
                    # if found the checkpoint name stored in .coach_checkpoint file
                    # break inner loop for file search
                    if curr_checkpoint_name.startswith(checkpoint_name):
                        has_checkpoint = True
                        break
                    # if the file name does not start with a number (not ckpt file)
                    # continue for next file
                    if not utils.is_int_repr(curr_checkpoint_name.split("_")[0]):
                        continue
                    # if the file name start with a number, update the last checkpoint name
                    # and number
                    curr_checkpoint_number = int(curr_checkpoint_name.split("_")[0])
                    if curr_checkpoint_number > last_checkpoint_number:
                        last_checkpoint_number = curr_checkpoint_number
                        last_checkpoint_name = curr_checkpoint_name.rsplit(".", 1)[0]
            # break out from pagination if find the checkpoint
            if has_checkpoint:
                break

        # update checkpoint_name to the last_checkpoint_name and overwrite local
        # .coach_checkpoint file to contain the last checkpoint
        if not has_checkpoint:
            if last_checkpoint_name:
                coach_checkpoint_state_file.write(
                    SingleCheckpoint(num=last_checkpoint_number, name=last_checkpoint_name)
                )
                LOG.info(
                    "%s not in s3 bucket, downloading %s checkpoints",
                    checkpoint_name,
                    last_checkpoint_name,
                )
                checkpoint_name = last_checkpoint_name
            else:
                log_and_exit(
                    "No checkpoint files",
                    SIMAPP_S3_DATA_STORE_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_400,
                )

        # download the desired checkpoint file
        for page in self._s3_client.paginate(bucket=self._bucket, prefix=self._s3_key_dir):
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    _, file_name = os.path.split(s3_key)
                    local_path = os.path.normpath(os.path.join(self._local_dir, file_name))
                    _, file_extension = os.path.splitext(s3_key)
                    if file_extension != ".pb" and file_name.startswith(checkpoint_name):
                        self._download(s3_key=s3_key, local_path=local_path)

    def persist(self, coach_checkpoint_state_file, s3_kms_extra_args):
        """upload tensorflow model specified in rl coach checkpoint state file into the s3 bucket

        Args:
            coach_checkpoint_state_file (CheckpointStateFile): CheckpointStateFile instance
            s3_kms_extra_args (dict): s3 key management service extra argument
        """
        ckpt_state = None
        check_point_key_list = []
        if coach_checkpoint_state_file.exists():
            ckpt_state = coach_checkpoint_state_file.read()
            checkpoint_file = None
            num_files_uploaded = 0
            for root, _, files in os.walk(self._local_dir):
                for filename in files:
                    if filename == CheckpointStateFile.checkpoint_state_filename:
                        checkpoint_file = (root, filename)
                        continue
                    if filename.startswith(ckpt_state.name):
                        abs_name = os.path.abspath(os.path.join(root, filename))
                        rel_name = os.path.relpath(abs_name, self._local_dir)
                        self._s3_client.upload_file(
                            bucket=self._bucket,
                            s3_key=os.path.normpath(os.path.join(self._s3_key_dir, rel_name)),
                            local_path=abs_name,
                            s3_kms_extra_args=s3_kms_extra_args,
                        )
                        check_point_key_list.append(
                            os.path.normpath(os.path.join(self._s3_key_dir, rel_name))
                        )
                        num_files_uploaded += 1
            LOG.info("Uploaded %s files for checkpoint %s", num_files_uploaded, ckpt_state.num)
            if check_point_key_list:
                self._delete_queue.put(check_point_key_list)

    def rename(self, coach_checkpoint_state_file, agent_name):
        """rename the tensorflow model specified in the rl coach checkpoint state file to include
        agent name

        Args:
            coach_checkpoint_state_file (CheckpointStateFile): CheckpointStateFile instance
            agent_name (str): agent name
        """
        try:
            LOG.info(
                "Renaming checkpoint from checkpoint_dir: {} for agent: {}".format(
                    self._local_dir, agent_name
                )
            )
            checkpoint_name = str(coach_checkpoint_state_file.read())
            tf_checkpoint_file = os.path.join(self._local_dir, "checkpoint")
            with open(tf_checkpoint_file, "w") as outfile:
                outfile.write('model_checkpoint_path: "{}"'.format(checkpoint_name))

            with tf.Session() as sess:
                for var_name, _ in tf.contrib.framework.list_variables(self._local_dir):
                    # Load the variable
                    var = tf.contrib.framework.load_variable(self._local_dir, var_name)
                    new_name = var_name
                    # Set the new name
                    # Replace agent/ or agent_#/ with {agent_name}/
                    new_name = re.sub("agent/|agent_\d+/", "{}/".format(agent_name), new_name)
                    # Rename the variable
                    var = tf.Variable(var, name=new_name)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                renamed_checkpoint_path = os.path.join(TEMP_RENAME_FOLDER, checkpoint_name)
                LOG.info("Saving updated checkpoint to {}".format(renamed_checkpoint_path))
                saver.save(sess, renamed_checkpoint_path)
            # Remove the tensorflow 'checkpoint' file
            os.remove(tf_checkpoint_file)
            # Remove the old checkpoint from the checkpoint dir
            for file_name in os.listdir(self._local_dir):
                if checkpoint_name in file_name:
                    os.remove(os.path.join(self._local_dir, file_name))
            # Copy the new checkpoint with renamed variable to the checkpoint dir
            for file_name in os.listdir(TEMP_RENAME_FOLDER):
                full_file_name = os.path.join(os.path.abspath(TEMP_RENAME_FOLDER), file_name)
                if os.path.isfile(full_file_name) and file_name != "checkpoint":
                    shutil.copy(full_file_name, self._local_dir)
            # Remove files from temp_rename_folder
            shutil.rmtree(TEMP_RENAME_FOLDER)
            tf.reset_default_graph()
        # If either of the checkpoint files (index, meta or data) not found
        except tf.errors.NotFoundError as err:
            log_and_exit(
                "No checkpoint found: {}".format(err),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        # Thrown when user modifies model, checkpoints get corrupted/truncated
        except tf.errors.DataLossError as err:
            log_and_exit(
                "User modified ckpt, unrecoverable dataloss or corruption: {}".format(err),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except tf.errors.OutOfRangeError as err:
            log_and_exit(
                "User modified ckpt: {}".format(err),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except ValueError as err:
            if utils.is_user_error(err):
                log_and_exit(
                    "Couldn't find 'checkpoint' file or checkpoints in given \
                                directory ./checkpoint: {}".format(
                        err
                    ),
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_400,
                )
            else:
                log_and_exit(
                    "ValueError in rename checkpoint: {}".format(err),
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_500,
                )
        except Exception as ex:
            log_and_exit(
                "Exception in rename checkpoint: {}".format(ex),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def persist_tensorflow_frozen_graph(
        self,
        agent_name,
        graph_manager,
        coach_checkpoint_state_file,
        best_checkpoint_number,
        last_checkpoint_number,
        s3_kms_extra_args,
    ):
        """persist the tensorflow frozen graph specified by rl coach checkpoint state file into s3

        Args:
            agent_name (str): agent name
            graph_manager (MultiAgentGraphManager): MultiAgentGraphManager class instance
            coach_checkpoint_state_file (CheckpointStateFile): CheckpointStateFile class instance
            best_checkpoint_number (int): best checkpoint number
            last_checkpoint_number (int): last checkpoint number
            s3_kms_extra_args (dict): s3 key management service extra argument
        """
        # checkpoint state is always present for the checkpoint dir passed.
        # We make same assumption while we get the best checkpoint in s3_metrics
        checkpoint_num = coach_checkpoint_state_file.read().num
        self.write_frozen_graph(graph_manager.sess, agent_name, checkpoint_num)
        frozen_name = "model_{}.pb".format(checkpoint_num)
        frozen_graph_local_path = os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_name, frozen_name)
        # upload the model_<ID>.pb to S3.
        self._s3_client.upload_file(
            bucket=self._bucket,
            s3_key=os.path.normpath(os.path.join(self._s3_key_dir, frozen_name)),
            local_path=frozen_graph_local_path,
            s3_kms_extra_args=s3_kms_extra_args,
        )

        LOG.info(
            "saved intermediate frozen graph: %s",
            os.path.normpath(os.path.join(self._s3_key_dir, frozen_name)),
        )

        # Copy the best checkpoint to the SM_MODEL_OUTPUT_DIR
        self.copy_best_frozen_graph_to_sm_output_dir(
            best_checkpoint_number=best_checkpoint_number,
            last_checkpoint_number=last_checkpoint_number,
            source_dir=os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_name),
            dest_dir=os.path.join(SM_MODEL_OUTPUT_DIR, agent_name),
        )

    def copy_best_frozen_graph_to_sm_output_dir(
        self, best_checkpoint_number, last_checkpoint_number, source_dir, dest_dir
    ):
        """Copy the frozen model for the current best checkpoint from soure directory to the destination directory.

        Args:
            s3_bucket (str): S3 bucket where the deepracer_checkpoints.json is stored
            s3_prefix (str): S3 prefix where the deepracer_checkpoints.json is stored
            region (str): AWS region where the deepracer_checkpoints.json is stored
            source_dir (str): Source directory where the frozen models are present
            dest_dir (str): Sagemaker output directory where we store the frozen models for best checkpoint
        """
        dest_dir_pb_files = [
            filename
            for filename in os.listdir(dest_dir)
            if os.path.isfile(os.path.join(dest_dir, filename)) and filename.endswith(".pb")
        ]
        source_dir_pb_files = [
            filename
            for filename in os.listdir(source_dir)
            if os.path.isfile(os.path.join(source_dir, filename)) and filename.endswith(".pb")
        ]

        LOG.info(
            "Best checkpoint number: {}, Last checkpoint number: {}".format(
                best_checkpoint_number, last_checkpoint_number
            )
        )
        best_model_name = "model_{}.pb".format(best_checkpoint_number)
        last_model_name = "model_{}.pb".format(last_checkpoint_number)
        if len(source_dir_pb_files) < 1:
            log_and_exit(
                "Could not find any frozen model file in the local directory",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        try:
            # Could not find the deepracer_checkpoints.json file or there are no model.pb files in destination
            if best_checkpoint_number == -1 or len(dest_dir_pb_files) == 0:
                if len(source_dir_pb_files) > 1:
                    LOG.info(
                        "More than one model.pb found in the source directory. Choosing the "
                        "first one to copy to destination: {}".format(source_dir_pb_files[0])
                    )
                # copy the frozen model present in the source directory
                LOG.info(
                    "Copying the frozen checkpoint from {} to {}.".format(
                        os.path.join(source_dir, source_dir_pb_files[0]),
                        os.path.join(dest_dir, "model.pb"),
                    )
                )
                shutil.copy(
                    os.path.join(source_dir, source_dir_pb_files[0]),
                    os.path.join(dest_dir, "model.pb"),
                )
            else:
                # Delete the current .pb files in the destination direcory
                for filename in dest_dir_pb_files:
                    os.remove(os.path.join(dest_dir, filename))

                # Copy the frozen model for the current best checkpoint to the destination directory
                LOG.info(
                    "Copying the frozen checkpoint from {} to {}.".format(
                        os.path.join(source_dir, best_model_name),
                        os.path.join(dest_dir, "model.pb"),
                    )
                )
                shutil.copy(
                    os.path.join(source_dir, best_model_name), os.path.join(dest_dir, "model.pb")
                )

                # Loop through the current list of frozen models in source directory and
                # delete the iterations lower than last_checkpoint_iteration except best_model
                for filename in source_dir_pb_files:
                    if filename not in [best_model_name, last_model_name]:
                        if len(filename.split("_")[1]) > 1 and len(
                            filename.split("_")[1].split(".pb")
                        ):
                            file_iteration = int(filename.split("_")[1].split(".pb")[0])
                            if file_iteration < last_checkpoint_number:
                                os.remove(os.path.join(source_dir, filename))
                        else:
                            LOG.error(
                                "Frozen model name not in the right format in the source directory: {}, {}".format(
                                    filename, source_dir
                                )
                            )
        except FileNotFoundError as err:
            log_and_exit(
                "No such file or directory: {}".format(err),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )

    def write_frozen_graph(self, sess, agent_name, iteration_id):
        """Write the frozen graph to the temporary folder with a name model_{}.pb for the iteration_id passed

        Args:
            sess (dict): key as agent name and value as agent params
            agent_name (str): Name of the agent
            iteration_id (int): Iteration id for which we are saving the model_{}.pb
        """
        if not os.path.exists(os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_name)):
            os.makedirs(os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_name))
        if not os.path.exists(os.path.join(SM_MODEL_OUTPUT_DIR, agent_name)):
            os.makedirs(os.path.join(SM_MODEL_OUTPUT_DIR, agent_name))
        output_head = [self.output_head_format.format(agent_name)]
        frozen = tf.graph_util.convert_variables_to_constants(
            sess[agent_name], sess[agent_name].graph_def, output_head
        )
        tf.train.write_graph(
            frozen,
            os.path.join(SM_MODEL_PB_TEMP_FOLDER, agent_name),
            "model_{}.pb".format(iteration_id),
            as_text=False,
        )

    def delete(self, coach_checkpoint_state_file, best_checkpoint):
        """delete tensorflow models from s3 bucket

        Args:
            coach_checkpoint_state_file (CheckpointStateFile): CheckpointStateFile class instance
            best_checkpoint (str): best checkpoing string
        """
        if coach_checkpoint_state_file.read() and self._delete_queue.qsize() > NUM_MODELS_TO_KEEP:
            while self._delete_queue.qsize() > NUM_MODELS_TO_KEEP:
                key_list = self._delete_queue.get()
                if best_checkpoint and all(
                    list(
                        map(
                            lambda file_name: best_checkpoint in file_name,
                            [os.path.split(file)[-1] for file in key_list],
                        )
                    )
                ):
                    self._delete_queue.put(key_list)
                else:
                    delete_iteration_ids = set()
                    for key in key_list:
                        self._s3_client.delete_object(bucket=self._bucket, s3_key=key)
                        # Get the name of the file in the checkpoint directory that has to be deleted
                        # and extract the iteration id out of the name
                        file_in_checkpoint_dir = os.path.split(key)[-1]
                        if len(file_in_checkpoint_dir.split("_Step")) > 0:
                            delete_iteration_ids.add(file_in_checkpoint_dir.split("_Step")[0])
                    LOG.info(
                        "Deleting the frozen models in s3 for the iterations: %s",
                        delete_iteration_ids,
                    )
                    # Delete the model_{}.pb files from the s3 bucket for the previous iterations
                    for iteration_id in list(delete_iteration_ids):
                        frozen_name = "model_{}.pb".format(iteration_id)
                        self._s3_client.delete_object(
                            bucket=self._bucket,
                            s3_key=os.path.normpath(os.path.join(self._s3_key_dir, frozen_name)),
                        )
