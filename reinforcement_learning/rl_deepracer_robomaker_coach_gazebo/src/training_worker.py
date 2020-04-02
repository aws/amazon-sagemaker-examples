import os
import shutil
import argparse
import json
import logging
import math
import numpy as np
import subprocess
import time

from rl_coach.base_parameters import TaskParameters, DistributedCoachSynchronizationType, RunType
from rl_coach import core_types
from rl_coach.data_stores.data_store import SyncFiles
from rl_coach.logger import screen
from rl_coach.memories.backend.redis import RedisPubSubMemoryBackendParameters
from rl_coach.utils import short_dynamic_import

from markov import utils
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.training_agent_factory import create_training_agent
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.s3_client import SageS3Client
from markov.sagemaker_graph_manager import get_graph_manager
from markov.utils_parse_model_metadata import parse_model_metadata
from markov.samples.sample_collector import SampleCollector

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

logger = utils.Logger(__name__, logging.INFO).get_logger()

PRETRAINED_MODEL_DIR = "./pretrained_checkpoint"
SM_MODEL_OUTPUT_DIR = os.environ.get("ALGO_MODEL_DIR", "/opt/ml/model")
CUSTOM_FILES_PATH = "./custom_files"

if not os.path.exists(CUSTOM_FILES_PATH):
    os.makedirs(CUSTOM_FILES_PATH)


def training_worker(graph_manager, task_parameters, user_batch_size,
                    user_episode_per_rollout):
    try:
        # initialize graph
        graph_manager.create_graph(task_parameters)

        # save initial checkpoint
        graph_manager.save_checkpoint()

        # training loop
        steps = 0

        graph_manager.setup_memory_backend()
        graph_manager.signal_ready()

        # To handle SIGTERM
        door_man = utils.DoorMan()

        while steps < graph_manager.improve_steps.num_steps:
            graph_manager.phase = core_types.RunPhase.TRAIN
            graph_manager.fetch_from_worker(graph_manager.agent_params.algorithm.num_consecutive_playing_steps)
            graph_manager.phase = core_types.RunPhase.UNDEFINED

            episodes_in_rollout = graph_manager.memory_backend.get_total_episodes_in_rollout()

            for level in graph_manager.level_managers:
                for agent in level.agents.values():
                    agent.ap.algorithm.num_consecutive_playing_steps.num_steps = episodes_in_rollout
                    agent.ap.algorithm.num_steps_between_copying_online_weights_to_target.num_steps = episodes_in_rollout

            if graph_manager.should_train():
                # Make sure we have enough data for the requested batches
                rollout_steps = graph_manager.memory_backend.get_rollout_steps()
                if any(rollout_steps.values()) <= 0:
                    utils.log_and_exit("No rollout data retrieved from the rollout worker",
                                       utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                                       utils.SIMAPP_EVENT_ERROR_CODE_500)

                episode_batch_size = user_batch_size if min(rollout_steps.values()) > user_batch_size else 2**math.floor(math.log(min(rollout_steps.values()), 2))
                # Set the batch size to the closest power of 2 such that we have at least two batches, this prevents coach from crashing
                # as  batch size less than 2 causes the batch list to become a scalar which causes an exception
                for level in graph_manager.level_managers:
                    for agent in level.agents.values():
                        agent.ap.network_wrappers['main'].batch_size = episode_batch_size

                steps += 1

                graph_manager.phase = core_types.RunPhase.TRAIN
                graph_manager.train()
                graph_manager.phase = core_types.RunPhase.UNDEFINED

                # Check for Nan's in all agents
                rollout_has_nan = False
                for level in graph_manager.level_managers:
                    for agent in level.agents.values():
                        if np.isnan(agent.loss.get_mean()):
                            rollout_has_nan = True
                if rollout_has_nan:
                    utils.log_and_exit("NaN detected in loss function, aborting training.",
                                       utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                                       utils.SIMAPP_EVENT_ERROR_CODE_500)

                if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
                    graph_manager.save_checkpoint()
                else:
                    graph_manager.occasionally_save_checkpoint()

                # Clear any data stored in signals that is no longer necessary
                graph_manager.reset_internal_state()

            for level in graph_manager.level_managers:
                for agent in level.agents.values():
                    agent.ap.algorithm.num_consecutive_playing_steps.num_steps = user_episode_per_rollout
                    agent.ap.algorithm.num_steps_between_copying_online_weights_to_target.num_steps = user_episode_per_rollout

            if door_man.terminate_now:
                utils.log_and_exit("Received SIGTERM. Checkpointing before exiting.",
                                   utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                                   utils.SIMAPP_EVENT_ERROR_CODE_500)
                graph_manager.save_checkpoint()
                break

    except ValueError as err:
        if utils.is_error_bad_ckpnt(err):
            utils.log_and_exit("User modified model: {}".format(err),
                               utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_400)
        else:
            utils.log_and_exit("An error occured while training: {}".format(err),
                               utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_500)
    except Exception as ex:
        utils.log_and_exit("An error occured while training: {}".format(ex),
                           utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)
    finally:
        graph_manager.data_store.upload_finished_file()

def start_redis_server():
    p = subprocess.Popen("redis-server /etc/redis/redis.conf", shell=True, stderr=subprocess.STDOUT)
    time.sleep(5)
    if p.poll() is not None:
        raise RuntimeError("Could not start Redis server.")
    else:
        print("Redis server started successfully!")
    return p

def main():
    screen.set_use_colors(False)

    parser = argparse.ArgumentParser()
    parser.add_argument('-pk', '--preset_s3_key',
                        help="(string) Name of a preset to download from S3",
                        type=str,
                        required=False)
    parser.add_argument('-ek', '--environment_s3_key',
                        help="(string) Name of an environment file to download from S3",
                        type=str,
                        required=False)
    parser.add_argument('--model_metadata_s3_key',
                        help="(string) Model Metadata File S3 Key",
                        type=str,
                        required=False)
    parser.add_argument('-c', '--checkpoint-dir',
                        help='(string) Path to a folder containing a checkpoint to write the model to.',
                        type=str,
                        default='./checkpoint')
    parser.add_argument('--pretrained-checkpoint-dir',
                        help='(string) Path to a folder for downloading a pre-trained model',
                        type=str,
                        default=PRETRAINED_MODEL_DIR)
    parser.add_argument('--s3_bucket',
                        help='(string) S3 bucket',
                        type=str,
                        default=os.environ.get("SAGEMAKER_SHARED_S3_BUCKET_PATH", "gsaur-test"))
    parser.add_argument('--s3_prefix',
                        help='(string) S3 prefix',
                        type=str,
                        default='sagemaker')
    parser.add_argument('--framework',
                        help='(string) tensorflow or mxnet',
                        type=str,
                        default='tensorflow')
    parser.add_argument('--pretrained_s3_bucket',
                        help='(string) S3 bucket for pre-trained model',
                        type=str)
    parser.add_argument('--pretrained_s3_prefix',
                        help='(string) S3 prefix for pre-trained model',
                        type=str,
                        default='sagemaker')
    parser.add_argument('--aws_region',
                        help='(string) AWS region',
                        type=str,
                        default=os.environ.get("AWS_REGION", "us-east-1"))

    args, _ = parser.parse_known_args()
    
    start_redis_server()

    s3_client = SageS3Client(bucket=args.s3_bucket, s3_prefix=args.s3_prefix, aws_region=args.aws_region)

    # Load the model metadata
    model_metadata_local_path = os.path.join(CUSTOM_FILES_PATH, 'model_metadata.json')
    utils.load_model_metadata(s3_client, args.model_metadata_s3_key, model_metadata_local_path)
    s3_client.upload_file(os.path.normpath("%s/model/model_metadata.json" % args.s3_prefix), model_metadata_local_path)
    shutil.copy2(model_metadata_local_path, SM_MODEL_OUTPUT_DIR)

    success_custom_preset = False
    if args.preset_s3_key:
        preset_local_path = "./markov/presets/preset.py"
        success_custom_preset = s3_client.download_file(s3_key=args.preset_s3_key, local_path=preset_local_path)
        if not success_custom_preset:
            logger.info("Could not download the preset file. Using the default DeepRacer preset.")
        else:
            preset_location = "markov.presets.preset:graph_manager"
            graph_manager = short_dynamic_import(preset_location, ignore_module_case=True)
            success_custom_preset = s3_client.upload_file(
                s3_key=os.path.normpath("%s/presets/preset.py" % args.s3_prefix), local_path=preset_local_path)
            if success_custom_preset:
                logger.info("Using preset: %s" % args.preset_s3_key)

    if not success_custom_preset:
        params_blob = os.environ.get('SM_TRAINING_ENV', '')
        if params_blob:
            params = json.loads(params_blob)
            sm_hyperparams_dict = params["hyperparameters"]
        else:
            sm_hyperparams_dict = {}

        #! TODO each agent should have own config
        agent_config = {'model_metadata': model_metadata_local_path,
                        ConfigParams.CAR_CTRL_CONFIG.value: {ConfigParams.LINK_NAME_LIST.value: [],
                                           ConfigParams.VELOCITY_LIST.value : {},
                                           ConfigParams.STEERING_LIST.value : {},
                                           ConfigParams.CHANGE_START.value : None,
                                           ConfigParams.ALT_DIR.value : None,
                                           ConfigParams.ACTION_SPACE_PATH.value : 'custom_files/model_metadata.json',
                                           ConfigParams.REWARD.value : None,
                                           ConfigParams.AGENT_NAME.value : 'racecar'}}

        agent_list = list()
        agent_list.append(create_training_agent(agent_config))

        graph_manager, robomaker_hyperparams_json = get_graph_manager(hp_dict=sm_hyperparams_dict,
                                                                      agent_list=agent_list,
                                                                      run_phase_subject=None)

        s3_client.upload_hyperparameters(robomaker_hyperparams_json)
        logger.info("Uploaded hyperparameters.json to S3")

        # Attach sample collector to graph_manager only if sample count > 0
        max_sample_count = int(sm_hyperparams_dict.get("max_sample_count", 0))
        if max_sample_count > 0:
            sample_collector = SampleCollector(s3_client=s3_client, s3_prefix=args.s3_prefix,
                                               max_sample_count=max_sample_count,
                                               sampling_frequency=int(sm_hyperparams_dict.get("sampling_frequency", 1)))
            graph_manager.sample_collector = sample_collector

    host_ip_address = utils.get_ip_from_host()
    s3_client.write_ip_config(host_ip_address)
    logger.info("Uploaded IP address information to S3: %s" % host_ip_address)
    use_pretrained_model = args.pretrained_s3_bucket and args.pretrained_s3_prefix
    if use_pretrained_model:
        # Handle backward compatibility
        _, _, version = parse_model_metadata(model_metadata_local_path)
        if float(version) < float(utils.SIMAPP_VERSION) and \
        not utils.has_current_ckpnt_name(args.pretrained_s3_bucket, args.pretrained_s3_prefix, args.aws_region):
            utils.make_compatible(args.pretrained_s3_bucket, args.pretrained_s3_prefix,
                                  args.aws_region, SyncFiles.TRAINER_READY.value)
        #Select the optimal model for the starting weights
        utils.do_model_selection(s3_bucket=args.s3_bucket,
                                 s3_prefix=args.s3_prefix,
                                 region=args.aws_region)

        ds_params_instance_pretrained = S3BotoDataStoreParameters(aws_region=args.aws_region,
                                                                  bucket_names={'agent':args.pretrained_s3_bucket},
                                                                  base_checkpoint_dir=args.pretrained_checkpoint_dir,
                                                                  s3_folders={'agent':args.pretrained_s3_prefix})
        data_store_pretrained = S3BotoDataStore(ds_params_instance_pretrained, graph_manager, True)
        data_store_pretrained.load_from_store()

    memory_backend_params = RedisPubSubMemoryBackendParameters(redis_address="localhost",
                                                               redis_port=6379,
                                                               run_type=str(RunType.TRAINER),
                                                               channel=args.s3_prefix)

    graph_manager.memory_backend_params = memory_backend_params

    ds_params_instance = S3BotoDataStoreParameters(aws_region=args.aws_region,
                                                   bucket_names={'agent':args.s3_bucket},
                                                   base_checkpoint_dir=args.checkpoint_dir,
                                                   s3_folders={'agent':args.s3_prefix})

    graph_manager.data_store_params = ds_params_instance

    graph_manager.data_store = S3BotoDataStore(ds_params_instance, graph_manager)

    task_parameters = TaskParameters()
    task_parameters.experiment_path = SM_MODEL_OUTPUT_DIR
    task_parameters.checkpoint_save_secs = 20
    if use_pretrained_model:
        task_parameters.checkpoint_restore_path = args.pretrained_checkpoint_dir
    task_parameters.checkpoint_save_dir = args.checkpoint_dir

    training_worker(
        graph_manager=graph_manager,
        task_parameters=task_parameters,
        user_batch_size=json.loads(robomaker_hyperparams_json)["batch_size"],
        user_episode_per_rollout=json.loads(robomaker_hyperparams_json)["num_episodes_between_training"]
    )

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        utils.log_and_exit("Training worker exited with exception: {}".format(ex),
                           utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)
