import argparse
import json
import logging
import os
import shutil
import time
import traceback
import sys
import numpy as np
import subprocess

import markov.defaults as defaults
import markov.deepracer_memory as deepracer_memory

from markov.s3_client import SageS3Client
from markov.utils import get_ip_from_host, load_model_metadata, DoorMan
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters

from rl_coach import core_types
from rl_coach.base_parameters import TaskParameters, DistributedCoachSynchronizationType, Frameworks
from rl_coach.logger import screen
from rl_coach.memories.backend.redis import RedisPubSubMemoryBackendParameters
from rl_coach.utils import short_dynamic_import
from markov import utils
from gym.envs.registration import register

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

logger = utils.Logger(__name__, logging.INFO).get_logger()

PRETRAINED_MODEL_DIR = "./pretrained_checkpoint"
SM_MODEL_OUTPUT_DIR = os.environ.get("ALGO_MODEL_DIR", "/opt/ml/model")
CUSTOM_FILES_PATH = "./custom_files"

if not os.path.exists(CUSTOM_FILES_PATH):
    os.makedirs(CUSTOM_FILES_PATH)


def data_store_ckpt_save(data_store):
    while True:
        data_store.save_to_store()
        time.sleep(10)


def training_worker(graph_manager, checkpoint_dir, use_pretrained_model, framework, memory_backend_params):
    """
    restore a checkpoint then perform rollouts using the restored model
    """
    # initialize graph
    task_parameters = TaskParameters()
    task_parameters.__dict__['checkpoint_save_dir'] = checkpoint_dir
    task_parameters.__dict__['checkpoint_save_secs'] = 20
    task_parameters.__dict__['experiment_path'] = SM_MODEL_OUTPUT_DIR


    if framework.lower() == "mxnet":
        task_parameters.framework_type = Frameworks.mxnet
        if hasattr(graph_manager, 'agent_params'):
            for network_parameters in graph_manager.agent_params.network_wrappers.values():
                network_parameters.framework = Frameworks.mxnet
        elif hasattr(graph_manager, 'agents_params'):
            for ap in graph_manager.agents_params:
                for network_parameters in ap.network_wrappers.values():
                    network_parameters.framework = Frameworks.mxnet

    if use_pretrained_model:
        task_parameters.__dict__['checkpoint_restore_dir'] = PRETRAINED_MODEL_DIR

    graph_manager.create_graph(task_parameters)

    # save randomly initialized graph
    graph_manager.save_checkpoint()

    # training loop
    steps = 0

    graph_manager.memory_backend = deepracer_memory.DeepRacerTrainerBackEnd(memory_backend_params)

    # To handle SIGTERM
    door_man = DoorMan()

    try:
        while steps < graph_manager.improve_steps.num_steps:
            graph_manager.phase = core_types.RunPhase.TRAIN
            graph_manager.fetch_from_worker(graph_manager.agent_params.algorithm.num_consecutive_playing_steps)
            graph_manager.phase = core_types.RunPhase.UNDEFINED

            if graph_manager.should_train():
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
                #! TODO handle NaN's on a per agent level for distributed training
                if rollout_has_nan:
                    utils.json_format_logger("NaN detected in loss function, aborting training. Job failed!",
                                 **utils.build_system_error_dict(utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                                                                 utils.SIMAPP_EVENT_ERROR_CODE_503))
                    sys.exit(1)

                if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
                    graph_manager.save_checkpoint()
                else:
                    graph_manager.occasionally_save_checkpoint()
                # Clear any data stored in signals that is no longer necessary
                graph_manager.reset_internal_state()

            if door_man.terminate_now:
                utils.json_format_logger("Received SIGTERM. Checkpointing before exiting.",
                           **utils.build_system_error_dict(utils.SIMAPP_TRAINING_WORKER_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_500))
                graph_manager.save_checkpoint()
                break

    except Exception as e:
        utils.json_format_logger("An error occured while training: {}. Job failed!.".format(e),
                   **utils.build_system_error_dict(utils.SIMAPP_TRAINING_WORKER_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_503))
        traceback.print_exc()
        sys.exit(1)
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

    start_redis_server()
    args, unknown = parser.parse_known_args()

    s3_client = SageS3Client(bucket=args.s3_bucket, s3_prefix=args.s3_prefix, aws_region=args.aws_region)

    # Load the model metadata
    model_metadata_local_path = os.path.join(CUSTOM_FILES_PATH, 'model_metadata.json')
    load_model_metadata(s3_client, args.model_metadata_s3_key, model_metadata_local_path)
    s3_client.upload_file(os.path.normpath("%s/model/model_metadata.json" % args.s3_prefix), model_metadata_local_path)
    shutil.copy2(model_metadata_local_path, SM_MODEL_OUTPUT_DIR)

    # Register the gym enviroment, this will give clients the ability to creat the enviroment object
    register(id=defaults.ENV_ID, entry_point=defaults.ENTRY_POINT,
             max_episode_steps=defaults.MAX_STEPS, reward_threshold=defaults.THRESHOLD)

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
        from markov.sagemaker_graph_manager import get_graph_manager
        params_blob = os.environ.get('SM_TRAINING_ENV', '')
        if params_blob:
            params = json.loads(params_blob)
            sm_hyperparams_dict = params["hyperparameters"]
        else:
            sm_hyperparams_dict = {}
        graph_manager, robomaker_hyperparams_json = get_graph_manager(**sm_hyperparams_dict)
        s3_client.upload_hyperparameters(robomaker_hyperparams_json)
        logger.info("Uploaded hyperparameters.json to S3")

    host_ip_address = get_ip_from_host()
    s3_client.write_ip_config(host_ip_address)
    logger.info("Uploaded IP address information to S3: %s" % host_ip_address)

    use_pretrained_model = False
    if args.pretrained_s3_bucket and args.pretrained_s3_prefix:
        s3_client_pretrained = SageS3Client(bucket=args.pretrained_s3_bucket,
                                            s3_prefix=args.pretrained_s3_prefix,
                                            aws_region=args.aws_region)
        use_pretrained_model = s3_client_pretrained.download_model(args.pretrained_checkpoint_dir)

    memory_backend_params = RedisPubSubMemoryBackendParameters(redis_address="localhost",
                                                               redis_port=6379,
                                                               run_type='trainer',
                                                               channel=args.s3_prefix)

    ds_params_instance = S3BotoDataStoreParameters(bucket_name=args.s3_bucket,
                                                   checkpoint_dir=args.checkpoint_dir, aws_region=args.aws_region,
                                                   s3_folder=args.s3_prefix)
    graph_manager.data_store_params = ds_params_instance

    data_store = S3BotoDataStore(ds_params_instance)
    data_store.graph_manager = graph_manager
    graph_manager.data_store = data_store

    training_worker(
        graph_manager=graph_manager,
        checkpoint_dir=args.checkpoint_dir,
        use_pretrained_model=use_pretrained_model,
        framework=args.framework,
        memory_backend_params=memory_backend_params
    )


if __name__ == '__main__':
    main()
