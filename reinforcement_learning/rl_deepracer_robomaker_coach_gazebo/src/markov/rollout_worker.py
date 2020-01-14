"""
this rollout worker:

- restores a model from disk
- evaluates a predefined number of episodes
- contributes them to a distributed memory
- exits
"""
import argparse
import json
import logging
import os
import time
import rospy
import future_fstrings

from rl_coach.base_parameters import TaskParameters, DistributedCoachSynchronizationType, RunType
from rl_coach.checkpoint import CheckpointStateReader
from rl_coach.core_types import RunPhase, EnvironmentSteps
from rl_coach.logger import screen
from rl_coach.memories.backend.redis import RedisPubSubMemoryBackendParameters
from rl_coach.rollout_worker import wait_for_checkpoint, wait_for_trainer_ready, should_stop
from rl_coach.utils import short_dynamic_import

from markov import utils, utils_parse_model_metadata
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.rollout_agent_factory import create_rollout_agent, create_obstacles_agent, create_bot_cars_agent
from markov.environments.constants import VELOCITY_TOPICS, STEERING_TOPICS, LINK_NAMES
from markov.metrics.s3_metrics import TrainingMetrics
from markov.metrics.constants import MetricsS3Keys
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.s3_client import SageS3Client
from markov.sagemaker_graph_manager import get_graph_manager
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.camera_utils import configure_camera

import markov.deepracer_memory_multi as deepracer_memory

from std_srvs.srv import Empty, EmptyRequest

logger = utils.Logger(__name__, logging.INFO).get_logger()

TRAINING_SIMTRACE_DATA_S3_OBJECT_KEY = "sim_inference_logs/TrainingSimTraceData.csv"

CUSTOM_FILES_PATH = "./custom_files"
if not os.path.exists(CUSTOM_FILES_PATH):
    os.makedirs(CUSTOM_FILES_PATH)

def fstring_decoded_reward_function(reward_function_local_path_preprocessed):
    try:
        reward_function_local_path_processed = os.path.join(CUSTOM_FILES_PATH, "customer_reward_function.py")
        with open(reward_function_local_path_preprocessed, 'rb') as filepointer:
            text, _ = future_fstrings.fstring_decode(filepointer.read())
        with open(reward_function_local_path_processed, 'wb') as filepointer:
            filepointer.write(text.encode('UTF-8'))
    except Exception as e:
        utils.log_and_exit("Failed to decode the fstring format in reward function: {}".format(e),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)

def download_customer_reward_function(s3_client, reward_file_s3_key):
    reward_function_local_path_preprocessed = os.path.join(CUSTOM_FILES_PATH, "customer_reward_function_preprocessed.py")
    success_reward_function_download = s3_client.download_file(s3_key=reward_file_s3_key, local_path=reward_function_local_path_preprocessed)
    if not success_reward_function_download:
        utils.log_and_exit("Unable to download the reward function code.",
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_400)
    fstring_decoded_reward_function(reward_function_local_path_preprocessed)

def download_custom_files_if_present(s3_client, s3_prefix):
    environment_file_s3_key = os.path.normpath(s3_prefix + "/environments/deepracer_racetrack_env.py")
    environment_local_path = os.path.join(CUSTOM_FILES_PATH, "deepracer_racetrack_env.py")
    success_environment_download = s3_client.download_file(s3_key=environment_file_s3_key,
                                                           local_path=environment_local_path)

    preset_file_s3_key = os.path.normpath(s3_prefix + "/presets/preset.py")
    preset_local_path = os.path.join(CUSTOM_FILES_PATH, "preset.py")
    success_preset_download = s3_client.download_file(s3_key=preset_file_s3_key,
                                                      local_path=preset_local_path)
    return success_preset_download, success_environment_download

def exit_if_trainer_done(checkpoint_dir):
    '''Helper method that shutsdown the sim app if the trainer is done
       checkpoint_dir - direcotry where the done file would be downloaded to
    '''
    if should_stop(checkpoint_dir):
        logger.info("Received termination signal from trainer. Goodbye.")
        utils.simapp_exit_gracefully()


def rollout_worker(graph_manager, data_store, num_workers, task_parameters):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    if not data_store:
        raise AttributeError("None type for data_store object")

    checkpoint_dir = task_parameters.checkpoint_restore_path
    wait_for_checkpoint(checkpoint_dir, data_store)
    wait_for_trainer_ready(checkpoint_dir, data_store)
    # Make the clients that will allow us to pause and unpause the physics
    rospy.wait_for_service('/gazebo/pause_physics')
    rospy.wait_for_service('/gazebo/unpause_physics')
    pause_physics = ServiceProxyWrapper('/gazebo/pause_physics', Empty)
    unpause_physics = ServiceProxyWrapper('/gazebo/unpause_physics', Empty)
    graph_manager.create_graph(task_parameters=task_parameters, stop_physics=pause_physics,
                               start_physics=unpause_physics, empty_service_call=EmptyRequest)

    with graph_manager.phase_context(RunPhase.TRAIN):
        chkpt_state_reader = CheckpointStateReader(checkpoint_dir, checkpoint_state_optional=False)
        last_checkpoint = chkpt_state_reader.get_latest().num

        for level in graph_manager.level_managers:
            for agent in level.agents.values():
                agent.memory.memory_backend.set_current_checkpoint(last_checkpoint)

        # this worker should play a fraction of the total playing steps per rollout
        act_steps = 1
        while True:
            exit_if_trainer_done(checkpoint_dir)
            unpause_physics(EmptyRequest())
            graph_manager.reset_internal_state(True)
            graph_manager.act(EnvironmentSteps(num_steps=act_steps), wait_for_full_episodes=graph_manager.agent_params.algorithm.act_for_full_episodes)
            graph_manager.reset_internal_state(True)
            time.sleep(1)
            pause_physics(EmptyRequest())

            new_checkpoint = data_store.get_latest_checkpoint()
            if new_checkpoint and new_checkpoint > last_checkpoint:
                if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.SYNC:
                    exit_if_trainer_done(checkpoint_dir)
                    data_store.load_from_store(expected_checkpoint_number=last_checkpoint+1)
                    graph_manager.restore_checkpoint()

                if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type == DistributedCoachSynchronizationType.ASYNC:
                    graph_manager.restore_checkpoint()

                last_checkpoint = new_checkpoint
                for level in graph_manager.level_managers:
                    for agent in level.agents.values():
                        agent.memory.memory_backend.set_current_checkpoint(last_checkpoint)

def main():
    screen.set_use_colors(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir',
                        help='(string) Path to a folder containing a checkpoint to restore the model from.',
                        type=str,
                        default='./checkpoint')
    parser.add_argument('--s3_bucket',
                        help='(string) S3 bucket',
                        type=str,
                        default=rospy.get_param("SAGEMAKER_SHARED_S3_BUCKET", "gsaur-test"))
    parser.add_argument('--s3_prefix',
                        help='(string) S3 prefix',
                        type=str,
                        default=rospy.get_param("SAGEMAKER_SHARED_S3_PREFIX", "sagemaker"))
    parser.add_argument('--num-workers',
                        help="(int) The number of workers started in this pool",
                        type=int,
                        default=1)
    parser.add_argument('-r', '--redis_ip',
                        help="(string) IP or host for the redis server",
                        default='localhost',
                        type=str)
    parser.add_argument('-rp', '--redis_port',
                        help="(int) Port of the redis server",
                        default=6379,
                        type=int)
    parser.add_argument('--aws_region',
                        help='(string) AWS region',
                        type=str,
                        default=rospy.get_param("AWS_REGION", "us-east-1"))
    parser.add_argument('--reward_file_s3_key',
                        help='(string) Reward File S3 Key',
                        type=str,
                        default=rospy.get_param("REWARD_FILE_S3_KEY", None))
    parser.add_argument('--model_metadata_s3_key',
                        help='(string) Model Metadata File S3 Key',
                        type=str,
                        default=rospy.get_param("MODEL_METADATA_FILE_S3_KEY", None))

    args = parser.parse_args()

    s3_client = SageS3Client(bucket=args.s3_bucket, s3_prefix=args.s3_prefix, aws_region=args.aws_region)
    logger.info("S3 bucket: %s" % args.s3_bucket)
    logger.info("S3 prefix: %s" % args.s3_prefix)

    # Load the model metadata
    model_metadata_local_path = os.path.join(CUSTOM_FILES_PATH, 'model_metadata.json')
    utils.load_model_metadata(s3_client, args.model_metadata_s3_key, model_metadata_local_path)

    # Download and import reward function
    if not args.reward_file_s3_key:
        utils.log_and_exit("Reward function code S3 key not available for S3 bucket {} and prefix {}"
                           .format(args.s3_bucket, args.s3_prefix),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)
    download_customer_reward_function(s3_client, args.reward_file_s3_key)

    try:
        from custom_files.customer_reward_function import reward_function
    except Exception as e:
        utils.log_and_exit("Failed to import user's reward_function: {}".format(e),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_400)

    # Instantiate Cameras
    configure_camera()

    redis_ip = s3_client.get_ip()
    logger.info("Received IP from SageMaker successfully: %s" % redis_ip)

    # Download hyperparameters from SageMaker
    hyperparameters_file_success = False
    hyperparams_s3_key = os.path.normpath(args.s3_prefix + "/ip/hyperparameters.json")
    hyperparameters_file_success = s3_client.download_file(s3_key=hyperparams_s3_key,
                                                           local_path="hyperparameters.json")
    sm_hyperparams_dict = {}
    if hyperparameters_file_success:
        logger.info("Received Sagemaker hyperparameters successfully!")
        with open("hyperparameters.json") as fp:
            sm_hyperparams_dict = json.load(fp)
    else:
        logger.info("SageMaker hyperparameters not found.")

    preset_file_success, _ = download_custom_files_if_present(s3_client, args.s3_prefix)

    #! TODO each agent should have own config
    _, _, version = utils_parse_model_metadata.parse_model_metadata(model_metadata_local_path)
    agent_config = {'model_metadata': model_metadata_local_path,
                    'car_ctrl_cnfig': {ConfigParams.LINK_NAME_LIST.value: LINK_NAMES,
                                       ConfigParams.VELOCITY_LIST.value : VELOCITY_TOPICS,
                                       ConfigParams.STEERING_LIST.value : STEERING_TOPICS,
                                       ConfigParams.CHANGE_START.value : utils.str2bool(rospy.get_param('CHANGE_START_POSITION', True)),
                                       ConfigParams.ALT_DIR.value : utils.str2bool(rospy.get_param('ALTERNATE_DRIVING_DIRECTION', False)),
                                       ConfigParams.ACTION_SPACE_PATH.value : 'custom_files/model_metadata.json',
                                       ConfigParams.REWARD.value : reward_function,
                                       ConfigParams.AGENT_NAME.value : 'racecar',
                                       ConfigParams.VERSION.value : version}}

    #! TODO each agent should have own s3 bucket
    metrics_s3_config = {MetricsS3Keys.METRICS_BUCKET.value: rospy.get_param('METRICS_S3_BUCKET'),
                         MetricsS3Keys.METRICS_KEY.value:  rospy.get_param('METRICS_S3_OBJECT_KEY'),
                         MetricsS3Keys.REGION.value: rospy.get_param('AWS_REGION'),
                         MetricsS3Keys.STEP_BUCKET.value: rospy.get_param('SAGEMAKER_SHARED_S3_BUCKET'),
                         MetricsS3Keys.STEP_KEY.value: os.path.join(rospy.get_param('SAGEMAKER_SHARED_S3_PREFIX'),
                                                                    TRAINING_SIMTRACE_DATA_S3_OBJECT_KEY)}

    agent_list = list()
    agent_list.append(create_rollout_agent(agent_config, TrainingMetrics(metrics_s3_config)))
    agent_list.append(create_obstacles_agent())
    agent_list.append(create_bot_cars_agent())

    if preset_file_success:
        preset_location = os.path.join(CUSTOM_FILES_PATH, "preset.py")
        preset_location += ":graph_manager"
        graph_manager = short_dynamic_import(preset_location, ignore_module_case=True)
        logger.info("Using custom preset file!")
    else:
        graph_manager, _ = get_graph_manager(sm_hyperparams_dict, agent_list)

    memory_backend_params = RedisPubSubMemoryBackendParameters(redis_address=redis_ip,
                                                               redis_port=6379,
                                                               run_type=str(RunType.ROLLOUT_WORKER),
                                                               channel=args.s3_prefix)

    graph_manager.memory_backend_params = memory_backend_params

    ds_params_instance = S3BotoDataStoreParameters(aws_region=args.aws_region,
                                                   bucket_name=args.s3_bucket,
                                                   checkpoint_dir=args.checkpoint_dir,
                                                   s3_folder=args.s3_prefix)

    data_store = S3BotoDataStore(ds_params_instance)
    data_store.graph_manager = graph_manager
    graph_manager.data_store = data_store

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = args.checkpoint_dir

    rollout_worker(
        graph_manager=graph_manager,
        data_store=data_store,
        num_workers=args.num_workers,
        task_parameters=task_parameters
    )

if __name__ == '__main__':
    try:
        rospy.init_node('rl_coach', anonymous=True)
        main()
    except ValueError as err:
        if utils.is_error_bad_ckpnt(err):
            utils.log_and_exit("User modified model: {}".format(err),
                               utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_400)
        else:
            utils.log_and_exit("Rollout worker value error: {}".format(err),
                               utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_500)
    except Exception as ex:
        utils.log_and_exit("Rollout worker exited with exception: {}".format(ex),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)
