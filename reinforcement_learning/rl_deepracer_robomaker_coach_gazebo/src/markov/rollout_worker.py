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
from rl_coach.rollout_worker import wait_for_checkpoint, wait_for_trainer_ready, should_stop
from rl_coach.utils import short_dynamic_import
from rl_coach.core_types import EnvironmentEpisodes

from markov import utils, utils_parse_model_metadata
from markov.constants import DEEPRACER_CHKPNT_KEY_SUFFIX, ROLLOUT_WORKER_PROFILER_PATH
from markov.log_handler.logger import Logger
from markov.log_handler.exception_handler import log_and_exit, simapp_exit_gracefully
from markov.log_handler.constants import (SIMAPP_SIMULATION_WORKER_EXCEPTION,
                                          SIMAPP_EVENT_ERROR_CODE_500,
                                          SIMAPP_EVENT_ERROR_CODE_400)
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.rollout_agent_factory import create_rollout_agent, create_obstacles_agent, create_bot_cars_agent
from markov.agents.utils import RunPhaseSubject
from markov.log_handler.deepracer_exceptions import GenericRolloutError, GenericRolloutException
from markov.environments.constants import VELOCITY_TOPICS, STEERING_TOPICS, LINK_NAMES
from markov.metrics.s3_metrics import TrainingMetrics
from markov.rollout_utils import (PhaseObserver, signal_robomaker_markov_package_ready,
                                  configure_environment_randomizer, get_robomaker_profiler_env)
from markov.metrics.s3_writer import S3Writer
from markov.metrics.iteration_data import IterationData
from markov.metrics.constants import MetricsS3Keys, IterationDataLocalFileNames, ITERATION_DATA_LOCAL_FILE_PATH
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.s3_client import SageS3Client
from markov.sagemaker_graph_manager import get_graph_manager
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.camera_utils import configure_camera
from markov.deepracer_memory import DeepRacerRedisPubSubMemoryBackendParameters

from std_srvs.srv import Empty, EmptyRequest

logger = Logger(__name__, logging.INFO).get_logger()

MIN_EVAL_TRIALS = 5

CUSTOM_FILES_PATH = "./custom_files"
if not os.path.exists(CUSTOM_FILES_PATH):
    os.makedirs(CUSTOM_FILES_PATH)

IS_PROFILER_ON, PROFILER_S3_BUCKET, PROFILER_S3_PREFIX = get_robomaker_profiler_env()

def fstring_decoded_reward_function(reward_function_local_path_preprocessed):
    """ python 3.6 supports fstring and console lambda function validates using python3.6.
    But all the simapp code is runs in python3.5 which does not support fstring. This funciton
    support fstring in python 3.5

    Arguments:
        reward_function_local_path_preprocessed {[str]} -- [Reward function file path]
    """
    try:
        reward_function_local_path_processed = os.path.join(CUSTOM_FILES_PATH, "customer_reward_function.py")
        with open(reward_function_local_path_preprocessed, 'rb') as filepointer:
            text, _ = future_fstrings.fstring_decode(filepointer.read())
        with open(reward_function_local_path_processed, 'wb') as filepointer:
            filepointer.write(text.encode('UTF-8'))
    except Exception as e:
        log_and_exit("Failed to decode the fstring format in reward function: {}"
                         .format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)

def download_customer_reward_function(s3_client, reward_file_s3_key):
    """ Download the customer reward function from s3 bucket
    Arguments:
        s3_client {[s3_session]} -- [S3 session object]
        reward_file_s3_key {[str]} -- [Reward function file path in s3]
    """
    reward_function_local_path_preprocessed = os.path.join(CUSTOM_FILES_PATH,
                                                           "customer_reward_function_preprocessed.py")
    success_reward_function_download = s3_client.download_file(s3_key=reward_file_s3_key,
                                                               local_path=reward_function_local_path_preprocessed)
    if not success_reward_function_download:
        log_and_exit("Unable to download the reward function code.",
                     SIMAPP_SIMULATION_WORKER_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_400)
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

def exit_if_trainer_done(checkpoint_dir, s3_writer, rollout_idx):
    '''Helper method that shutsdown the sim app if the trainer is done
       checkpoint_dir - direcotry where the done file would be downloaded to
    '''
    if should_stop(checkpoint_dir):
        is_save_mp4_enabled = rospy.get_param('MP4_S3_BUCKET', None) and rollout_idx == 0
        if is_save_mp4_enabled:
            unsubscribe_from_save_mp4 = ServiceProxyWrapper('/racecar/save_mp4/unsubscribe_from_save_mp4', Empty)
            unsubscribe_from_save_mp4(EmptyRequest())
        s3_writer.upload_to_s3()
        logger.info("Received termination signal from trainer. Goodbye.")
        simapp_exit_gracefully()


def rollout_worker(graph_manager, num_workers, rollout_idx, task_parameters, s3_writer):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    if not graph_manager.data_store:
        raise AttributeError("None type for data_store object")

    data_store = graph_manager.data_store

    checkpoint_dir = task_parameters.checkpoint_restore_path
    wait_for_checkpoint(checkpoint_dir, data_store)
    wait_for_trainer_ready(checkpoint_dir, data_store)
    # Make the clients that will allow us to pause and unpause the physics
    rospy.wait_for_service('/gazebo/pause_physics')
    rospy.wait_for_service('/gazebo/unpause_physics')
    rospy.wait_for_service('/racecar/save_mp4/subscribe_to_save_mp4')
    rospy.wait_for_service('/racecar/save_mp4/unsubscribe_from_save_mp4')
    pause_physics = ServiceProxyWrapper('/gazebo/pause_physics', Empty)
    unpause_physics = ServiceProxyWrapper('/gazebo/unpause_physics', Empty)
    subscribe_to_save_mp4 = ServiceProxyWrapper('/racecar/save_mp4/subscribe_to_save_mp4', Empty)
    unsubscribe_from_save_mp4 = ServiceProxyWrapper('/racecar/save_mp4/unsubscribe_from_save_mp4', Empty)
    graph_manager.create_graph(task_parameters=task_parameters, stop_physics=pause_physics,
                               start_physics=unpause_physics, empty_service_call=EmptyRequest)

    chkpt_state_reader = CheckpointStateReader(checkpoint_dir, checkpoint_state_optional=False)
    last_checkpoint = chkpt_state_reader.get_latest().num

    # this worker should play a fraction of the total playing steps per rollout
    episode_steps_per_rollout = graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps
    act_steps = int(episode_steps_per_rollout / num_workers)
    if rollout_idx < episode_steps_per_rollout % num_workers:
        act_steps += 1
    act_steps = EnvironmentEpisodes(act_steps)

    configure_environment_randomizer()

    for _ in range((graph_manager.improve_steps / act_steps.num_steps).num_steps):
        # Collect profiler information only IS_PROFILER_ON is true
        with utils.Profiler(s3_bucket=PROFILER_S3_BUCKET, s3_prefix=PROFILER_S3_PREFIX,
                            output_local_path=ROLLOUT_WORKER_PROFILER_PATH, enable_profiling=IS_PROFILER_ON):
            graph_manager.phase = RunPhase.TRAIN
            exit_if_trainer_done(checkpoint_dir, s3_writer, rollout_idx)
            unpause_physics(EmptyRequest())
            graph_manager.reset_internal_state(True)
            graph_manager.act(act_steps, wait_for_full_episodes=graph_manager.agent_params.algorithm.act_for_full_episodes)
            graph_manager.reset_internal_state(True)
            time.sleep(1)
            pause_physics(EmptyRequest())

            graph_manager.phase = RunPhase.UNDEFINED
            new_checkpoint = -1
            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type\
                    == DistributedCoachSynchronizationType.SYNC:
                unpause_physics(EmptyRequest())
                is_save_mp4_enabled = rospy.get_param('MP4_S3_BUCKET', None) and rollout_idx == 0
                if is_save_mp4_enabled:
                    subscribe_to_save_mp4(EmptyRequest())
                if rollout_idx == 0:
                    for _ in range(MIN_EVAL_TRIALS):
                        graph_manager.evaluate(EnvironmentSteps(1))

                while new_checkpoint < last_checkpoint + 1:
                    exit_if_trainer_done(checkpoint_dir, s3_writer, rollout_idx)
                    if rollout_idx == 0:
                        graph_manager.evaluate(EnvironmentSteps(1))
                    new_checkpoint = data_store.get_chkpoint_num('agent')
                if is_save_mp4_enabled:
                    unsubscribe_from_save_mp4(EmptyRequest())
                s3_writer.upload_to_s3()

                pause_physics(EmptyRequest())
                data_store.load_from_store(expected_checkpoint_number=last_checkpoint+1)
                graph_manager.restore_checkpoint()

            if graph_manager.agent_params.algorithm.distributed_coach_synchronization_type\
                    == DistributedCoachSynchronizationType.ASYNC:
                if new_checkpoint > last_checkpoint:
                    graph_manager.restore_checkpoint()

            last_checkpoint = new_checkpoint

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
    parser.add_argument('--num_workers',
                        help="(int) The number of workers started in this pool",
                        type=int,
                        default=int(rospy.get_param("NUM_WORKERS", 1)))
    parser.add_argument('--rollout_idx',
                        help="(int) The index of current rollout worker",
                        type=int,
                        default=0)
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
    # For training job, reset is not allowed. penalty_seconds, off_track_penalty, and
    # collision_penalty will all be 0 be default
    parser.add_argument('--number_of_resets',
                        help='(integer) Number of resets',
                        type=int,
                        default=int(rospy.get_param("NUMBER_OF_RESETS", 0)))
    parser.add_argument('--penalty_seconds',
                        help='(float) penalty second',
                        type=float,
                        default=float(rospy.get_param("PENALTY_SECONDS", 0.0)))
    parser.add_argument('--job_type',
                        help='(string) job type',
                        type=str,
                        default=rospy.get_param("JOB_TYPE", "TRAINING"))
    parser.add_argument('--is_continuous',
                        help='(boolean) is continous after lap completion',
                        type=bool,
                        default=utils.str2bool(rospy.get_param("IS_CONTINUOUS", False)))
    parser.add_argument('--race_type',
                        help='(string) Race type',
                        type=str,
                        default=rospy.get_param("RACE_TYPE", "TIME_TRIAL"))
    parser.add_argument('--off_track_penalty',
                        help='(float) off track penalty second',
                        type=float,
                        default=float(rospy.get_param("OFF_TRACK_PENALTY", 0.0)))
    parser.add_argument('--collision_penalty',
                        help='(float) collision penalty second',
                        type=float,
                        default=float(rospy.get_param("COLLISION_PENALTY", 0.0)))

    args = parser.parse_args()

    s3_client = SageS3Client(bucket=args.s3_bucket, s3_prefix=args.s3_prefix, aws_region=args.aws_region)
    logger.info("S3 bucket: %s", args.s3_bucket)
    logger.info("S3 prefix: %s", args.s3_prefix)

    # Load the model metadata
    model_metadata_local_path = os.path.join(CUSTOM_FILES_PATH, 'model_metadata.json')
    utils.load_model_metadata(s3_client, args.model_metadata_s3_key, model_metadata_local_path)

    # Download and import reward function
    if not args.reward_file_s3_key:
        log_and_exit("Reward function code S3 key not available for S3 bucket {} and prefix {}"
                         .format(args.s3_bucket, args.s3_prefix),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)
    download_customer_reward_function(s3_client, args.reward_file_s3_key)

    try:
        from custom_files.customer_reward_function import reward_function
    except Exception as e:
        log_and_exit("Failed to import user's reward_function: {}"
                         .format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_400)

    # Instantiate Cameras
    configure_camera(namespaces=['racecar'])

    preset_file_success, _ = download_custom_files_if_present(s3_client, args.s3_prefix)

    #! TODO each agent should have own config
    _, _, version = utils_parse_model_metadata.parse_model_metadata(model_metadata_local_path)
    agent_config = {
        'model_metadata': model_metadata_local_path,
        ConfigParams.CAR_CTRL_CONFIG.value: {
            ConfigParams.LINK_NAME_LIST.value: LINK_NAMES,
            ConfigParams.VELOCITY_LIST.value : VELOCITY_TOPICS,
            ConfigParams.STEERING_LIST.value : STEERING_TOPICS,
            ConfigParams.CHANGE_START.value : utils.str2bool(rospy.get_param('CHANGE_START_POSITION', True)),
            ConfigParams.ALT_DIR.value : utils.str2bool(rospy.get_param('ALTERNATE_DRIVING_DIRECTION', False)),
            ConfigParams.ACTION_SPACE_PATH.value : 'custom_files/model_metadata.json',
            ConfigParams.REWARD.value : reward_function,
            ConfigParams.AGENT_NAME.value : 'racecar',
            ConfigParams.VERSION.value : version,
            ConfigParams.NUMBER_OF_RESETS.value: args.number_of_resets,
            ConfigParams.PENALTY_SECONDS.value: args.penalty_seconds,
            ConfigParams.NUMBER_OF_TRIALS.value: None,
            ConfigParams.IS_CONTINUOUS.value: args.is_continuous,
            ConfigParams.RACE_TYPE.value: args.race_type,
            ConfigParams.COLLISION_PENALTY.value: args.collision_penalty,
            ConfigParams.OFF_TRACK_PENALTY.value: args.off_track_penalty
        }
    }

    #! TODO each agent should have own s3 bucket
    step_metrics_prefix = rospy.get_param('SAGEMAKER_SHARED_S3_PREFIX')
    if args.num_workers > 1:
        step_metrics_prefix = os.path.join(step_metrics_prefix, str(args.rollout_idx))
    metrics_s3_config = {MetricsS3Keys.METRICS_BUCKET.value: rospy.get_param('METRICS_S3_BUCKET'),
                         MetricsS3Keys.METRICS_KEY.value:  rospy.get_param('METRICS_S3_OBJECT_KEY'),
                         MetricsS3Keys.REGION.value: rospy.get_param('AWS_REGION')}
    metrics_s3_model_cfg = {MetricsS3Keys.METRICS_BUCKET.value: args.s3_bucket,
                            MetricsS3Keys.METRICS_KEY.value: os.path.join(args.s3_prefix,
                                                                          DEEPRACER_CHKPNT_KEY_SUFFIX),
                            MetricsS3Keys.REGION.value: args.aws_region}
    run_phase_subject = RunPhaseSubject()

    agent_list = list()
    agent_list.append(create_rollout_agent(agent_config,
                                           TrainingMetrics(agent_name='agent',
                                                           s3_dict_metrics=metrics_s3_config,
                                                           s3_dict_model=metrics_s3_model_cfg,
                                                           ckpnt_dir=args.checkpoint_dir,
                                                           run_phase_sink=run_phase_subject,
                                                           use_model_picker=(args.rollout_idx == 0)),
                                           run_phase_subject))
    agent_list.append(create_obstacles_agent())
    agent_list.append(create_bot_cars_agent())
    # ROS service to indicate all the robomaker markov packages are ready for consumption
    signal_robomaker_markov_package_ready()

    PhaseObserver('/agent/training_phase', run_phase_subject)

    aws_region = rospy.get_param('AWS_REGION', args.aws_region)
    simtrace_s3_bucket = rospy.get_param('SIMTRACE_S3_BUCKET', None)
    mp4_s3_bucket = rospy.get_param('MP4_S3_BUCKET', None) if args.rollout_idx == 0 else None
    if simtrace_s3_bucket:
        simtrace_s3_object_prefix = rospy.get_param('SIMTRACE_S3_PREFIX')
        if args.num_workers > 1:
            simtrace_s3_object_prefix = os.path.join(simtrace_s3_object_prefix, str(args.rollout_idx))
    if mp4_s3_bucket:
        mp4_s3_object_prefix = rospy.get_param('MP4_S3_OBJECT_PREFIX')

    s3_writer_job_info = []
    if simtrace_s3_bucket:
        s3_writer_job_info.append(
            IterationData('simtrace', simtrace_s3_bucket, simtrace_s3_object_prefix, aws_region,
                          os.path.join(ITERATION_DATA_LOCAL_FILE_PATH, 'agent',
                                       IterationDataLocalFileNames.SIM_TRACE_TRAINING_LOCAL_FILE.value)))
    if mp4_s3_bucket:
        s3_writer_job_info.extend([
            IterationData('pip', mp4_s3_bucket, mp4_s3_object_prefix, aws_region,
                          os.path.join(
                              ITERATION_DATA_LOCAL_FILE_PATH, 'agent',
                              IterationDataLocalFileNames.CAMERA_PIP_MP4_VALIDATION_LOCAL_PATH.value)),
            IterationData('45degree', mp4_s3_bucket, mp4_s3_object_prefix, aws_region,
                          os.path.join(
                              ITERATION_DATA_LOCAL_FILE_PATH, 'agent',
                              IterationDataLocalFileNames.CAMERA_45DEGREE_MP4_VALIDATION_LOCAL_PATH.value)),
            IterationData('topview', mp4_s3_bucket, mp4_s3_object_prefix, aws_region,
                          os.path.join(
                              ITERATION_DATA_LOCAL_FILE_PATH, 'agent',
                              IterationDataLocalFileNames.CAMERA_TOPVIEW_MP4_VALIDATION_LOCAL_PATH.value))])

    s3_writer = S3Writer(job_info=s3_writer_job_info)

    redis_ip = s3_client.get_ip()
    logger.info("Received IP from SageMaker successfully: %s", redis_ip)

    # Download hyperparameters from SageMaker
    hyperparameters_file_success = False
    hyperparams_s3_key = os.path.normpath(args.s3_prefix + "/ip/hyperparameters.json")
    hyperparameters_file_success = s3_client.download_file(s3_key=hyperparams_s3_key,
                                                           local_path="hyperparameters.json")
    sm_hyperparams_dict = {}
    if hyperparameters_file_success:
        logger.info("Received Sagemaker hyperparameters successfully!")
        with open("hyperparameters.json") as filepointer:
            sm_hyperparams_dict = json.load(filepointer)
    else:
        logger.info("SageMaker hyperparameters not found.")

    enable_domain_randomization = utils.str2bool(rospy.get_param('ENABLE_DOMAIN_RANDOMIZATION', False))
    if preset_file_success:
        preset_location = os.path.join(CUSTOM_FILES_PATH, "preset.py")
        preset_location += ":graph_manager"
        graph_manager = short_dynamic_import(preset_location, ignore_module_case=True)
        logger.info("Using custom preset file!")
    else:
        graph_manager, _ = get_graph_manager(hp_dict=sm_hyperparams_dict, agent_list=agent_list,
                                             run_phase_subject=run_phase_subject,
                                             enable_domain_randomization=enable_domain_randomization)

    # If num_episodes_between_training is smaller than num_workers then cancel worker early.
    episode_steps_per_rollout = graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps
    # Reduce number of workers if allocated more than num_episodes_between_training
    if args.num_workers > episode_steps_per_rollout:
        logger.info("Excess worker allocated. Reducing from {} to {}...".format(args.num_workers,
                                                                                episode_steps_per_rollout))
        args.num_workers = episode_steps_per_rollout
    if args.rollout_idx >= episode_steps_per_rollout or args.rollout_idx >= args.num_workers:
        err_msg_format = "Exiting excess worker..."
        err_msg_format += "(rollout_idx[{}] >= num_workers[{}] or num_episodes_between_training[{}])"
        logger.info(err_msg_format.format(args.rollout_idx, args.num_workers, episode_steps_per_rollout))
        # Close the down the job
        utils.cancel_simulation_job(os.environ.get('AWS_ROBOMAKER_SIMULATION_JOB_ARN'),
                                    rospy.get_param('AWS_REGION'))

    memory_backend_params = DeepRacerRedisPubSubMemoryBackendParameters(redis_address=redis_ip,
                                                                        redis_port=6379,
                                                                        run_type=str(RunType.ROLLOUT_WORKER),
                                                                        channel=args.s3_prefix,
                                                                        num_workers=args.num_workers,
                                                                        rollout_idx=args.rollout_idx)

    graph_manager.memory_backend_params = memory_backend_params

    ds_params_instance = S3BotoDataStoreParameters(aws_region=args.aws_region,
                                                   bucket_names={'agent':args.s3_bucket},
                                                   base_checkpoint_dir=args.checkpoint_dir,
                                                   s3_folders={'agent':args.s3_prefix})

    graph_manager.data_store = S3BotoDataStore(ds_params_instance, graph_manager)

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = args.checkpoint_dir

    rollout_worker(
        graph_manager=graph_manager,
        num_workers=args.num_workers,
        rollout_idx=args.rollout_idx,
        task_parameters=task_parameters,
        s3_writer=s3_writer
    )

if __name__ == '__main__':
    try:
        rospy.init_node('rl_coach', anonymous=True)
        main()
    except ValueError as err:
        if utils.is_error_bad_ckpnt(err):
            log_and_exit("User modified model: {}".format(err),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        else:
            log_and_exit("Rollout worker value error: {}".format(err),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
    except GenericRolloutError as ex:
        ex.log_except_and_exit()
    except GenericRolloutException as ex:
        ex.log_except_and_exit()
    except Exception as ex:
        log_and_exit("Rollout worker exited with exception: {}".format(ex),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)
