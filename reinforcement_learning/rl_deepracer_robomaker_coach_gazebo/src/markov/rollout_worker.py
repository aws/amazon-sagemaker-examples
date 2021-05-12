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

import botocore
import rospy
from markov import utils
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.rollout_agent_factory import (
    create_bot_cars_agent,
    create_obstacles_agent,
    create_rollout_agent,
)
from markov.agents.utils import RunPhaseSubject
from markov.boto.s3.constants import (
    CAMERA_45DEGREE_LOCAL_PATH_FORMAT,
    CAMERA_PIP_MP4_LOCAL_PATH_FORMAT,
    CAMERA_TOPVIEW_LOCAL_PATH_FORMAT,
    HYPERPARAMETER_LOCAL_PATH_FORMAT,
    HYPERPARAMETER_S3_POSTFIX,
    IP_ADDRESS_LOCAL_PATH,
    MODEL_METADATA_LOCAL_PATH_FORMAT,
    REWARD_FUCTION_LOCAL_PATH_FORMAT,
    SIMTRACE_TRAINING_LOCAL_PATH_FORMAT,
    ModelMetadataKeys,
    SimtraceVideoNames,
)
from markov.boto.s3.files.checkpoint import Checkpoint
from markov.boto.s3.files.hyperparameters import Hyperparameters
from markov.boto.s3.files.ip_config import IpConfig
from markov.boto.s3.files.model_metadata import ModelMetadata
from markov.boto.s3.files.reward_function import RewardFunction
from markov.boto.s3.files.simtrace_video import SimtraceVideo
from markov.boto.s3.s3_client import S3Client
from markov.boto.s3.utils import get_s3_key
from markov.camera_utils import configure_camera
from markov.constants import ROLLOUT_WORKER_PROFILER_PATH
from markov.deepracer_memory import DeepRacerRedisPubSubMemoryBackendParameters
from markov.environments.constants import LINK_NAMES, STEERING_TOPICS, VELOCITY_TOPICS
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericRolloutError, GenericRolloutException
from markov.log_handler.exception_handler import log_and_exit, simapp_exit_gracefully
from markov.log_handler.logger import Logger
from markov.metrics.constants import MetricsS3Keys
from markov.metrics.iteration_data import IterationData
from markov.metrics.s3_metrics import TrainingMetrics
from markov.rollout_utils import (
    PhaseObserver,
    configure_environment_randomizer,
    get_robomaker_profiler_env,
    signal_robomaker_markov_package_ready,
)
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.sagemaker_graph_manager import get_graph_manager
from rl_coach.base_parameters import DistributedCoachSynchronizationType, RunType, TaskParameters
from rl_coach.checkpoint import CheckpointStateReader
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.logger import screen
from rl_coach.rollout_worker import should_stop
from rl_coach.utils import short_dynamic_import
from std_srvs.srv import Empty, EmptyRequest

logger = Logger(__name__, logging.INFO).get_logger()

MIN_EVAL_TRIALS = 5

CUSTOM_FILES_PATH = "./custom_files"
if not os.path.exists(CUSTOM_FILES_PATH):
    os.makedirs(CUSTOM_FILES_PATH)

IS_PROFILER_ON, PROFILER_S3_BUCKET, PROFILER_S3_PREFIX = get_robomaker_profiler_env()


def download_custom_files_if_present(s3_bucket, s3_prefix, aws_region):
    """download custom environment and preset files

    Args:
        s3_bucket (str): s3 bucket string
        s3_prefix (str): s3 prefix string
        aws_region (str): aws region string

    Returns:
        tuple (bool, bool): tuple of bool on whether preset and environemnt
        is downloaded successfully
    """
    success_environment_download, success_preset_download = False, False
    try:
        s3_client = S3Client(region_name=aws_region, max_retry_attempts=0)
        environment_file_s3_key = os.path.normpath(
            s3_prefix + "/environments/deepracer_racetrack_env.py"
        )
        environment_local_path = os.path.join(CUSTOM_FILES_PATH, "deepracer_racetrack_env.py")
        s3_client.download_file(
            bucket=s3_bucket, s3_key=environment_file_s3_key, local_path=environment_local_path
        )
        success_environment_download = True
    except botocore.exceptions.ClientError:
        pass
    try:
        preset_file_s3_key = os.path.normpath(s3_prefix + "/presets/preset.py")
        preset_local_path = os.path.join(CUSTOM_FILES_PATH, "preset.py")
        s3_client.download_file(
            bucket=s3_bucket, s3_key=preset_file_s3_key, local_path=preset_local_path
        )
        success_preset_download = True
    except botocore.exceptions.ClientError:
        pass
    return success_preset_download, success_environment_download


def exit_if_trainer_done(checkpoint_dir, simtrace_video_s3_writers, rollout_idx):
    """Helper method that shutsdown the sim app if the trainer is done
    checkpoint_dir - direcotry where the done file would be downloaded to
    """
    if should_stop(checkpoint_dir):
        is_save_mp4_enabled = rospy.get_param("MP4_S3_BUCKET", None) and rollout_idx == 0
        if is_save_mp4_enabled:
            unsubscribe_from_save_mp4 = ServiceProxyWrapper(
                "/racecar/save_mp4/unsubscribe_from_save_mp4", Empty
            )
            unsubscribe_from_save_mp4(EmptyRequest())
        # upload simtrace and mp4 into s3 bucket
        for s3_writer in simtrace_video_s3_writers:
            s3_writer.persist(utils.get_s3_kms_extra_args())
        logger.info("Received termination signal from trainer. Goodbye.")
        simapp_exit_gracefully()


def rollout_worker(
    graph_manager,
    num_workers,
    rollout_idx,
    task_parameters,
    simtrace_video_s3_writers,
    pause_physics,
    unpause_physics,
):
    """
    wait for first checkpoint then perform rollouts using the model
    """
    if not graph_manager.data_store:
        raise AttributeError("None type for data_store object")

    data_store = graph_manager.data_store

    # TODO change agent to specific agent name for multip agent case
    checkpoint_dir = os.path.join(task_parameters.checkpoint_restore_path, "agent")
    graph_manager.data_store.wait_for_checkpoints()
    graph_manager.data_store.wait_for_trainer_ready()
    # wait for the required cancel services to become available
    rospy.wait_for_service("/robomaker/job/cancel")
    # Make the clients that will allow us to pause and unpause the physics
    rospy.wait_for_service("/gazebo/pause_physics_dr")
    rospy.wait_for_service("/gazebo/unpause_physics_dr")
    rospy.wait_for_service("/racecar/save_mp4/subscribe_to_save_mp4")
    rospy.wait_for_service("/racecar/save_mp4/unsubscribe_from_save_mp4")

    subscribe_to_save_mp4 = ServiceProxyWrapper("/racecar/save_mp4/subscribe_to_save_mp4", Empty)
    unsubscribe_from_save_mp4 = ServiceProxyWrapper(
        "/racecar/save_mp4/unsubscribe_from_save_mp4", Empty
    )
    graph_manager.create_graph(
        task_parameters=task_parameters,
        stop_physics=pause_physics,
        start_physics=unpause_physics,
        empty_service_call=EmptyRequest,
    )

    chkpt_state_reader = CheckpointStateReader(checkpoint_dir, checkpoint_state_optional=False)
    last_checkpoint = chkpt_state_reader.get_latest().num

    # this worker should play a fraction of the total playing steps per rollout
    episode_steps_per_rollout = (
        graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps
    )
    act_steps = int(episode_steps_per_rollout / num_workers)
    if rollout_idx < episode_steps_per_rollout % num_workers:
        act_steps += 1
    act_steps = EnvironmentEpisodes(act_steps)

    configure_environment_randomizer()

    for _ in range((graph_manager.improve_steps / act_steps.num_steps).num_steps):
        # Collect profiler information only IS_PROFILER_ON is true
        with utils.Profiler(
            s3_bucket=PROFILER_S3_BUCKET,
            s3_prefix=PROFILER_S3_PREFIX,
            output_local_path=ROLLOUT_WORKER_PROFILER_PATH,
            enable_profiling=IS_PROFILER_ON,
        ):
            graph_manager.phase = RunPhase.TRAIN
            exit_if_trainer_done(checkpoint_dir, simtrace_video_s3_writers, rollout_idx)
            unpause_physics(EmptyRequest())
            graph_manager.reset_internal_state(True)
            graph_manager.act(
                act_steps,
                wait_for_full_episodes=graph_manager.agent_params.algorithm.act_for_full_episodes,
            )
            graph_manager.reset_internal_state(True)
            time.sleep(1)
            pause_physics(EmptyRequest())

            graph_manager.phase = RunPhase.UNDEFINED
            new_checkpoint = -1
            if (
                graph_manager.agent_params.algorithm.distributed_coach_synchronization_type
                == DistributedCoachSynchronizationType.SYNC
            ):
                unpause_physics(EmptyRequest())
                is_save_mp4_enabled = rospy.get_param("MP4_S3_BUCKET", None) and rollout_idx == 0
                if is_save_mp4_enabled:
                    subscribe_to_save_mp4(EmptyRequest())
                if rollout_idx == 0:
                    for _ in range(MIN_EVAL_TRIALS):
                        graph_manager.evaluate(EnvironmentSteps(1))

                while new_checkpoint < last_checkpoint + 1:
                    exit_if_trainer_done(checkpoint_dir, simtrace_video_s3_writers, rollout_idx)
                    if rollout_idx == 0:
                        graph_manager.evaluate(EnvironmentSteps(1))
                    new_checkpoint = data_store.get_coach_checkpoint_number("agent")
                if is_save_mp4_enabled:
                    unsubscribe_from_save_mp4(EmptyRequest())
                # upload simtrace and mp4 into s3 bucket
                for s3_writer in simtrace_video_s3_writers:
                    s3_writer.persist(utils.get_s3_kms_extra_args())
                pause_physics(EmptyRequest())
                data_store.load_from_store(expected_checkpoint_number=last_checkpoint + 1)
                graph_manager.restore_checkpoint()

            if (
                graph_manager.agent_params.algorithm.distributed_coach_synchronization_type
                == DistributedCoachSynchronizationType.ASYNC
            ):
                if new_checkpoint > last_checkpoint:
                    graph_manager.restore_checkpoint()

            last_checkpoint = new_checkpoint


def main():
    screen.set_use_colors(False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        help="(string) Path to a folder containing a checkpoint to restore the model from.",
        type=str,
        default="./checkpoint",
    )
    parser.add_argument(
        "--s3_bucket",
        help="(string) S3 bucket",
        type=str,
        default=rospy.get_param("SAGEMAKER_SHARED_S3_BUCKET", "gsaur-test"),
    )
    parser.add_argument(
        "--s3_prefix",
        help="(string) S3 prefix",
        type=str,
        default=rospy.get_param("SAGEMAKER_SHARED_S3_PREFIX", "sagemaker"),
    )
    parser.add_argument(
        "--num_workers",
        help="(int) The number of workers started in this pool",
        type=int,
        default=int(rospy.get_param("NUM_WORKERS", 1)),
    )
    parser.add_argument(
        "--rollout_idx", help="(int) The index of current rollout worker", type=int, default=0
    )
    parser.add_argument(
        "-r",
        "--redis_ip",
        help="(string) IP or host for the redis server",
        default="localhost",
        type=str,
    )
    parser.add_argument(
        "-rp", "--redis_port", help="(int) Port of the redis server", default=6379, type=int
    )
    parser.add_argument(
        "--aws_region",
        help="(string) AWS region",
        type=str,
        default=rospy.get_param("AWS_REGION", "us-east-1"),
    )
    parser.add_argument(
        "--reward_file_s3_key",
        help="(string) Reward File S3 Key",
        type=str,
        default=rospy.get_param("REWARD_FILE_S3_KEY", None),
    )
    parser.add_argument(
        "--model_metadata_s3_key",
        help="(string) Model Metadata File S3 Key",
        type=str,
        default=rospy.get_param("MODEL_METADATA_FILE_S3_KEY", None),
    )
    # For training job, reset is not allowed. penalty_seconds, off_track_penalty, and
    # collision_penalty will all be 0 be default
    parser.add_argument(
        "--number_of_resets",
        help="(integer) Number of resets",
        type=int,
        default=int(rospy.get_param("NUMBER_OF_RESETS", 0)),
    )
    parser.add_argument(
        "--penalty_seconds",
        help="(float) penalty second",
        type=float,
        default=float(rospy.get_param("PENALTY_SECONDS", 0.0)),
    )
    parser.add_argument(
        "--job_type",
        help="(string) job type",
        type=str,
        default=rospy.get_param("JOB_TYPE", "TRAINING"),
    )
    parser.add_argument(
        "--is_continuous",
        help="(boolean) is continous after lap completion",
        type=bool,
        default=utils.str2bool(rospy.get_param("IS_CONTINUOUS", False)),
    )
    parser.add_argument(
        "--race_type",
        help="(string) Race type",
        type=str,
        default=rospy.get_param("RACE_TYPE", "TIME_TRIAL"),
    )
    parser.add_argument(
        "--off_track_penalty",
        help="(float) off track penalty second",
        type=float,
        default=float(rospy.get_param("OFF_TRACK_PENALTY", 0.0)),
    )
    parser.add_argument(
        "--collision_penalty",
        help="(float) collision penalty second",
        type=float,
        default=float(rospy.get_param("COLLISION_PENALTY", 0.0)),
    )

    args = parser.parse_args()

    logger.info("S3 bucket: %s", args.s3_bucket)
    logger.info("S3 prefix: %s", args.s3_prefix)

    # Download and import reward function
    # TODO: replace 'agent' with name of each agent for multi-agent training
    reward_function_file = RewardFunction(
        bucket=args.s3_bucket,
        s3_key=args.reward_file_s3_key,
        region_name=args.aws_region,
        local_path=REWARD_FUCTION_LOCAL_PATH_FORMAT.format("agent"),
    )
    reward_function = reward_function_file.get_reward_function()

    # Instantiate Cameras
    configure_camera(namespaces=["racecar"])

    preset_file_success, _ = download_custom_files_if_present(
        s3_bucket=args.s3_bucket, s3_prefix=args.s3_prefix, aws_region=args.aws_region
    )

    # download model metadata
    # TODO: replace 'agent' with name of each agent
    model_metadata = ModelMetadata(
        bucket=args.s3_bucket,
        s3_key=args.model_metadata_s3_key,
        region_name=args.aws_region,
        local_path=MODEL_METADATA_LOCAL_PATH_FORMAT.format("agent"),
    )
    model_metadata_info = model_metadata.get_model_metadata_info()
    version = model_metadata_info[ModelMetadataKeys.VERSION.value]

    agent_config = {
        "model_metadata": model_metadata,
        ConfigParams.CAR_CTRL_CONFIG.value: {
            ConfigParams.LINK_NAME_LIST.value: LINK_NAMES,
            ConfigParams.VELOCITY_LIST.value: VELOCITY_TOPICS,
            ConfigParams.STEERING_LIST.value: STEERING_TOPICS,
            ConfigParams.CHANGE_START.value: utils.str2bool(
                rospy.get_param("CHANGE_START_POSITION", True)
            ),
            ConfigParams.ALT_DIR.value: utils.str2bool(
                rospy.get_param("ALTERNATE_DRIVING_DIRECTION", False)
            ),
            ConfigParams.MODEL_METADATA.value: model_metadata,
            ConfigParams.REWARD.value: reward_function,
            ConfigParams.AGENT_NAME.value: "racecar",
            ConfigParams.VERSION.value: version,
            ConfigParams.NUMBER_OF_RESETS.value: args.number_of_resets,
            ConfigParams.PENALTY_SECONDS.value: args.penalty_seconds,
            ConfigParams.NUMBER_OF_TRIALS.value: None,
            ConfigParams.IS_CONTINUOUS.value: args.is_continuous,
            ConfigParams.RACE_TYPE.value: args.race_type,
            ConfigParams.COLLISION_PENALTY.value: args.collision_penalty,
            ConfigParams.OFF_TRACK_PENALTY.value: args.off_track_penalty,
        },
    }

    #! TODO each agent should have own s3 bucket
    metrics_key = rospy.get_param("METRICS_S3_OBJECT_KEY")
    if args.num_workers > 1 and args.rollout_idx > 0:
        key_tuple = os.path.splitext(metrics_key)
        metrics_key = "{}_{}{}".format(key_tuple[0], str(args.rollout_idx), key_tuple[1])
    metrics_s3_config = {
        MetricsS3Keys.METRICS_BUCKET.value: rospy.get_param("METRICS_S3_BUCKET"),
        MetricsS3Keys.METRICS_KEY.value: metrics_key,
        MetricsS3Keys.REGION.value: rospy.get_param("AWS_REGION"),
    }

    run_phase_subject = RunPhaseSubject()

    agent_list = list()

    # TODO: replace agent for multi agent training
    # checkpoint s3 instance
    # TODO replace agent with agent_0 and so on for multiagent case
    checkpoint = Checkpoint(
        bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        region_name=args.aws_region,
        agent_name="agent",
        checkpoint_dir=args.checkpoint_dir,
    )

    agent_list.append(
        create_rollout_agent(
            agent_config,
            TrainingMetrics(
                agent_name="agent",
                s3_dict_metrics=metrics_s3_config,
                deepracer_checkpoint_json=checkpoint.deepracer_checkpoint_json,
                ckpnt_dir=os.path.join(args.checkpoint_dir, "agent"),
                run_phase_sink=run_phase_subject,
                use_model_picker=(args.rollout_idx == 0),
            ),
            run_phase_subject,
        )
    )
    agent_list.append(create_obstacles_agent())
    agent_list.append(create_bot_cars_agent())
    # ROS service to indicate all the robomaker markov packages are ready for consumption
    signal_robomaker_markov_package_ready()

    PhaseObserver("/agent/training_phase", run_phase_subject)

    aws_region = rospy.get_param("AWS_REGION", args.aws_region)
    simtrace_s3_bucket = rospy.get_param("SIMTRACE_S3_BUCKET", None)
    mp4_s3_bucket = rospy.get_param("MP4_S3_BUCKET", None) if args.rollout_idx == 0 else None
    if simtrace_s3_bucket:
        simtrace_s3_object_prefix = rospy.get_param("SIMTRACE_S3_PREFIX")
        if args.num_workers > 1:
            simtrace_s3_object_prefix = os.path.join(
                simtrace_s3_object_prefix, str(args.rollout_idx)
            )
    if mp4_s3_bucket:
        mp4_s3_object_prefix = rospy.get_param("MP4_S3_OBJECT_PREFIX")

    simtrace_video_s3_writers = []
    # TODO: replace 'agent' with 'agent_0' for multi agent training and
    # mp4_s3_object_prefix, mp4_s3_bucket will be a list, so need to access with index
    if simtrace_s3_bucket:
        simtrace_video_s3_writers.append(
            SimtraceVideo(
                upload_type=SimtraceVideoNames.SIMTRACE_TRAINING.value,
                bucket=simtrace_s3_bucket,
                s3_prefix=simtrace_s3_object_prefix,
                region_name=aws_region,
                local_path=SIMTRACE_TRAINING_LOCAL_PATH_FORMAT.format("agent"),
            )
        )
    if mp4_s3_bucket:
        simtrace_video_s3_writers.extend(
            [
                SimtraceVideo(
                    upload_type=SimtraceVideoNames.PIP.value,
                    bucket=mp4_s3_bucket,
                    s3_prefix=mp4_s3_object_prefix,
                    region_name=aws_region,
                    local_path=CAMERA_PIP_MP4_LOCAL_PATH_FORMAT.format("agent"),
                ),
                SimtraceVideo(
                    upload_type=SimtraceVideoNames.DEGREE45.value,
                    bucket=mp4_s3_bucket,
                    s3_prefix=mp4_s3_object_prefix,
                    region_name=aws_region,
                    local_path=CAMERA_45DEGREE_LOCAL_PATH_FORMAT.format("agent"),
                ),
                SimtraceVideo(
                    upload_type=SimtraceVideoNames.TOPVIEW.value,
                    bucket=mp4_s3_bucket,
                    s3_prefix=mp4_s3_object_prefix,
                    region_name=aws_region,
                    local_path=CAMERA_TOPVIEW_LOCAL_PATH_FORMAT.format("agent"),
                ),
            ]
        )

    # TODO: replace 'agent' with specific agent name for multi agent training
    ip_config = IpConfig(
        bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        region_name=args.aws_region,
        local_path=IP_ADDRESS_LOCAL_PATH.format("agent"),
    )
    redis_ip = ip_config.get_ip_config()

    # Download hyperparameters from SageMaker shared s3 bucket
    # TODO: replace 'agent' with name of each agent
    hyperparameters = Hyperparameters(
        bucket=args.s3_bucket,
        s3_key=get_s3_key(args.s3_prefix, HYPERPARAMETER_S3_POSTFIX),
        region_name=args.aws_region,
        local_path=HYPERPARAMETER_LOCAL_PATH_FORMAT.format("agent"),
    )
    sm_hyperparams_dict = hyperparameters.get_hyperparameters_dict()

    enable_domain_randomization = utils.str2bool(
        rospy.get_param("ENABLE_DOMAIN_RANDOMIZATION", False)
    )
    # Make the clients that will allow us to pause and unpause the physics
    rospy.wait_for_service("/gazebo/pause_physics_dr")
    rospy.wait_for_service("/gazebo/unpause_physics_dr")
    pause_physics = ServiceProxyWrapper("/gazebo/pause_physics_dr", Empty)
    unpause_physics = ServiceProxyWrapper("/gazebo/unpause_physics_dr", Empty)

    if preset_file_success:
        preset_location = os.path.join(CUSTOM_FILES_PATH, "preset.py")
        preset_location += ":graph_manager"
        graph_manager = short_dynamic_import(preset_location, ignore_module_case=True)
        logger.info("Using custom preset file!")
    else:
        graph_manager, _ = get_graph_manager(
            hp_dict=sm_hyperparams_dict,
            agent_list=agent_list,
            run_phase_subject=run_phase_subject,
            enable_domain_randomization=enable_domain_randomization,
            pause_physics=pause_physics,
            unpause_physics=unpause_physics,
        )

    # If num_episodes_between_training is smaller than num_workers then cancel worker early.
    episode_steps_per_rollout = (
        graph_manager.agent_params.algorithm.num_consecutive_playing_steps.num_steps
    )
    # Reduce number of workers if allocated more than num_episodes_between_training
    if args.num_workers > episode_steps_per_rollout:
        logger.info(
            "Excess worker allocated. Reducing from {} to {}...".format(
                args.num_workers, episode_steps_per_rollout
            )
        )
        args.num_workers = episode_steps_per_rollout
    if args.rollout_idx >= episode_steps_per_rollout or args.rollout_idx >= args.num_workers:
        err_msg_format = "Exiting excess worker..."
        err_msg_format += (
            "(rollout_idx[{}] >= num_workers[{}] or num_episodes_between_training[{}])"
        )
        logger.info(
            err_msg_format.format(args.rollout_idx, args.num_workers, episode_steps_per_rollout)
        )
        # Close the down the job
        utils.cancel_simulation_job()

    memory_backend_params = DeepRacerRedisPubSubMemoryBackendParameters(
        redis_address=redis_ip,
        redis_port=6379,
        run_type=str(RunType.ROLLOUT_WORKER),
        channel=args.s3_prefix,
        num_workers=args.num_workers,
        rollout_idx=args.rollout_idx,
    )

    graph_manager.memory_backend_params = memory_backend_params

    checkpoint_dict = {"agent": checkpoint}
    ds_params_instance = S3BotoDataStoreParameters(checkpoint_dict=checkpoint_dict)

    graph_manager.data_store = S3BotoDataStore(ds_params_instance, graph_manager)

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = args.checkpoint_dir

    rollout_worker(
        graph_manager=graph_manager,
        num_workers=args.num_workers,
        rollout_idx=args.rollout_idx,
        task_parameters=task_parameters,
        simtrace_video_s3_writers=simtrace_video_s3_writers,
        pause_physics=pause_physics,
        unpause_physics=unpause_physics,
    )


if __name__ == "__main__":
    try:
        rospy.init_node("rl_coach", anonymous=True)
        main()
    except ValueError as err:
        if utils.is_user_error(err):
            log_and_exit(
                "User modified model/model_metadata: {}".format(err),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        else:
            log_and_exit(
                "Rollout worker value error: {}".format(err),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
    except GenericRolloutError as ex:
        ex.log_except_and_exit()
    except GenericRolloutException as ex:
        ex.log_except_and_exit()
    except Exception as ex:
        log_and_exit(
            "Rollout worker exited with exception: {}".format(ex),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
